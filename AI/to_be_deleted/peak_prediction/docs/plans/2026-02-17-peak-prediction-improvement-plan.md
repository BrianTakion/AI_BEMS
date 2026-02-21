# Peak Prediction Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix target leakage, improve feature engineering, and add proper time-series validation to bring peak prediction within Power RMSE <15 kW and Time error <45 min.

**Architecture:** Dual LightGBM models (power + time) with ~57 features per day, trained on ~180 days of 15-min power data. Fix same-day feature leak by randomizing hours_elapsed during training. Replace random split with chronological split + TimeSeriesSplit CV. Add 17 new features (quantile, time distribution, interaction, ramp). Separate hyperparameters per model.

**Tech Stack:** Python 3.8+, LightGBM, pandas, numpy, scikit-learn, holidays

---

### Task 1: Update `_config.json` — Split Model Config

**Files:**
- Modify: `peak_prediction/_config.json`

**Step 1: Edit the config file**

Replace the single `"model"` key with `"model_power"` and `"model_time"` keys. Keep the old `"model"` key for backward compatibility.

```json
{
  "data_source": "csv",
  "csv": {
    "data_path": "/workspace/2024_AI_BEMS/YiUmGoV2/dataset/data_colec_h_250730_260212_B0019.csv",
    "config_peak_devices_path": "config_peak_devices.csv",
    "peak_results_path": "output/peak_results.csv"
  },
  "db": {
    "host": "localhost",
    "port": 5432,
    "database": "bems",
    "user": "ai_user",
    "password": "changeme"
  },
  "data": {
    "fetch_window_hours": 336,
    "sampling_minutes": 15,
    "collection_table": "DATA_COLEC_H",
    "tag_cd": 30001
  },
  "training": {
    "min_history_days": 14,
    "test_size": 0.15
  },
  "peak": {
    "dev_id": 2001,
    "model_dir": "models",
    "config_table": "DEV_USE_PURP_REL_R",
    "result_table": "MAX_DMAND_FCST_H"
  },
  "model_power": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 20,
    "min_child_samples": 10,
    "num_boost_round": 1000,
    "early_stopping_rounds": 50,
    "verbose": -1
  },
  "model_time": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 15,
    "num_boost_round": 1000,
    "early_stopping_rounds": 50,
    "verbose": -1
  }
}
```

**Step 2: Verify JSON is valid**

Run: `python -c "import json; json.load(open('peak_prediction/_config.json'))"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add peak_prediction/_config.json
git commit -m "feat(peak_prediction): split model config into model_power and model_time

Separate hyperparameters: lower learning rate (0.05), more rounds (1000),
more regularization (fewer leaves, min_child_samples) for ~180-sample dataset."
```

---

### Task 2: Fix Same-Day Feature Leak in `data_preprocessing.py`

**Files:**
- Modify: `peak_prediction/data_preprocessing.py:128-188` (preprocess_for_training function)
- Modify: `peak_prediction/data_preprocessing.py:1-23` (module docstring — update feature count)

This is the most critical fix. The current code passes `hours_elapsed=24` and `today_raw_15min=None` during training, which leaks the actual daily peak into `today_max_so_far`.

**Step 1: Update the module docstring**

At line 12, change `~40 total` to `~57 total` and add the new feature categories to the docstring list. Also update line 3 reference from `~40` to `~57`.

**Step 2: Modify `preprocess_for_training()` at line 128**

Replace the function body (lines 128-188) with this implementation:

```python
def preprocess_for_training(df_raw, config):
    """Build training dataset: daily feature vectors + targets from raw 15-min data.

    To prevent target leakage, each training sample gets a random hours_elapsed
    value in [0, 23], simulating the partial-day data available at inference time.
    The raw 15-min data is passed through so same-day features use actual 15-min
    resolution (matching inference behavior).

    Args:
        df_raw: DataFrame with columns ['colec_dt', 'colec_val'] at 15-min intervals.
        config: Configuration dict.

    Returns:
        Tuple (X_df, y_power, y_slot):
          - X_df: DataFrame of ~57 features per day
          - y_power: Series of daily peak power values
          - y_slot: Series of daily peak slot indices (0-95)
    """
    min_history_days = config["training"]["min_history_days"]  # 14

    # Build a value-indexed time series
    df = pd.DataFrame({
        "value": df_raw["colec_val"].values,
    }, index=pd.to_datetime(df_raw["colec_dt"]))
    df.index = df.index.round("15min")
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("15min").asfreq()
    df["value"] = df["value"].interpolate(method="time")
    df["value"] = df["value"].ffill().bfill()

    # Extract daily peaks from 15-min data
    daily_peaks = _extract_daily_peaks(df)
    if len(daily_peaks) < min_history_days + 1:
        raise ValueError(
            f"Need at least {min_history_days + 1} days of data, "
            f"got {len(daily_peaks)}"
        )

    # Resample to hourly for feature construction
    hourly = _resample_to_hourly(df)
    hourly["value"] = hourly["value"].interpolate(method="time").ffill().bfill()

    # Build features for each day (skip first min_history_days for lag availability)
    all_dates = daily_peaks.index.sort_values()
    year_range = range(all_dates[0].year, all_dates[-1].year + 1)
    holiday_set = _get_korean_holidays(year_range)

    rng = np.random.RandomState(42)
    feature_rows = []
    target_power = []
    target_slot = []

    for i in range(min_history_days, len(all_dates)):
        date = all_dates[i]
        # Randomize hours_elapsed to prevent same-day feature leakage
        hours_elapsed = int(rng.randint(0, 24))  # 0 to 23 inclusive
        features = _build_features_for_date(
            date, daily_peaks, hourly, holiday_set,
            hours_elapsed=hours_elapsed, today_raw_15min=df
        )
        if features is not None:
            feature_rows.append(features)
            target_power.append(daily_peaks.loc[date, "peak_power"])
            target_slot.append(daily_peaks.loc[date, "peak_slot"])

    X_df = pd.DataFrame(feature_rows)
    y_power = pd.Series(target_power, name="peak_power")
    y_slot = pd.Series(target_slot, name="peak_slot")

    return X_df, y_power, y_slot
```

Key changes:
- `rng = np.random.RandomState(42)` for reproducibility
- `hours_elapsed = int(rng.randint(0, 24))` per sample (0-23 inclusive)
- `today_raw_15min=df` passed through so same-day features use 15-min resolution

**Step 3: Verify the fix doesn't break imports**

Run: `cd /workspace/2024_AI_BEMS/YiUmGoV2 && python -c "import peak_prediction.data_preprocessing as DP; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add peak_prediction/data_preprocessing.py
git commit -m "fix(peak_prediction): eliminate same-day feature target leakage

Randomize hours_elapsed per training sample [0,23] and pass raw 15-min data
through to _build_features_for_date(). Same-day features now see only partial
data during training, matching inference conditions exactly."
```

---

### Task 3: Add 17 New Features in `data_preprocessing.py`

**Files:**
- Modify: `peak_prediction/data_preprocessing.py:296-334` (after rolling peak stats 7d, before section 6)
- Modify: `peak_prediction/data_preprocessing.py:413-438` (trend features section 10)

**Step 1: Add quantile features after section 4 (rolling peak stats 7d)**

After the existing `mean_peak_slot_7d` assignment (around line 311), add:

```python
    # --- Quantile features (4) ---
    features["peak_75pct_7d"] = np.percentile(past_7d, 75)
    features["peak_90pct_7d"] = np.percentile(past_7d, 90)
```

**Step 2: Add 14d quantile features after section 5**

After `features["trend_peak_14d"]` (around line 334), add:

```python
    features["peak_75pct_14d"] = np.percentile(past_14d, 75)
    features["peak_90pct_14d"] = np.percentile(past_14d, 90)
```

**Step 3: Add peak time distribution features after section 4**

After the quantile 7d features added in step 1, add:

```python
    # --- Peak time distribution features (4) ---
    features["peak_slot_std_7d"] = np.std(past_7d_slots) if len(past_7d_slots) >= 3 else 0.0
    features["peak_slot_median_7d"] = np.median(past_7d_slots)

    # Most common peak period (0=night 0-6, 1=morning 6-12, 2=afternoon 12-18, 3=evening 18-24)
    slot_periods = [s // 24 for s in past_7d_slots]  # 96 slots / 4 periods = 24 slots per period
    if slot_periods:
        from collections import Counter
        features["most_common_peak_period"] = Counter(slot_periods).most_common(1)[0][0]
    else:
        features["most_common_peak_period"] = 2  # default afternoon

    # Same weekday peak slot over last 2 weeks
    same_wd_slots = []
    for w in [7, 14]:
        wd_date = d - pd.Timedelta(days=w)
        if wd_date in daily_peaks.index:
            same_wd_slots.append(daily_peaks.loc[wd_date, "peak_slot"])
    features["same_weekday_peak_slot_2w"] = np.mean(same_wd_slots) if same_wd_slots else features.get("mean_peak_slot_7d", 0.0)
```

**Step 4: Add day-type interaction features in section 10 (trend features)**

Replace the existing section 10 (lines 413-438) with an expanded version. Keep the existing `peak_trend_7d` and `weekday_weekend_peak_ratio`, and add after them:

```python
    # --- Day-type interaction features (3) ---
    features["is_weekday_x_mean_peak_7d"] = (1 if d.weekday() < 5 else 0) * features["mean_peak_7d"]

    prev_day_date = d - pd.Timedelta(days=1)
    features["prev_day_was_holiday"] = 1 if (prev_day_date.date() in holiday_set or prev_day_date.weekday() >= 5) else 0

    # Consecutive workdays before this date
    consec = 0
    for back in range(1, 15):
        check_date = d - pd.Timedelta(days=back)
        if check_date.date() in holiday_set or check_date.weekday() >= 5:
            break
        consec += 1
    features["consecutive_workdays"] = consec
```

**Step 5: Add load ramp features after the compressed profile section 7**

After `features["prev_day_evening_mean"]` (line 365), add:

```python
    # --- Load ramp features (2) ---
    features["prev_day_morning_ramp"] = profile["morning_mean"] - profile["night_mean"]
    prev_day_mean_val = features.get("prev_day_mean", 0.0)
    features["prev_day_peak_to_mean_ratio"] = (
        features["prev_1d_peak_power"] / prev_day_mean_val
        if prev_day_mean_val > 0.01 else 0.0
    )
```

**Step 6: Move `from collections import Counter` to top of file**

Add `from collections import Counter` to the imports at line 28 (after `import datetime`), and remove the inline import from step 3.

**Step 7: Verify import still works**

Run: `cd /workspace/2024_AI_BEMS/YiUmGoV2 && python -c "import peak_prediction.data_preprocessing as DP; print('OK')"`
Expected: `OK`

**Step 8: Commit**

```bash
git add peak_prediction/data_preprocessing.py
git commit -m "feat(peak_prediction): add 17 new features for better peak prediction

Add quantile (4), peak time distribution (4), day-type interaction (3),
and load ramp (2) features. Total features: ~57."
```

---

### Task 4: Update `train_peak.py` — Chronological Split + Time-Series CV + Per-Model Config

**Files:**
- Modify: `peak_prediction/train_peak.py`

**Step 1: Update imports**

At line 25, replace:
```python
from sklearn.model_selection import train_test_split
```
with:
```python
from sklearn.model_selection import TimeSeriesSplit
```

**Step 2: Add helper function `_get_model_config`**

Add after line 33 (after `import data_source as DS`):

```python
def _get_model_config(config, target):
    """Get model config for a specific target, falling back to shared 'model' key.

    Args:
        config: Full configuration dict.
        target: 'power' or 'time'.

    Returns:
        Dict of LightGBM parameters.
    """
    key = f"model_{target}"
    if key in config:
        return config[key]
    return config["model"]
```

**Step 3: Add `cross_validate_model` function**

Add after the new `_get_model_config` function:

```python
def cross_validate_model(X, y, model_cfg, n_splits=5, target_name=""):
    """Run time-series cross-validation and report metrics.

    Uses expanding window: each fold trains on all data up to split point,
    validates on the next chunk.

    Args:
        X: Feature DataFrame (sorted chronologically).
        y: Target Series.
        model_cfg: LightGBM parameter dict.
        n_splits: Number of CV folds.
        target_name: Label for logging.

    Returns:
        Dict with 'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std'.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes = [], []

    params = {
        "objective": model_cfg["objective"],
        "metric": model_cfg["metric"],
        "boosting_type": model_cfg["boosting_type"],
        "learning_rate": model_cfg["learning_rate"],
        "num_leaves": model_cfg["num_leaves"],
        "verbose": model_cfg["verbose"],
    }
    if "min_child_samples" in model_cfg:
        params["min_child_samples"] = model_cfg["min_child_samples"]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params, train_data,
            valid_sets=[valid_data], valid_names=["validation"],
            num_boost_round=model_cfg["num_boost_round"],
            callbacks=[lgb.early_stopping(stopping_rounds=model_cfg["early_stopping_rounds"])],
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmses.append(root_mean_squared_error(y_val, y_pred))
        maes.append(mean_absolute_error(y_val, y_pred))

    cv_metrics = {
        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
        "mae_mean": np.mean(maes), "mae_std": np.std(maes),
    }
    print(f"  [CV-{target_name}] RMSE={cv_metrics['rmse_mean']:.4f} +/- {cv_metrics['rmse_std']:.4f} | "
          f"MAE={cv_metrics['mae_mean']:.4f} +/- {cv_metrics['mae_std']:.4f}")
    return cv_metrics
```

**Step 4: Update `main()` — replace random split with chronological split**

Replace lines 99-163 in `main()`. The key changes:

1. Replace `model_cfg = config["model"]` with per-model config lookup
2. Replace `train_test_split(shuffle=True)` with chronological split
3. Add cross-validation before final training
4. Use separate configs for power and time models

```python
    power_model_cfg = _get_model_config(config, "power")
    time_model_cfg = _get_model_config(config, "time")
    model_dir = os.path.join(SCRIPT_DIR, config["peak"]["model_dir"])
    os.makedirs(model_dir, exist_ok=True)
```

For the split section (replacing lines 142-149):
```python
        # 6. Chronological train/test split (no shuffle for time series)
        test_size = config["training"]["test_size"]
        n_test = max(1, int(len(X_df) * test_size))
        n_train = len(X_df) - n_test

        X_train = X_df.iloc[:n_train]
        X_test = X_df.iloc[n_train:]
        y_pow_train = y_power.iloc[:n_train]
        y_pow_test = y_power.iloc[n_train:]
        y_slot_train = y_slot.iloc[:n_train]
        y_slot_test = y_slot.iloc[n_train:]
```

For CV + training (replacing lines 152-163):
```python
        # 7. Cross-validation on training set
        print(f"\n--- Cross-Validation (5-fold time-series) ---")
        power_cv = cross_validate_model(X_train, y_pow_train, power_model_cfg, target_name="POWER")
        time_cv = cross_validate_model(X_train, y_slot_train, time_model_cfg, target_name="TIME")

        # 8. Train final power model on full training set
        print(f"\n--- Training final POWER model ---")
        power_model, power_metrics = train_model(
            X_train, y_pow_train, X_test, y_pow_test,
            power_model_cfg, power_model_path, "POWER"
        )

        # 9. Train final time model on full training set
        print(f"\n--- Training final TIME model ---")
        time_model, time_metrics = train_model(
            X_train, y_slot_train, X_test, y_slot_test,
            time_model_cfg, time_model_path, "TIME"
        )
```

**Step 5: Update `train_model()` to handle `min_child_samples`**

In the `train_model` function, after line 51 (`"verbose": model_cfg["verbose"]`), add:

```python
    if "min_child_samples" in model_cfg:
        params["min_child_samples"] = model_cfg["min_child_samples"]
```

**Step 6: Update the summary section to include CV results**

After the existing summary (line 172+), add CV metrics:

```python
        print(f"  --- Cross-Validation (5-fold) ---")
        print(f"  Power RMSE (CV)     : {power_cv['rmse_mean']:.4f} +/- {power_cv['rmse_std']:.4f}")
        print(f"  Time RMSE (CV)      : {time_cv['rmse_mean']:.4f} +/- {time_cv['rmse_std']:.4f}")
```

**Step 7: Verify training still works end-to-end**

First delete the old models so training runs fresh:

```bash
rm -f peak_prediction/models/2001_power.txt peak_prediction/models/2001_time.txt
```

Then run training:

```bash
cd /workspace/2024_AI_BEMS/YiUmGoV2 && python peak_prediction/train_peak.py --csv
```

Expected: Training completes without errors. Metrics will differ from before (likely worse in test metrics since the leak is fixed, but more honest). Look for:
- No Python errors
- Both models saved
- CV and test metrics printed
- Feature count should be ~57

**Step 8: Commit**

```bash
git add peak_prediction/train_peak.py
git commit -m "feat(peak_prediction): chronological split, time-series CV, per-model config

Replace random shuffle with chronological train/test split. Add 5-fold
TimeSeriesSplit cross-validation for reliable metrics. Use separate
model_power/model_time configs from _config.json."
```

---

### Task 5: Retrain, Validate, and Update README

**Files:**
- Run: `peak_prediction/train_peak.py --csv`
- Modify: `peak_prediction/README.md` (update metrics)

**Step 1: Delete old models and retrain**

```bash
rm -f peak_prediction/models/2001_power.txt peak_prediction/models/2001_time.txt
cd /workspace/2024_AI_BEMS/YiUmGoV2 && python peak_prediction/train_peak.py --csv
```

Capture the full output. Record:
- Power RMSE (test) and MAE (test)
- Time RMSE (test) and MAE in minutes
- CV RMSE mean +/- std for both models
- Feature count
- Best iterations

**Step 2: Run inference to verify end-to-end**

```bash
cd /workspace/2024_AI_BEMS/YiUmGoV2 && python peak_prediction/ai_peak_runner.py --csv
```

Expected: Inference completes, writes to `output/peak_results.csv`. The prediction values should change from before (reflecting the new models).

**Step 3: Update README with new metrics**

Edit `peak_prediction/README.md` to reflect the new training results:
- Update feature count from ~40 to ~57
- Update RMSE/MAE numbers
- Add a note about time-series CV metrics
- Mention the data leak fix

**Step 4: Commit**

```bash
git add peak_prediction/models/ peak_prediction/output/ peak_prediction/README.md
git commit -m "feat(peak_prediction): retrain models with leak fix and new features

Updated metrics after fixing same-day feature leakage, adding 17 new features,
chronological split, and per-model hyperparameters."
```

---

### Task 6: Assess Results Against Targets

**No files to modify — evaluation only.**

**Step 1: Check if targets are met**

Compare the training output against targets:
- Power RMSE < 15 kW?
- Time error < 45 minutes?

**Step 2: If targets are NOT met**

Consider these additional improvements (in priority order):
1. Increase `num_boost_round` to 2000 with lower `learning_rate` (0.02)
2. Add L1/L2 regularization (`reg_alpha`, `reg_lambda`) to config
3. Try `min_child_samples=20` to reduce overfitting further
4. Check feature importance and drop features with near-zero importance

**Step 3: If targets ARE met**

The implementation is complete. Update the design doc with final results.

---

## File Change Summary

| File | Task | Type of Change |
|---|---|---|
| `_config.json` | 1 | Replace `model` with `model_power` + `model_time` |
| `data_preprocessing.py` | 2, 3 | Fix leak (randomize hours_elapsed), add 17 features |
| `train_peak.py` | 4 | Chronological split, CV, per-model config, min_child_samples |
| `README.md` | 5 | Update metrics and feature count |
| `models/2001_power.txt` | 5 | Retrained model binary |
| `models/2001_time.txt` | 5 | Retrained model binary |

## Dependency Order

```
Task 1 (_config.json) ──┐
                         ├──> Task 4 (train_peak.py) ──> Task 5 (retrain) ──> Task 6 (assess)
Task 2+3 (data_preprocessing.py) ─┘
```

Tasks 1 and 2+3 can be done in parallel. Task 4 depends on both. Task 5 depends on Task 4. Task 6 depends on Task 5.
