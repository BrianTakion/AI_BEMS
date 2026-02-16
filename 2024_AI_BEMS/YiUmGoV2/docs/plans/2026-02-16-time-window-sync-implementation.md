# Time-Window Synchronization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Synchronize training and inference so both use identical 176h windows through `preprocess()`, ensuring feature engineering consistency.

**Architecture:** Random 176h window sampling for training (CSV-only, 1000 steps, 8 samples/window). Inference uses TIME-based 176h window (CSV or DB). Both paths call `preprocess(window_df, config)` identically.

**Tech Stack:** Python, pandas, LightGBM, numpy

**Design doc:** `docs/plans/2026-02-16-time-window-sync-design.md`

---

### Task 1: Update _config.json

**Files:**
- Modify: `_config.json`

**Step 1: Edit config**

Change `scoring_window_hours` from 4 to 2, and add `training` section:

```json
{
  "data_source": "csv",
  "csv": {
    "data_path": "/workspace/2024_AI_BEMS/YiUmGoV2/dataset/data_colec_h_250730_260212_B0019.csv"
  },
  "db": {
    "host": "localhost",
    "port": 5432,
    "database": "bems",
    "user": "ai_user",
    "password": "changeme"
  },
  "data": {
    "fetch_window_hours": 176,
    "scoring_window_hours": 2,
    "sampling_minutes": 15,
    "collection_table": "DATA_COLEC_H",
    "tag_cd": 30001
  },
  "training": {
    "max_steps": 1000,
    "samples_per_window": 8
  },
  "anomaly": {
    "score_threshold": 50,
    "model_dir": "models/anomaly",
    "config_table": "DEV_USE_PURP_REL_R",
    "result_table": "FALT_PRCV_FCST"
  },
  "model": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 50,
    "learning_rate": 0.1,
    "num_boost_round": 1000,
    "early_stopping_rounds": 30,
    "test_size": 0.03,
    "verbose": -1
  }
}
```

**Step 2: Commit**

```bash
git add _config.json
git commit -m "config: add training section, change scoring_window_hours to 2"
```

---

### Task 2: Rename raw_df to window_df in data_preprocessing.py

**Files:**
- Modify: `data_preprocessing.py:12,16`

**Step 1: Rename parameter and internal reference**

At line 12, change function signature:

```python
# Before
def preprocess(raw_df, config, only_cleansing=False, fill_method='zero'):

# After
def preprocess(window_df, config, only_cleansing=False, fill_method='zero'):
```

At line 16, change the copy:

```python
# Before
    df = raw_df.copy()

# After
    df = window_df.copy()
```

These are the only two references to `raw_df` in the file.

**Step 2: Verify no other references**

Run: `grep -n "raw_df" data_preprocessing.py`
Expected: no matches

**Step 3: Commit**

```bash
git add data_preprocessing.py
git commit -m "refactor: rename raw_df to window_df in preprocess()"
```

---

### Task 3: Fix db_connection.py CSV time-based filtering

**Files:**
- Modify: `db_connection.py:193-194`

**Step 1: Replace row-count filter with time-based filter**

In `read_sensor_data()`, at lines 193-194, replace:

```python
    # Before (row-count based -- inconsistent with DB mode)
    # Keep only the tail -- DB 모드와 동일하게 fetch_window_hours 기반으로 제한
    df = df.tail(max_tail_rows).reset_index(drop=True)
```

With:

```python
    # After (time-based -- consistent with DB mode)
    # Keep only the last fetch_window_hours by TIME (consistent with DB mode cutoff)
    cutoff = df["colec_dt"].iloc[-1] - timedelta(hours=fetch_hours)
    df = df[df["colec_dt"] >= cutoff].reset_index(drop=True)
```

Also remove the now-unused `max_tail_rows` variable at line 167:

```python
    # Before
    sampling_min = config["data"]["sampling_minutes"]
    max_tail_rows = (60 // sampling_min) * fetch_hours  # DB 모드와 동일한 fetch_window_hours 적용

    # After
    # (remove sampling_min and max_tail_rows -- no longer needed here)
```

Update the log message at line 197:

```python
    # Before
    logger.info("CSV: returning %d rows (last %d, %dh window) for dev_id=%s", len(df), max_tail_rows, fetch_hours, dev_id)

    # After
    logger.info("CSV: returning %d rows (%dh window) for dev_id=%s", len(df), fetch_hours, dev_id)
```

**Step 2: Verify no remaining references to max_tail_rows**

Run: `grep -n "max_tail_rows" db_connection.py`
Expected: no matches

**Step 3: Commit**

```bash
git add db_connection.py
git commit -m "fix: use time-based filtering in CSV mode for window consistency"
```

---

### Task 4: Refactor train_anomaly.py for random window sampling

**Files:**
- Modify: `train_anomaly.py` (major rewrite of main())

This is the largest task. The new flow:
1. Load ALL historical CSV data (keep `_load_csv_chunked`)
2. Remove `_load_db_all` function (CSV-only for training)
3. Rewrite main() to loop random 176h windows

**Step 1: Remove _load_db_all function**

Delete lines 94-141 (the entire `_load_db_all` function). Training is CSV-only.

**Step 2: Rewrite main() function**

Replace the entire `main()` function (lines 148-318) with the following:

```python
def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM anomaly detection model for a single device"
    )
    parser.add_argument(
        "--dev_id", type=int, required=True,
        help="Device ID to train for (e.g. 2001)"
    )
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="CSV data file path. Overrides csv.data_path in config."
    )
    parser.add_argument(
        "--start_date", type=str, default=None,
        help="Training data start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date", type=str, default=None,
        help="Training data end date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    config_path = os.path.join(SCRIPT_DIR, "_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    tag_cd = config["data"]["tag_cd"]  # 30001
    model_cfg = config["model"]
    fetch_hours = config["data"]["fetch_window_hours"]  # 176
    max_steps = config["training"]["max_steps"]  # 1000
    samples_per_window = config["training"]["samples_per_window"]  # 8

    print("=" * 60)
    print(f"[TRAIN] Device ID      : {args.dev_id}")
    print(f"[TRAIN] Tag CD         : {tag_cd}")
    print(f"[TRAIN] Date range     : {args.start_date or '(all)'} ~ {args.end_date or '(all)'}")
    print(f"[TRAIN] Window size    : {fetch_hours}h")
    print(f"[TRAIN] Max steps      : {max_steps}")
    print(f"[TRAIN] Samples/window : {samples_per_window}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. Load all historical CSV data
    # ------------------------------------------------------------------
    t0 = time.time()

    csv_path = args.csv_path or config["csv"]["data_path"]
    csv_abs = os.path.normpath(os.path.join(SCRIPT_DIR, csv_path))
    if not os.path.isfile(csv_abs):
        csv_abs = csv_path  # Try as absolute path

    df_sensor = _load_csv_chunked(
        csv_abs, args.dev_id, tag_cd, args.start_date, args.end_date
    )

    load_time = time.time() - t0
    print(f"[TRAIN] Data loaded in {load_time:.1f}s  ({len(df_sensor)} rows)")

    # ------------------------------------------------------------------
    # 3. Build full historical DataFrame: index=colec_dt, columns=['value']
    # ------------------------------------------------------------------
    df_all = pd.DataFrame(
        data=df_sensor["colec_val"].values,
        index=df_sensor["colec_dt"],
        columns=["value"],
    )
    print(f"[TRAIN] Full data range: {df_all.index[0]} ~ {df_all.index[-1]}")

    # ------------------------------------------------------------------
    # 4. Random window sampling
    # ------------------------------------------------------------------
    window_td = pd.Timedelta(hours=fetch_hours)
    min_end = df_all.index[0] + window_td  # earliest valid window end
    max_end = df_all.index[-1]             # latest valid window end

    if min_end > max_end:
        print(f"[ERROR] Not enough data for a {fetch_hours}h window. "
              f"Data spans {df_all.index[0]} ~ {df_all.index[-1]}")
        sys.exit(1)

    # Convert to timestamps for random sampling
    min_ts = min_end.timestamp()
    max_ts = max_end.timestamp()

    np.random.seed(42)
    X_list = []
    y_list = []
    skipped = 0

    print(f"[TRAIN] Sampling {max_steps} random {fetch_hours}h windows ...")
    t0 = time.time()

    for step in range(max_steps):
        # Pick random end time
        rand_ts = np.random.uniform(min_ts, max_ts)
        end_time = pd.Timestamp.fromtimestamp(rand_ts)
        start_time = end_time - window_td

        # Slice the window by TIME
        window_df = df_all.loc[start_time:end_time].copy()

        if len(window_df) < 2:
            skipped += 1
            continue

        # Preprocess the 176h window (identical to inference)
        try:
            X_df, y_df, _, _ = DP.preprocess(window_df, config, fill_method="ffill")
        except Exception:
            skipped += 1
            continue

        if len(X_df) < samples_per_window:
            skipped += 1
            continue

        # Take the last N samples (matching scoring window)
        X_list.append(X_df.iloc[-samples_per_window:])
        y_list.append(y_df.iloc[-samples_per_window:])

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{max_steps}  "
                  f"(collected {len(X_list)} windows, skipped {skipped})")

    sample_time = time.time() - t0
    print(f"[TRAIN] Sampling done in {sample_time:.1f}s  "
          f"({len(X_list)} windows, {skipped} skipped)")

    if not X_list:
        print("[ERROR] No valid windows collected. Check data quality / date range.")
        sys.exit(1)

    # Concatenate all collected samples
    X_all = pd.concat(X_list, ignore_index=True)
    y_all = pd.concat(y_list, ignore_index=True)
    print(f"[TRAIN] Total samples: {len(X_all)}  Features: {X_all.shape[1]}")

    # ------------------------------------------------------------------
    # 5. Train / test split (shuffled, since samples are from random windows)
    # ------------------------------------------------------------------
    test_size = model_cfg["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )
    print(f"[TRAIN] Train size: {len(X_train)},  Test size: {len(X_test)}")

    # ------------------------------------------------------------------
    # 6. Train LightGBM
    # ------------------------------------------------------------------
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": model_cfg["objective"],
        "metric": model_cfg["metric"],
        "boosting_type": model_cfg["boosting_type"],
        "learning_rate": model_cfg["learning_rate"],
        "num_leaves": model_cfg["num_leaves"],
        "verbose": model_cfg["verbose"],
    }

    print(f"[TRAIN] LightGBM params: {params}")
    print(f"[TRAIN] num_boost_round={model_cfg['num_boost_round']}, "
          f"early_stopping_rounds={model_cfg['early_stopping_rounds']}")

    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        valid_names=["validation"],
        num_boost_round=model_cfg["num_boost_round"],
        callbacks=[lgb.early_stopping(stopping_rounds=model_cfg["early_stopping_rounds"])],
    )
    train_time = time.time() - t0

    # ------------------------------------------------------------------
    # 7. Evaluate on test set
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"[TRAIN] Training completed in {train_time:.1f}s")
    print(f"[TRAIN] Best iteration: {model.best_iteration}")
    print(f"[TRAIN] Test RMSE: {rmse:.4f}  ({X_all.shape[1]} features)")

    # ------------------------------------------------------------------
    # 8. Save model
    # ------------------------------------------------------------------
    model_dir = os.path.join(SCRIPT_DIR, config["anomaly"]["model_dir"])
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{args.dev_id}.txt")
    model.save_model(model_path)
    print(f"[TRAIN] Model saved: {model_path}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Device ID          : {args.dev_id}")
    print(f"  Tag CD             : {tag_cd}")
    print(f"  Data rows (raw)    : {len(df_sensor)}")
    print(f"  Windows sampled    : {len(X_list)}")
    print(f"  Windows skipped    : {skipped}")
    print(f"  Total samples      : {len(X_all)}")
    print(f"  Features           : {X_all.shape[1]}")
    print(f"  Train / Test       : {len(X_train)} / {len(X_test)}")
    print(f"  Test RMSE          : {rmse:.4f}")
    print(f"  Best iteration     : {model.best_iteration}")
    print(f"  Model file         : {model_path}")
    print("=" * 60)
```

**Step 3: Update module docstring**

Replace lines 3-18 with:

```python
"""
train_anomaly.py
LightGBM-based anomaly detection model -- training script.

Reads sensor data from CSV, samples random 176h windows (identical to inference
window size), performs feature engineering via data_preprocessing.preprocess(),
collects training samples, and trains a LightGBM regression model.

CSV-only: DB connection is not used for training (speed optimization).

Usage:
    # Train using CSV path from _config.json
    python train_anomaly.py --dev_id 2001

    # Train with explicit CSV path and date range
    python train_anomaly.py --dev_id 2001 \
        --csv_path dataset/data_colec_h_250730_260212_B0019.csv \
        --start_date 2025-03-24 --end_date 2025-09-09
"""
```

**Step 4: Verify it runs**

Run: `cd /workspace/2024_AI_BEMS/YiUmGoV2 && python train_anomaly.py --dev_id 2001`
Expected: completes successfully with ~8000 total samples, model saved

**Step 5: Commit**

```bash
git add train_anomaly.py
git commit -m "feat: refactor training to use random 176h window sampling for train/inference consistency"
```

---

### Task 5: Update ai_anomaly_runner.py for renamed parameter

**Files:**
- Modify: `ai_anomaly_runner.py:84,87`

**Step 1: Rename variable**

At line 84, rename to match the new parameter name:

```python
# Before
    df_raw = raw_df.set_index("colec_dt")[["colec_val"]].rename(columns={"colec_val": "value"})

# After
    window_df = raw_df.set_index("colec_dt")[["colec_val"]].rename(columns={"colec_val": "value"})
```

At line 87, update the preprocess call:

```python
# Before
    result = data_preprocessing.preprocess(df_raw, config, fill_method="ffill")

# After
    result = data_preprocessing.preprocess(window_df, config, fill_method="ffill")
```

Note: The variable `raw_df` on the right side of line 84 is the return value from `db_connection.read_sensor_data()` -- it is NOT the renamed parameter. Only the local variable name `df_raw` changes to `window_df`.

**Step 2: Commit**

```bash
git add ai_anomaly_runner.py
git commit -m "refactor: rename df_raw to window_df in runner for consistency"
```

---

### Task 6: Update test_integration.py for renamed parameter

**Files:**
- Modify: `test_integration.py:72-83`

**Step 1: Rename variable**

Replace lines 72-83:

```python
# Before
    # 4c. Build df_raw: index=colec_dt, columns=['value']
    df_raw = df_sensor[["colec_dt", "colec_val"]].copy()
    df_raw = df_raw.rename(columns={"colec_val": "value"})
    df_raw = df_raw.set_index("colec_dt")
    df_raw.index.name = None
    print(f"  df_raw shape: {df_raw.shape}")

    # 4d. Preprocess
    import data_preprocessing

    X_df, y_df, nan_counts, missing_ratio = data_preprocessing.preprocess(
        df_raw, config, fill_method="ffill"
    )

# After
    # 4c. Build window_df: index=colec_dt, columns=['value']
    window_df = df_sensor[["colec_dt", "colec_val"]].copy()
    window_df = window_df.rename(columns={"colec_val": "value"})
    window_df = window_df.set_index("colec_dt")
    window_df.index.name = None
    print(f"  window_df shape: {window_df.shape}")

    # 4d. Preprocess
    import data_preprocessing

    X_df, y_df, nan_counts, missing_ratio = data_preprocessing.preprocess(
        window_df, config, fill_method="ffill"
    )
```

**Step 2: Run integration test**

Run: `cd /workspace/2024_AI_BEMS/YiUmGoV2 && python test_integration.py`
Expected: `=== INTEGRATION TEST PASSED ===`

**Step 3: Commit**

```bash
git add test_integration.py
git commit -m "refactor: rename df_raw to window_df in integration test"
```

---

### Task 7: Re-train model and verify end-to-end

**Step 1: Re-train the model with the new window-sampling approach**

Run: `cd /workspace/2024_AI_BEMS/YiUmGoV2 && python train_anomaly.py --dev_id 2001`

Expected output should include:
- `Sampling 1000 random 176h windows ...`
- `Total samples: ~8000`
- `Model saved: models/anomaly/2001.txt`
- Test RMSE value

**Step 2: Run integration test with the new model**

Run: `cd /workspace/2024_AI_BEMS/YiUmGoV2 && python test_integration.py`

Expected: `=== INTEGRATION TEST PASSED ===` with scoring window now showing 2h

**Step 3: Run dry-run inference**

Run: `cd /workspace/2024_AI_BEMS/YiUmGoV2 && python ai_anomaly_runner.py --dry-run`

Expected: completes without errors, prints AD_SCORE for dev_id=2001

**Step 4: Commit the new model**

```bash
git add models/anomaly/2001.txt
git commit -m "feat: re-train model with random 176h window sampling"
```
