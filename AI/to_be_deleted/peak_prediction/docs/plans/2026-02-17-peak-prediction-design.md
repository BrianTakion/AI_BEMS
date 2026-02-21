# Peak Prediction Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a peak prediction service that predicts daily peak power (kW) and peak time for building B0019's main power meter (dev_id=2001) using two LightGBM regression models.

**Architecture:** Two separate LightGBM models -- one predicts peak power, one predicts peak time slot (0-95). Features are built from 14 days of hourly-aggregated data (~40 hybrid features). Inference runs hourly, refining predictions with same-day partial data. Output format: `"207.33@13:15"` written to `MAX_DMAND_FCST_H` table.

**Tech Stack:** Python, LightGBM, pandas, numpy, scikit-learn, holidays (Korean), SQLAlchemy (DB mode)

**Reference codebase:** `/workspace/2024_AI_BEMS/YiUmGoV2/anomaly_detection/` -- the peak_prediction directory contains copies of these files that will be adapted.

---

### Task 1: Update configuration and device config files

**Files:**
- Modify: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/_config.json`
- Create: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/config_peak_devices.csv`
- Delete: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/config_anomaly_devices.csv`

**Step 1: Rewrite _config.json for peak prediction**

Replace the entire file with:

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
  "model": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "num_boost_round": 500,
    "early_stopping_rounds": 30,
    "test_size": 0.15,
    "verbose": -1
  }
}
```

Key changes from anomaly config:
- `fetch_window_hours`: 176 → 336 (14 days for peak lag features)
- Removed `scoring_window_hours` (not applicable; peak prediction is daily)
- Removed `training.max_steps` and `training.samples_per_window` (peak uses daily aggregation, not random window sampling)
- Added `training.min_history_days`: 14
- Changed `model.test_size`: 0.03 → 0.15 (larger test split for fewer samples)
- Changed `model.learning_rate`: 0.1 → 0.05 (slower for small dataset)
- Changed `model.num_leaves`: 50 → 31 (smaller to prevent overfitting)
- Changed `model.num_boost_round`: 1000 → 500
- Renamed `anomaly` section → `peak` section with `dev_id`, `model_dir`, `result_table=MAX_DMAND_FCST_H`
- CSV paths updated: `config_peak_devices_path`, `peak_results_path`

**Step 2: Create config_peak_devices.csv**

```csv
BLDG_ID,DEV_ID,PEAK_PRCV_YN
B0019,2001,Y
```

Only dev_id=2001 (main_power). Uses `PEAK_PRCV_YN` flag instead of `FALT_PRCV_YN`.

**Step 3: Delete config_anomaly_devices.csv**

```bash
rm /workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/config_anomaly_devices.csv
```

**Step 4: Commit**

```bash
git add peak_prediction/_config.json peak_prediction/config_peak_devices.csv
git rm peak_prediction/config_anomaly_devices.csv
git commit -m "feat(peak): update config and device list for peak prediction"
```

---

### Task 2: Rewrite data_preprocessing.py for hybrid peak features

**Files:**
- Modify: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/data_preprocessing.py`

This is the **major rewrite**. The anomaly detection version produces 44 per-timestamp features from a 176h window. The peak prediction version produces ~40 per-day features from 14 days of hourly data.

**Step 1: Replace data_preprocessing.py entirely**

```python
"""
data_preprocessing.py  --  Peak Prediction Feature Engineering

Transforms raw 15-min sensor data into daily feature vectors for peak
power and peak time prediction.

Pipeline:
  1. Resample raw 15-min data to 1-hour means
  2. Extract daily peaks (power value + 15-min slot index) as targets
  3. Build ~40 hybrid features per day from 14-day hourly history

Feature categories (~40 total):
  - Temporal (7): weekday, month, is_holiday, sin/cos encodings
  - Previous-day peaks (4): peak power/slot from prior 1-3 days
  - Same-weekday lag (2): peak power/slot from same weekday last week
  - Rolling peak stats 7d (5): mean, std, max, min peak; mean peak slot
  - Rolling peak stats 14d (4): mean, std, max peak; trend (slope)
  - Daily load shape (4): prev day mean, std, min, max/min ratio
  - Compressed profile prev day (4): morning/afternoon/evening/night means
  - Compressed profile prev week same day (4): morning/afternoon/evening/night means
  - Same-day partial (4): today_max_so_far, today_max_slot_so_far, today_mean_so_far, hours_elapsed
  - Trend (2): peak_trend_7d slope, weekday/weekend peak ratio
"""

import numpy as np
import pandas as pd
import holidays
import datetime


def _get_korean_holidays(year_range):
    """Build a set of Korean holiday dates including extended major holidays."""
    kr_holidays = holidays.KR(years=year_range)
    holiday_set = set(kr_holidays.keys())

    # 설날/추석: add day before and after
    for date in list(kr_holidays.keys()):
        name = kr_holidays[date]
        if name in ["설날", "추석"]:
            holiday_set.add(date - datetime.timedelta(days=1))
            holiday_set.add(date + datetime.timedelta(days=1))

    # 대체공휴일: if holiday falls on weekend, next weekday is substitute
    for date in list(kr_holidays.keys()):
        if date.weekday() in (5, 6):  # Saturday or Sunday
            replacement = date + datetime.timedelta(days=1)
            while replacement.weekday() in (5, 6) or replacement in kr_holidays:
                replacement += datetime.timedelta(days=1)
            holiday_set.add(replacement)

    return holiday_set


def _extract_daily_peaks(df_15min):
    """Extract daily peak power and peak slot from 15-min data.

    Args:
        df_15min: DataFrame with DatetimeIndex and column 'value', at 15-min freq.

    Returns:
        DataFrame indexed by date with columns:
          - peak_power: max value of the day
          - peak_slot: 15-min slot index (0-95) when peak occurred
    """
    daily_records = []
    for date, group in df_15min.groupby(df_15min.index.date):
        if group.empty:
            continue
        peak_idx = group["value"].idxmax()
        peak_power = group["value"].max()
        # slot index: (hour * 4) + (minute // 15)
        peak_slot = peak_idx.hour * 4 + peak_idx.minute // 15
        daily_records.append({
            "date": pd.Timestamp(date),
            "peak_power": peak_power,
            "peak_slot": peak_slot,
        })
    if not daily_records:
        return pd.DataFrame(columns=["peak_power", "peak_slot"])
    result = pd.DataFrame(daily_records).set_index("date")
    return result


def _resample_to_hourly(df_15min):
    """Resample 15-min data to hourly means.

    Args:
        df_15min: DataFrame with DatetimeIndex and column 'value'.

    Returns:
        DataFrame with hourly DatetimeIndex and column 'value'.
    """
    hourly = df_15min[["value"]].resample("1h").mean()
    return hourly


def _build_compressed_profile(hourly_df, target_date):
    """Build 4-period compressed profile for a given date from hourly data.

    Periods: night (0-6h), morning (6-12h), afternoon (12-18h), evening (18-24h).

    Args:
        hourly_df: DataFrame with hourly DatetimeIndex and column 'value'.
        target_date: date to extract profile for.

    Returns:
        Dict with keys: night_mean, morning_mean, afternoon_mean, evening_mean.
        Returns zeros if no data available for that date.
    """
    day_start = pd.Timestamp(target_date)
    day_data = hourly_df.loc[day_start:day_start + pd.Timedelta(hours=23)]

    if day_data.empty:
        return {"night_mean": 0.0, "morning_mean": 0.0,
                "afternoon_mean": 0.0, "evening_mean": 0.0}

    hours = day_data.index.hour
    result = {}
    for name, (h_start, h_end) in [("night", (0, 6)), ("morning", (6, 12)),
                                     ("afternoon", (12, 18)), ("evening", (18, 24))]:
        mask = (hours >= h_start) & (hours < h_end)
        subset = day_data.loc[mask, "value"]
        result[f"{name}_mean"] = subset.mean() if len(subset) > 0 else 0.0

    return result


def preprocess_for_training(df_raw, config):
    """Build training dataset: daily feature vectors + targets from raw 15-min data.

    Args:
        df_raw: DataFrame with columns ['colec_dt', 'colec_val'] at 15-min intervals.
        config: Configuration dict.

    Returns:
        Tuple (X_df, y_power, y_slot):
          - X_df: DataFrame of ~40 features per day
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

    feature_rows = []
    target_power = []
    target_slot = []

    for i in range(min_history_days, len(all_dates)):
        date = all_dates[i]
        features = _build_features_for_date(
            date, daily_peaks, hourly, holiday_set, hours_elapsed=24
        )
        if features is not None:
            feature_rows.append(features)
            target_power.append(daily_peaks.loc[date, "peak_power"])
            target_slot.append(daily_peaks.loc[date, "peak_slot"])

    X_df = pd.DataFrame(feature_rows)
    y_power = pd.Series(target_power, name="peak_power")
    y_slot = pd.Series(target_slot, name="peak_slot")

    return X_df, y_power, y_slot


def preprocess_for_inference(df_raw, config):
    """Build feature vector for today's peak prediction from raw 15-min data.

    Args:
        df_raw: DataFrame with columns ['colec_dt', 'colec_val'], last 14+ days.
        config: Configuration dict.

    Returns:
        DataFrame with a single row of ~40 features for today's prediction.
    """
    # Build a value-indexed time series
    df = pd.DataFrame({
        "value": df_raw["colec_val"].values,
    }, index=pd.to_datetime(df_raw["colec_dt"]))
    df.index = df.index.round("15min")
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("15min").asfreq()
    df["value"] = df["value"].ffill().bfill()

    # Extract daily peaks from 15-min data
    daily_peaks = _extract_daily_peaks(df)

    # Resample to hourly
    hourly = _resample_to_hourly(df)
    hourly["value"] = hourly["value"].ffill().bfill()

    # Today = the last date in the data
    now = df.index[-1]
    today = pd.Timestamp(now.date())

    # hours_elapsed = how many full hours of today's data we have
    today_start = today
    today_data = df.loc[today_start:now]
    hours_elapsed = max(0, (now - today_start).total_seconds() / 3600)

    all_dates = daily_peaks.index.sort_values()
    year_range = range(all_dates[0].year, all_dates[-1].year + 1)
    holiday_set = _get_korean_holidays(year_range)

    features = _build_features_for_date(
        today, daily_peaks, hourly, holiday_set,
        hours_elapsed=hours_elapsed, today_raw_15min=df
    )

    if features is None:
        raise ValueError("Could not build features for today's prediction")

    return pd.DataFrame([features])


def _build_features_for_date(date, daily_peaks, hourly, holiday_set,
                              hours_elapsed=24, today_raw_15min=None):
    """Build the ~40 hybrid feature vector for a single date.

    Args:
        date: pd.Timestamp of the target date.
        daily_peaks: DataFrame indexed by date with columns peak_power, peak_slot.
        hourly: DataFrame with hourly DatetimeIndex and column 'value'.
        holiday_set: Set of holiday dates.
        hours_elapsed: Hours of today's data available (24 for training, partial for inference).
        today_raw_15min: If provided, raw 15-min data for computing same-day partial features.

    Returns:
        Dict of feature name -> value, or None if insufficient history.
    """
    features = {}
    d = date

    # =====================================================================
    # 1. Temporal features (7)
    # =====================================================================
    features["weekday"] = d.weekday()
    features["month"] = d.month
    features["is_holiday"] = 1 if d.date() in holiday_set or d.weekday() >= 5 else 0
    features["sin_month"] = np.sin(2 * np.pi * d.month / 12)
    features["cos_month"] = np.cos(2 * np.pi * d.month / 12)
    features["sin_weekday"] = np.sin(2 * np.pi * d.weekday() / 7)
    features["cos_weekday"] = np.cos(2 * np.pi * d.weekday() / 7)

    # =====================================================================
    # 2. Previous-day peaks (4)
    # =====================================================================
    for lag in [1, 2, 3]:
        lag_date = d - pd.Timedelta(days=lag)
        if lag_date in daily_peaks.index:
            features[f"prev_{lag}d_peak_power"] = daily_peaks.loc[lag_date, "peak_power"]
            if lag == 1:
                features["prev_1d_peak_slot"] = daily_peaks.loc[lag_date, "peak_slot"]
        else:
            features[f"prev_{lag}d_peak_power"] = 0.0
            if lag == 1:
                features["prev_1d_peak_slot"] = 0.0

    # =====================================================================
    # 3. Same-weekday lag (2)
    # =====================================================================
    same_weekday_date = d - pd.Timedelta(days=7)
    if same_weekday_date in daily_peaks.index:
        features["prev_week_same_day_peak_power"] = daily_peaks.loc[same_weekday_date, "peak_power"]
        features["prev_week_same_day_peak_slot"] = daily_peaks.loc[same_weekday_date, "peak_slot"]
    else:
        features["prev_week_same_day_peak_power"] = 0.0
        features["prev_week_same_day_peak_slot"] = 0.0

    # =====================================================================
    # 4. Rolling peak stats - 7 day (5)
    # =====================================================================
    past_7d = []
    past_7d_slots = []
    for lag in range(1, 8):
        lag_date = d - pd.Timedelta(days=lag)
        if lag_date in daily_peaks.index:
            past_7d.append(daily_peaks.loc[lag_date, "peak_power"])
            past_7d_slots.append(daily_peaks.loc[lag_date, "peak_slot"])

    if len(past_7d) >= 3:  # Need at least 3 days for meaningful stats
        features["mean_peak_7d"] = np.mean(past_7d)
        features["std_peak_7d"] = np.std(past_7d)
        features["max_peak_7d"] = np.max(past_7d)
        features["min_peak_7d"] = np.min(past_7d)
        features["mean_peak_slot_7d"] = np.mean(past_7d_slots)
    else:
        return None  # Insufficient history

    # =====================================================================
    # 5. Rolling peak stats - 14 day (4)
    # =====================================================================
    past_14d = list(past_7d)  # start with 7d data
    for lag in range(8, 15):
        lag_date = d - pd.Timedelta(days=lag)
        if lag_date in daily_peaks.index:
            past_14d.append(daily_peaks.loc[lag_date, "peak_power"])

    features["mean_peak_14d"] = np.mean(past_14d)
    features["std_peak_14d"] = np.std(past_14d)
    features["max_peak_14d"] = np.max(past_14d)

    # Linear trend over 14 days (slope of peak power)
    if len(past_14d) >= 7:
        x = np.arange(len(past_14d))
        slope = np.polyfit(x, past_14d[::-1], 1)[0]  # reversed so oldest first
        features["trend_peak_14d"] = slope
    else:
        features["trend_peak_14d"] = 0.0

    # =====================================================================
    # 6. Daily load shape - prev day (4)
    # =====================================================================
    prev_day = d - pd.Timedelta(days=1)
    prev_day_start = pd.Timestamp(prev_day)
    prev_day_end = prev_day_start + pd.Timedelta(hours=23)
    prev_day_hourly = hourly.loc[prev_day_start:prev_day_end, "value"]

    if len(prev_day_hourly) > 0:
        features["prev_day_mean"] = prev_day_hourly.mean()
        features["prev_day_std"] = prev_day_hourly.std() if len(prev_day_hourly) > 1 else 0.0
        features["prev_day_min"] = prev_day_hourly.min()
        min_val = prev_day_hourly.min()
        features["prev_day_max_min_ratio"] = (
            prev_day_hourly.max() / min_val if min_val > 0.01 else 0.0
        )
    else:
        features["prev_day_mean"] = 0.0
        features["prev_day_std"] = 0.0
        features["prev_day_min"] = 0.0
        features["prev_day_max_min_ratio"] = 0.0

    # =====================================================================
    # 7. Compressed profile - prev day (4)
    # =====================================================================
    profile = _build_compressed_profile(hourly, prev_day)
    features["prev_day_night_mean"] = profile["night_mean"]
    features["prev_day_morning_mean"] = profile["morning_mean"]
    features["prev_day_afternoon_mean"] = profile["afternoon_mean"]
    features["prev_day_evening_mean"] = profile["evening_mean"]

    # =====================================================================
    # 8. Compressed profile - prev week same day (4)
    # =====================================================================
    pw_profile = _build_compressed_profile(hourly, same_weekday_date)
    features["pw_night_mean"] = pw_profile["night_mean"]
    features["pw_morning_mean"] = pw_profile["morning_mean"]
    features["pw_afternoon_mean"] = pw_profile["afternoon_mean"]
    features["pw_evening_mean"] = pw_profile["evening_mean"]

    # =====================================================================
    # 9. Same-day partial features (4)
    # =====================================================================
    features["hours_elapsed"] = hours_elapsed

    if hours_elapsed > 0 and today_raw_15min is not None:
        today_start = pd.Timestamp(d)
        now = today_start + pd.Timedelta(hours=hours_elapsed)
        today_data = today_raw_15min.loc[today_start:now, "value"]
        if len(today_data) > 0:
            features["today_max_so_far"] = today_data.max()
            max_idx = today_data.idxmax()
            features["today_max_slot_so_far"] = max_idx.hour * 4 + max_idx.minute // 15
            features["today_mean_so_far"] = today_data.mean()
        else:
            features["today_max_so_far"] = 0.0
            features["today_max_slot_so_far"] = 0.0
            features["today_mean_so_far"] = 0.0
    elif hours_elapsed > 0:
        # Training mode: use hourly data for partial features
        today_start = pd.Timestamp(d)
        today_end = today_start + pd.Timedelta(hours=min(hours_elapsed, 23))
        today_hourly = hourly.loc[today_start:today_end, "value"]
        if len(today_hourly) > 0:
            features["today_max_so_far"] = today_hourly.max()
            max_idx = today_hourly.idxmax()
            features["today_max_slot_so_far"] = max_idx.hour * 4
            features["today_mean_so_far"] = today_hourly.mean()
        else:
            features["today_max_so_far"] = 0.0
            features["today_max_slot_so_far"] = 0.0
            features["today_mean_so_far"] = 0.0
    else:
        features["today_max_so_far"] = 0.0
        features["today_max_slot_so_far"] = 0.0
        features["today_mean_so_far"] = 0.0

    # =====================================================================
    # 10. Trend features (2)
    # =====================================================================
    # 7-day slope (reuse past_7d, reversed to chronological order)
    if len(past_7d) >= 3:
        x = np.arange(len(past_7d))
        features["peak_trend_7d"] = np.polyfit(x, past_7d[::-1], 1)[0]
    else:
        features["peak_trend_7d"] = 0.0

    # Weekday vs weekend peak ratio
    weekday_peaks = []
    weekend_peaks = []
    for lag in range(1, 15):
        lag_date = d - pd.Timedelta(days=lag)
        if lag_date in daily_peaks.index:
            pp = daily_peaks.loc[lag_date, "peak_power"]
            if lag_date.weekday() < 5:
                weekday_peaks.append(pp)
            else:
                weekend_peaks.append(pp)

    if weekday_peaks and weekend_peaks and np.mean(weekend_peaks) > 0.01:
        features["weekday_weekend_peak_ratio"] = np.mean(weekday_peaks) / np.mean(weekend_peaks)
    else:
        features["weekday_weekend_peak_ratio"] = 1.0

    return features
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/data_preprocessing.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add peak_prediction/data_preprocessing.py
git commit -m "feat(peak): rewrite data_preprocessing.py with ~40 hybrid features

Daily feature vectors: temporal, peak lags, rolling stats, compressed
profiles, same-day partial features, and trend indicators."
```

---

### Task 3: Rewrite infer_peak.py (replaces infer_anomaly.py)

**Files:**
- Create: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/infer_peak.py`
- Delete: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/infer_anomaly.py`

**Step 1: Create infer_peak.py**

```python
"""
infer_peak.py  --  Peak Prediction Inference Utilities

Provides model loading, prediction, and result formatting for peak
power and peak time prediction.

Two models:
  - Power model: predicts daily peak power (kW)
  - Time model: predicts daily peak time slot (0-95, each = 15 min)

Output format: "207.33@13:15" (peak_power@peak_time)
"""

import os
import numpy as np
import lightgbm as lgb


def load_model(model_path):
    """Load a LightGBM Booster from a .txt file.

    Args:
        model_path: Path to the LightGBM model text file.

    Returns:
        lgb.Booster instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return lgb.Booster(model_file=model_path)


def predict_peak(model_power, model_time, X_df):
    """Predict peak power and peak time from feature DataFrame.

    Args:
        model_power: LightGBM Booster for peak power prediction.
        model_time: LightGBM Booster for peak time slot prediction.
        X_df: pandas DataFrame with feature columns (single row for inference).

    Returns:
        Tuple (peak_power, peak_slot):
          - peak_power: float, predicted peak power in kW
          - peak_slot: int, predicted 15-min slot index (0-95)
    """
    power_pred = model_power.predict(X_df)
    time_pred = model_time.predict(X_df)

    # Take the last prediction (most relevant for single-row inference)
    peak_power = float(power_pred[-1])
    peak_slot = int(np.clip(np.round(time_pred[-1]), 0, 95))

    # Ensure peak_power is non-negative
    peak_power = max(0.0, peak_power)

    return peak_power, peak_slot


def format_peak_result(peak_power, peak_slot):
    """Format peak prediction as "power@HH:MM" string.

    Args:
        peak_power: Predicted peak power (float).
        peak_slot: Predicted 15-min slot index 0-95.

    Returns:
        String in format "207.33@13:15".
    """
    total_minutes = peak_slot * 15
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{peak_power:.2f}@{hours:02d}:{minutes:02d}"
```

**Step 2: Delete infer_anomaly.py**

```bash
rm /workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/infer_anomaly.py
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/infer_peak.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add peak_prediction/infer_peak.py
git rm peak_prediction/infer_anomaly.py
git commit -m "feat(peak): add infer_peak.py with two-model prediction and formatting

Replaces infer_anomaly.py. Predicts peak power + time slot,
formats as '207.33@13:15' for MAX_DMAND_FCST_H table."
```

---

### Task 4: Adapt data_source.py for peak prediction tables

**Files:**
- Modify: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/data_source.py`

Changes from anomaly version:
- `read_enabled_devices()`: read from `config_peak_devices.csv`, filter by `PEAK_PRCV_YN`
- `write_anomaly_result()` → `write_peak_result()`: write to `MAX_DMAND_FCST_H` with `DLY_MAX_DMAND_FCST_INF` column
- `read_sensor_data()`: unchanged (same data source, same structure)

**Step 1: Replace data_source.py entirely**

```python
"""
data_source.py  --  Peak Prediction Data Access Layer

Abstraction for data access in both CSV (dev) and PostgreSQL DB (production) modes.

Usage:
    import json, data_source as DS
    with open('_config.json') as f:
        config = json.load(f)
    source = DS.create_data_source(config)
    devices = DS.read_enabled_devices(source, config)
    df = DS.read_sensor_data(source, config, bldg_id='B0019', dev_id=2001)
    DS.write_peak_result(source, config, 'B0019', '207.33@13:15')
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(rel_path: str) -> str:
    """Resolve a path relative to the peak_prediction/ directory."""
    return os.path.normpath(os.path.join(_SCRIPT_DIR, rel_path))


# ===================================================================
# 1. create_data_source
# ===================================================================
def create_data_source(config: dict):
    """Return a data-access handle (SQLAlchemy engine for DB, config dict for CSV)."""
    mode = config.get("data_source", "csv")

    if mode == "db":
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise ImportError(
                "sqlalchemy is required for DB mode. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            ) from exc

        db_cfg = config["db"]
        url = (
            f"postgresql://{db_cfg['user']}:{db_cfg['password']}"
            f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
        )
        engine = create_engine(url)
        logger.info("DB engine created: %s:%s/%s",
                     db_cfg["host"], db_cfg["port"], db_cfg["database"])
        return engine

    logger.info("CSV mode: data_path=%s", config["csv"].get("data_path"))
    return config


# ===================================================================
# 2. read_enabled_devices
# ===================================================================
def read_enabled_devices(source, config: dict) -> list[dict]:
    """Return list of devices enabled for peak prediction.

    Each element: {'bldg_id': str, 'dev_id': int}.
    """
    mode = config.get("data_source", "csv")

    if mode == "db":
        table = config["peak"]["config_table"]
        query = (
            f'SELECT "BLDG_ID" AS bldg_id, "DEV_ID" AS dev_id '
            f'FROM "{table}" '
            f"WHERE \"FALT_PRCV_YN\" = 'Y'"
        )
        df = pd.read_sql(query, source)
        # Filter to only dev_id matching peak config
        peak_dev_id = str(config["peak"]["dev_id"])
        df = df[df["dev_id"].astype(str) == peak_dev_id]
        devices = df.to_dict(orient="records")
        logger.info("DB: %d peak devices found", len(devices))
        return devices

    # CSV mode
    csv_path = config["csv"].get("config_peak_devices_path")
    if csv_path:
        abs_path = _resolve_path(csv_path)
        if os.path.isfile(abs_path):
            df = pd.read_csv(abs_path, dtype={"BLDG_ID": str, "DEV_ID": int})
            # Support both PEAK_PRCV_YN and FALT_PRCV_YN column names
            yn_col = "PEAK_PRCV_YN" if "PEAK_PRCV_YN" in df.columns else "FALT_PRCV_YN"
            df = df[df[yn_col] == "Y"]
            devices = [{"bldg_id": row["BLDG_ID"], "dev_id": row["DEV_ID"]}
                       for _, row in df.iterrows()]
            logger.info("CSV: read %d enabled devices from %s", len(devices), abs_path)
            return devices
        logger.warning("CSV: device config not found: %s", abs_path)

    # Fallback
    devices = [{"bldg_id": "B0019", "dev_id": 2001}]
    logger.info("CSV: returning fallback device list")
    return devices


# ===================================================================
# 3. read_sensor_data  (unchanged from anomaly_detection)
# ===================================================================
def read_sensor_data(
    source,
    config: dict,
    bldg_id: str,
    dev_id: int,
    fetch_hours: int | None = None,
) -> pd.DataFrame:
    """Read sensor data for a device. Returns DataFrame with [colec_dt, colec_val].

    Parameters:
        fetch_hours: None=use config default (336h), 0=all history, N=last N hours.
    """
    mode = config.get("data_source", "csv")
    tag_cd = config["data"]["tag_cd"]
    if fetch_hours is None:
        fetch_hours = config["data"]["fetch_window_hours"]

    if mode == "db":
        table = config["data"]["collection_table"]
        if fetch_hours > 0:
            cutoff = datetime.now() - timedelta(hours=fetch_hours)
            query = (
                f'SELECT "COLEC_DT" AS colec_dt, "COLEC_VAL" AS colec_val '
                f'FROM "{table}" '
                f"WHERE \"BLDG_ID\" = %(bldg_id)s "
                f"  AND \"DEV_ID\" = %(dev_id)s "
                f"  AND \"TAG_CD\" = %(tag_cd)s "
                f"  AND \"COLEC_DT\" >= %(cutoff)s "
                f'ORDER BY "COLEC_DT" ASC'
            )
            params = {"bldg_id": bldg_id, "dev_id": str(dev_id),
                      "tag_cd": str(tag_cd), "cutoff": cutoff}
        else:
            query = (
                f'SELECT "COLEC_DT" AS colec_dt, "COLEC_VAL" AS colec_val '
                f'FROM "{table}" '
                f"WHERE \"BLDG_ID\" = %(bldg_id)s "
                f"  AND \"DEV_ID\" = %(dev_id)s "
                f"  AND \"TAG_CD\" = %(tag_cd)s "
                f'ORDER BY "COLEC_DT" ASC'
            )
            params = {"bldg_id": bldg_id, "dev_id": str(dev_id),
                      "tag_cd": str(tag_cd)}
        df = pd.read_sql(query, source, params=params)
        df["colec_dt"] = pd.to_datetime(df["colec_dt"])
        df["colec_val"] = df["colec_val"].astype(float)
        label = f"{fetch_hours}h" if fetch_hours > 0 else "all"
        logger.info("DB: %d rows for dev_id=%s (%s)", len(df), dev_id, label)
        return df

    # CSV mode
    csv_path = _resolve_path(config["csv"]["data_path"])
    logger.info("CSV: reading %s (dev_id=%s, tag_cd=%s)", csv_path, dev_id, tag_cd)

    CHUNK_SIZE = 500_000
    usecols = ["dev_id", "tag_cd", "colec_dt", "colec_val"]
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype={"dev_id": str, "tag_cd": str, "colec_val": float},
        parse_dates=["colec_dt"],
        chunksize=CHUNK_SIZE,
    ):
        mask = (chunk["dev_id"] == str(dev_id)) & (chunk["tag_cd"] == str(tag_cd))
        filtered = chunk.loc[mask]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        logger.warning("CSV: no data for dev_id=%s tag_cd=%s", dev_id, tag_cd)
        return pd.DataFrame(columns=["colec_dt", "colec_val"])

    df = pd.concat(chunks, ignore_index=True)
    df["colec_dt"] = pd.to_datetime(df["colec_dt"]).dt.floor("min")
    df = df.sort_values("colec_dt").reset_index(drop=True)

    if fetch_hours > 0:
        cutoff = df["colec_dt"].iloc[-1] - timedelta(hours=fetch_hours)
        df = df[df["colec_dt"] >= cutoff].reset_index(drop=True)

    df = df[["colec_dt", "colec_val"]]
    label = f"{fetch_hours}h" if fetch_hours > 0 else "all"
    logger.info("CSV: %d rows (%s) for dev_id=%s", len(df), label, dev_id)
    return df


# ===================================================================
# 4. write_peak_result  (replaces write_anomaly_result)
# ===================================================================
def write_peak_result(
    source,
    config: dict,
    bldg_id: str,
    peak_info: str,
) -> None:
    """Persist a peak prediction result.

    Args:
        source: Data source handle.
        config: Configuration dict.
        bldg_id: Building ID string (e.g., "B0019").
        peak_info: Formatted string "207.33@13:15" (peak_power@peak_time).

    DB mode:  INSERT INTO MAX_DMAND_FCST_H (USE_DT, BLDG_ID, DLY_MAX_DMAND_FCST_INF).
    CSV mode: print + append to output/peak_results.csv.
    """
    mode = config.get("data_source", "csv")
    now = datetime.now()

    if mode == "db":
        table = config["peak"]["result_table"]  # MAX_DMAND_FCST_H
        row = pd.DataFrame([{
            "USE_DT": now,
            "BLDG_ID": bldg_id,
            "DLY_MAX_DMAND_FCST_INF": peak_info,
        }])
        row.to_sql(table, source, if_exists="append", index=False)
        logger.info("DB: inserted peak result for %s: %s", bldg_id, peak_info)
        return

    # CSV mode
    msg = (
        f"[{now:%Y-%m-%d %H:%M:%S}] "
        f"bldg_id={bldg_id}, "
        f"DLY_MAX_DMAND_FCST_INF={peak_info}"
    )
    print(msg)
    logger.info("CSV (write_peak_result): %s", msg)

    result_path = config["csv"].get("peak_results_path")
    if result_path:
        abs_path = _resolve_path(result_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        row = pd.DataFrame([{
            "USE_DT": now.strftime("%Y-%m-%d %H:%M:%S"),
            "BLDG_ID": bldg_id,
            "DLY_MAX_DMAND_FCST_INF": peak_info,
        }])
        write_header = not os.path.isfile(abs_path)
        row.to_csv(abs_path, mode="a", header=write_header, index=False)
        logger.info("CSV: result appended to %s", abs_path)
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/data_source.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add peak_prediction/data_source.py
git commit -m "feat(peak): adapt data_source.py for peak prediction tables

write_anomaly_result -> write_peak_result writing to MAX_DMAND_FCST_H.
read_enabled_devices reads config_peak_devices.csv with PEAK_PRCV_YN."
```

---

### Task 5: Create train_peak.py (replaces train_anomaly.py)

**Files:**
- Create: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/train_peak.py`
- Delete: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/train_anomaly.py`

Key differences from train_anomaly.py:
- No random window sampling; uses `preprocess_for_training()` which builds daily samples
- Trains two models: `{dev_id}_power.txt` and `{dev_id}_time.txt`
- Uses 15% test split (not 3%)
- Evaluates RMSE for power and MAE for time slot

**Step 1: Create train_peak.py**

```python
#!/usr/bin/env python
"""
train_peak.py  --  Train LightGBM models for daily peak prediction.

Trains two models per device:
  - {dev_id}_power.txt: predicts daily peak power (kW)
  - {dev_id}_time.txt: predicts daily peak time slot (0-95)

Reads all historical sensor data, builds daily feature vectors via
data_preprocessing.preprocess_for_training(), then trains LightGBM
regression models.

Usage:
    python peak_prediction/train_peak.py --csv
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import data_preprocessing as DP
import data_source as DS


def train_model(X_train, y_train, X_test, y_test, model_cfg, model_path, target_name):
    """Train a single LightGBM model and save it.

    Returns:
        Tuple (model, metrics_dict).
    """
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

    print(f"  [TRAIN-{target_name}] LightGBM params: {params}")

    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        valid_names=["validation"],
        num_boost_round=model_cfg["num_boost_round"],
        callbacks=[lgb.early_stopping(stopping_rounds=model_cfg["early_stopping_rounds"])],
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    model.save_model(model_path)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "best_iteration": model.best_iteration,
        "train_time": elapsed,
    }

    print(f"  [TRAIN-{target_name}] Done in {elapsed:.1f}s | "
          f"RMSE={rmse:.4f} MAE={mae:.4f} | "
          f"best_iter={model.best_iteration} | saved: {model_path}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM peak prediction models")
    parser.add_argument("--csv", action="store_true", required=True,
                        help="Train using CSV data source")
    args = parser.parse_args()

    # 1. Load configuration
    config_path = os.path.join(SCRIPT_DIR, "_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["data_source"] = "csv"

    model_cfg = config["model"]
    model_dir = os.path.join(SCRIPT_DIR, config["peak"]["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    # 2. Read enabled devices
    source = DS.create_data_source(config)
    devices = DS.read_enabled_devices(source, config)
    print(f"[TRAIN] Enabled devices ({len(devices)}): {[d['dev_id'] for d in devices]}")

    # 3. Device loop
    for device in devices:
        bldg_id = device["bldg_id"]
        dev_id = device["dev_id"]
        power_model_path = os.path.join(model_dir, f"{dev_id}_power.txt")
        time_model_path = os.path.join(model_dir, f"{dev_id}_time.txt")

        # Skip if both models already exist
        if os.path.isfile(power_model_path) and os.path.isfile(time_model_path):
            print(f"[SKIP] Models already exist: {power_model_path}, {time_model_path}")
            continue

        print("=" * 60)
        print(f"[TRAIN] Device: bldg_id={bldg_id}, dev_id={dev_id}")
        print("=" * 60)

        # 4. Load ALL historical sensor data
        t0 = time.time()
        df_sensor = DS.read_sensor_data(source, config, bldg_id, dev_id, fetch_hours=0)
        if df_sensor.empty:
            print(f"[ERROR] No data found for dev_id={dev_id}")
            continue
        print(f"[TRAIN] Data loaded in {time.time() - t0:.1f}s ({len(df_sensor)} rows)")

        # 5. Build daily training dataset
        t0 = time.time()
        try:
            X_df, y_power, y_slot = DP.preprocess_for_training(df_sensor, config)
        except ValueError as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            continue
        print(f"[TRAIN] Features built in {time.time() - t0:.1f}s: "
              f"{len(X_df)} days, {X_df.shape[1]} features")

        # 6. Train/test split
        test_size = model_cfg["test_size"]
        X_train, X_test, y_pow_train, y_pow_test, y_slot_train, y_slot_test = (
            train_test_split(
                X_df, y_power, y_slot,
                test_size=test_size, random_state=42, shuffle=True,
            )
        )
        print(f"[TRAIN] Train: {len(X_train)} days, Test: {len(X_test)} days")

        # 7. Train peak power model
        print(f"\n--- Training POWER model ---")
        power_model, power_metrics = train_model(
            X_train, y_pow_train, X_test, y_pow_test,
            model_cfg, power_model_path, "POWER"
        )

        # 8. Train peak time model
        print(f"\n--- Training TIME model ---")
        time_model, time_metrics = train_model(
            X_train, y_slot_train, X_test, y_slot_test,
            model_cfg, time_model_path, "TIME"
        )

        # 9. Summary
        # Convert test slot predictions to hours for interpretability
        y_slot_pred = time_model.predict(X_test, num_iteration=time_model.best_iteration)
        slot_errors = np.abs(y_slot_test.values - np.round(y_slot_pred))
        time_error_minutes = slot_errors.mean() * 15  # Each slot = 15 min

        print()
        print("=" * 60)
        print("  Training Summary")
        print("=" * 60)
        print(f"  Device              : {bldg_id}/{dev_id}")
        print(f"  Training days       : {len(X_train)}")
        print(f"  Test days           : {len(X_test)}")
        print(f"  Features            : {X_df.shape[1]}")
        print(f"  --- Power Model ---")
        print(f"  RMSE                : {power_metrics['rmse']:.4f} kW")
        print(f"  MAE                 : {power_metrics['mae']:.4f} kW")
        print(f"  Best iteration      : {power_metrics['best_iteration']}")
        print(f"  --- Time Model ---")
        print(f"  RMSE (slots)        : {time_metrics['rmse']:.4f}")
        print(f"  MAE (slots)         : {time_metrics['mae']:.4f}")
        print(f"  Avg time error      : {time_error_minutes:.0f} minutes")
        print(f"  Best iteration      : {time_metrics['best_iteration']}")
        print(f"  --- Files ---")
        print(f"  Power model         : {power_model_path}")
        print(f"  Time model          : {time_model_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
```

**Step 2: Delete train_anomaly.py**

```bash
rm /workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/train_anomaly.py
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/train_peak.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add peak_prediction/train_peak.py
git rm peak_prediction/train_anomaly.py
git commit -m "feat(peak): add train_peak.py for dual-model training

Trains two LightGBM models per device: peak power (kW) and peak
time slot (0-95). Uses daily feature vectors from preprocess_for_training()."
```

---

### Task 6: Create ai_peak_runner.py (replaces ai_anomaly_runner.py)

**Files:**
- Create: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/ai_peak_runner.py`
- Delete: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/ai_anomaly_runner.py`

Key differences from ai_anomaly_runner.py:
- Loads two models per device: `{dev_id}_power.txt` and `{dev_id}_time.txt`
- Uses `preprocess_for_inference()` instead of `preprocess()`
- Calls `infer_peak.predict_peak()` and `infer_peak.format_peak_result()`
- Writes via `data_source.write_peak_result()`
- No anomaly scoring or description generation

**Step 1: Create ai_peak_runner.py**

```python
#!/usr/bin/env python
"""
ai_peak_runner.py  --  Peak Prediction Inference Runner

Predicts daily peak power and peak time for enabled devices.
Designed to run hourly via cron, refining predictions as same-day data arrives.

Usage:
    python peak_prediction/ai_peak_runner.py          # DB mode (production)
    python peak_prediction/ai_peak_runner.py --csv    # CSV mode (development)
"""

import argparse
import json
import logging
import os
import time

import data_source
import data_preprocessing
import infer_peak

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("ai_peak_runner")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config():
    config_path = os.path.join(SCRIPT_DIR, "_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Config loaded from %s", config_path)
    return config


def process_device(source, config, bldg_id, dev_id):
    """Process a single device: load models, fetch data, preprocess, predict, write result.

    Returns:
        Formatted peak_info string on success, or None on skip/error.
    """
    model_dir = config["peak"]["model_dir"]
    power_model_path = os.path.join(SCRIPT_DIR, model_dir, f"{dev_id}_power.txt")
    time_model_path = os.path.join(SCRIPT_DIR, model_dir, f"{dev_id}_time.txt")

    # 1. Load models
    try:
        model_power = infer_peak.load_model(power_model_path)
        model_time = infer_peak.load_model(time_model_path)
    except FileNotFoundError as e:
        logger.warning("Model not found for dev_id=%s: %s -- skipping", dev_id, e)
        return None
    logger.info("Models loaded: %s, %s", power_model_path, time_model_path)

    # 2. Read sensor data (last 14 days = 336 hours)
    raw_df = data_source.read_sensor_data(source, config, bldg_id, dev_id)
    if len(raw_df) < 2:
        logger.warning("Insufficient data for dev_id=%s (%d rows) -- skipping",
                       dev_id, len(raw_df))
        return None

    # 3. Preprocess for inference
    try:
        X_df = data_preprocessing.preprocess_for_inference(raw_df, config)
    except (ValueError, Exception) as e:
        logger.error("Preprocessing failed for dev_id=%s: %s", dev_id, e)
        return None
    logger.info("Features built: %d columns", X_df.shape[1])

    # 4. Predict
    peak_power, peak_slot = infer_peak.predict_peak(model_power, model_time, X_df)

    # 5. Format result
    peak_info = infer_peak.format_peak_result(peak_power, peak_slot)
    logger.info("dev_id=%s => %s", dev_id, peak_info)

    # 6. Write result
    data_source.write_peak_result(source, config, bldg_id, peak_info)

    return peak_info


def main():
    parser = argparse.ArgumentParser(description="AI BEMS Peak Prediction Runner")
    parser.add_argument("--csv", action="store_true",
                        help="Run in CSV mode (development)")
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    config["data_source"] = "csv" if args.csv else "db"

    logger.info("=== AI Peak Prediction Start ===")
    logger.info("Mode: %s", config["data_source"].upper())

    start_time = time.time()

    source = data_source.create_data_source(config)
    devices = data_source.read_enabled_devices(source, config)
    logger.info("Enabled devices: %d", len(devices))

    success_count = 0
    for device in devices:
        bldg_id = device["bldg_id"]
        dev_id = device["dev_id"]
        logger.info("Processing: bldg_id=%s, dev_id=%s", bldg_id, dev_id)

        try:
            result = process_device(source, config, bldg_id, dev_id)
            if result is not None:
                success_count += 1
        except Exception:
            logger.exception("Error processing bldg_id=%s, dev_id=%s", bldg_id, dev_id)

    elapsed = time.time() - start_time
    logger.info("=== Done: %d/%d devices in %.1fs ===",
                success_count, len(devices), elapsed)


if __name__ == "__main__":
    main()
```

**Step 2: Delete ai_anomaly_runner.py**

```bash
rm /workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/ai_anomaly_runner.py
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/ai_peak_runner.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add peak_prediction/ai_peak_runner.py
git rm peak_prediction/ai_anomaly_runner.py
git commit -m "feat(peak): add ai_peak_runner.py for hourly peak inference

Loads two models, preprocesses last 14 days, predicts peak power
and time, writes '207.33@13:15' format to MAX_DMAND_FCST_H."
```

---

### Task 7: Run training and verify end-to-end

**Files:** No new files; verifies Tasks 1-6.

**Step 1: Run training**

Run: `cd /workspace/2024_AI_BEMS/YiUmGoV2 && python peak_prediction/train_peak.py --csv`

Expected output (approximate):
```
[TRAIN] Enabled devices (1): [2001]
============================================================
[TRAIN] Device: bldg_id=B0019, dev_id=2001
============================================================
[TRAIN] Data loaded in ...s (... rows)
[TRAIN] Features built in ...s: ~180 days, ~40 features
[TRAIN] Train: ~153 days, Test: ~27 days

--- Training POWER model ---
  [TRAIN-POWER] Done in ...s | RMSE=... MAE=...

--- Training TIME model ---
  [TRAIN-TIME] Done in ...s | RMSE=... MAE=...

  Training Summary
  ...
```

Verify: `ls -la /workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/models/`
Expected: Files `2001_power.txt` and `2001_time.txt` exist.

**Step 2: Run inference**

Run: `cd /workspace/2024_AI_BEMS/YiUmGoV2 && python peak_prediction/ai_peak_runner.py --csv`

Expected output (approximate):
```
... [INFO] === AI Peak Prediction Start ===
... [INFO] Mode: CSV
... [INFO] Processing: bldg_id=B0019, dev_id=2001
... [INFO] Models loaded: ...
... [INFO] Features built: ~40 columns
... [INFO] dev_id=2001 => 207.33@13:15
[2026-02-17 ...] bldg_id=B0019, DLY_MAX_DMAND_FCST_INF=207.33@13:15
... [INFO] === Done: 1/1 devices in ...s ===
```

Verify: `cat /workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/output/peak_results.csv`
Expected: CSV with columns `USE_DT,BLDG_ID,DLY_MAX_DMAND_FCST_INF` and one data row.

**Step 3: Fix any issues that arise, then commit**

```bash
git add peak_prediction/models/ peak_prediction/output/
git commit -m "feat(peak): verify training and inference end-to-end

Trained models for dev_id=2001 and verified CSV-mode inference
produces correctly formatted output."
```

---

### Task 8: Clean up leftover anomaly_detection references

**Files:**
- Verify: `/workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/` contains no anomaly-specific files

**Step 1: Check for any remaining anomaly files**

Run: `ls /workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/`

Verify these files are gone:
- `train_anomaly.py` (deleted in Task 5)
- `ai_anomaly_runner.py` (deleted in Task 6)
- `infer_anomaly.py` (deleted in Task 3)
- `config_anomaly_devices.csv` (deleted in Task 1)

Verify these files exist:
- `_config.json` (Task 1)
- `config_peak_devices.csv` (Task 1)
- `data_preprocessing.py` (Task 2)
- `infer_peak.py` (Task 3)
- `data_source.py` (Task 4)
- `train_peak.py` (Task 5)
- `ai_peak_runner.py` (Task 6)
- `__init__.py`
- `models/` directory with trained models
- `output/` directory with results
- `docs/` directory

**Step 2: Search for stale references**

Run: `grep -r "anomaly" /workspace/2024_AI_BEMS/YiUmGoV2/peak_prediction/ --include="*.py" --include="*.json" --include="*.csv" -l`

Expected: No files should contain "anomaly" references. If any are found, update them.

**Step 3: Final commit if needed**

```bash
git add -A peak_prediction/
git commit -m "chore(peak): clean up stale anomaly_detection references"
```
