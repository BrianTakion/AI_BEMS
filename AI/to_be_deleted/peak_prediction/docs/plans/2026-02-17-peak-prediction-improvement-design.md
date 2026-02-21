# Peak Prediction Improvement Design

**Date:** 2026-02-17
**Approach:** B — Fix Data Integrity + Feature Enhancement
**Targets:** Power RMSE < 15 kW, Time error < 45 min

## Problem Analysis

Three critical issues cause poor real-world prediction performance:

### 1. Target Leakage in Same-Day Features (Critical)

`preprocess_for_training()` calls `_build_features_for_date(hours_elapsed=24)`,
which sets `today_max_so_far` to the actual daily peak power — the target being predicted.
The model learns `peak_power ≈ today_max_so_far` during training, but at inference time
these features contain only partial-day data (e.g., 8h of 24h), causing a severe
train/inference distribution mismatch.

### 2. Random Train/Test Split on Time Series

`train_test_split(shuffle=True)` leaks temporal autocorrelation between train and test
sets, inflating reported metrics beyond real-world performance.

### 3. No Per-Model Hyperparameter Tuning

Both power and time models share identical default LightGBM parameters despite predicting
fundamentally different targets (continuous kW vs. discrete time slots).

## Design

### Change 1: Fix Same-Day Feature Leak

**File:** `data_preprocessing.py`

- `preprocess_for_training()` will pass the raw 15-min DataFrame to
  `_build_features_for_date()` via `today_raw_15min` parameter.
- For each training sample, `hours_elapsed` is sampled uniformly from `[0, 23]`
  using a fixed random seed for reproducibility.
- When `hours_elapsed < 24`, same-day features are computed from only the first
  N hours of data — matching inference conditions exactly.
- When `hours_elapsed = 0`, same-day features are all zeros (early morning prediction).

### Change 2: Chronological Split + Time-Series CV

**File:** `train_peak.py`

- Replace `train_test_split(shuffle=True)` with chronological split: sort by date,
  last 15% becomes the test set.
- Add `TimeSeriesSplit(n_splits=5)` expanding-window cross-validation during training
  to get reliable performance estimates across different time windows.
- Report average + std of RMSE/MAE across CV folds.
- Final model is retrained on full training set, evaluated once on held-out test set.

### Change 3: Enhanced Feature Engineering (+17 features)

**File:** `data_preprocessing.py`

#### A. Quantile features (4)
- `peak_75pct_7d`: 75th percentile of past 7 daily peaks
- `peak_90pct_7d`: 90th percentile of past 7 daily peaks
- `peak_75pct_14d`: 75th percentile of past 14 daily peaks
- `peak_90pct_14d`: 90th percentile of past 14 daily peaks

#### B. Peak time distribution features (4)
- `peak_slot_std_7d`: Std deviation of peak slots over 7 days
- `peak_slot_median_7d`: Median peak slot over 7 days
- `most_common_peak_period`: Mode of peak period (0=night, 1=morning, 2=afternoon, 3=evening)
- `same_weekday_peak_slot_2w`: Average peak slot for same weekday over last 2 weeks

#### C. Day-type interaction features (3)
- `is_weekday_x_mean_peak_7d`: weekday indicator × mean_peak_7d
- `prev_day_was_holiday`: Whether yesterday was holiday/weekend
- `consecutive_workdays`: Number of consecutive workdays before this date

#### D. Load ramp features (2)
- `prev_day_morning_ramp`: morning_mean - night_mean of previous day
- `prev_day_peak_to_mean_ratio`: prev_1d_peak_power / prev_day_mean

Total: ~57 features (40 existing + 17 new).

### Change 4: Separate Model Configurations

**Files:** `_config.json`, `train_peak.py`

Split `model` into `model_power` and `model_time`:

**Power model:** `learning_rate=0.05, num_leaves=20, min_child_samples=10,
num_boost_round=1000, early_stopping_rounds=50`

**Time model:** `learning_rate=0.05, num_leaves=15, min_child_samples=15,
num_boost_round=1000, early_stopping_rounds=50`

Shared: `objective=regression, metric=rmse, boosting_type=gbdt`

Backward compatible: falls back to `model` key if per-model keys absent.

## Files Modified

| File | Changes |
|---|---|
| `data_preprocessing.py` | Fix leak, add 17 features |
| `train_peak.py` | Chronological split, time-series CV, per-model config |
| `_config.json` | Split model config into model_power / model_time |

No new files. No new dependencies.

## Expected Outcome

- Honest metrics (reported numbers will reflect real-world performance)
- Power RMSE target: < 15 kW
- Time error target: < 45 min
- More reliable evaluation through time-series CV
