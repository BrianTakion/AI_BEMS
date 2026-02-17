# Peak Prediction Service Design

## Summary

Predict daily peak power (kW) and peak time for building B0019's main power meter (dev_id=2001) using two LightGBM regression models with hourly-aggregated hybrid features.

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model count | Two separate models (power + time) | Each target gets optimized independently |
| Peak time modeling | Regression on slot index (0-95) | Least compute, captures proximity, snaps to 15-min grid |
| Data granularity | Hourly aggregation | Balance between detail and sample count |
| Feature approach | Hybrid (peak patterns + compressed profile) | ~40 features, low overfitting risk |
| Lag window | 14 days | Captures two full weekly cycles |
| Inference schedule | Hourly | Refines prediction as same-day data arrives |

## Architecture

```
Training:
  CSV (dev_id=2001, tag_cd=30001, 15-min intervals)
    -> Resample to 1-hour intervals
    -> Extract daily peaks (power + slot) as targets
    -> Build 14-day hybrid features per day (~40 features)
    -> Train Model A: features -> peak_power (regression, RMSE)
    -> Train Model B: features -> peak_slot 0-95 (regression, RMSE)
    -> Save: models/2001_power.txt, models/2001_time.txt

Inference (hourly):
  Fetch last 14 days of 15-min data (336 hours)
    -> Resample to 1-hour + build today's partial features
    -> Model A predicts peak_power
    -> Model B predicts peak_slot -> round to int -> HH:MM
    -> Format: "207.33@13:15"
    -> Write to MAX_DMAND_FCST_H table (DB) or CSV (dev mode)
```

## File Structure

| File | Purpose | Change from anomaly_detection |
|------|---------|-------------------------------|
| `train_peak.py` | Training CLI | Rename + adapt for daily targets |
| `ai_peak_runner.py` | Inference runner | Rename + adapt for peak output |
| `infer_peak.py` | Model loading, prediction, formatting | Rewrite for peak power/time |
| `data_preprocessing.py` | Feature engineering | **Major rewrite** -- hourly aggregation + hybrid features |
| `data_source.py` | Data access layer | Adapt read/write for peak tables |
| `_config.json` | Configuration | New parameters for peak prediction |
| `config_peak_devices.csv` | Device config | dev_id=2001 only |

## Feature Engineering (~40 features)

### Input
- 14 days of raw 15-min data (dev_id=2001, tag_cd=30001)
- Resampled to 1-hour means for feature construction
- Daily peaks extracted from raw 15-min data for targets

### Features

| # | Category | Features | Count |
|---|----------|----------|-------|
| 1 | Temporal | weekday, month, is_holiday, sin_month, cos_month, sin_weekday, cos_weekday | 7 |
| 2 | Previous-day peaks | prev_1d_peak_power, prev_1d_peak_slot, prev_2d_peak_power, prev_3d_peak_power | 4 |
| 3 | Same-weekday lag | prev_week_same_day_peak_power, prev_week_same_day_peak_slot | 2 |
| 4 | Rolling peak stats (7d) | mean_peak_7d, std_peak_7d, max_peak_7d, min_peak_7d, mean_peak_slot_7d | 5 |
| 5 | Rolling peak stats (14d) | mean_peak_14d, std_peak_14d, max_peak_14d, trend_peak_14d | 4 |
| 6 | Daily load shape (prev day) | prev_day_mean, prev_day_std, prev_day_min, prev_day_max_min_ratio | 4 |
| 7 | Compressed profile (prev day) | morning_mean (6-12h), afternoon_mean (12-18h), evening_mean (18-24h), night_mean (0-6h) | 4 |
| 8 | Compressed profile (prev week same day) | pw_morning_mean, pw_afternoon_mean, pw_evening_mean, pw_night_mean | 4 |
| 9 | Same-day partial (hourly update) | today_max_so_far, today_max_slot_so_far, today_mean_so_far, hours_elapsed | 4 |
| 10 | Trend | peak_trend_7d (slope), weekday_weekend_peak_ratio | 2 |
| | | **Total** | **~40** |

### Same-day partial features
- At 00:30: hours_elapsed=0, today_max_so_far=0 -> relies on history
- At 13:00: hours_elapsed=13, today_max_so_far=actual_morning_peak -> much more accurate
- Model learns to weight historical vs same-day features based on hours_elapsed

## Output Format

### DB table: MAX_DMAND_FCST_H

| Column | Type | Value |
|--------|------|-------|
| USE_DT (PK) | TIMESTAMP | Current timestamp |
| BLDG_ID (PK, FK) | VARCHAR(10) | "B0019" |
| DLY_MAX_DMAND_FCST_INF | VARCHAR(30) | "207.33@13:15" |

### Formatting logic
```
peak_power = model_power.predict(features)  -> round to 2 decimals
peak_slot  = model_time.predict(features)   -> round to int, clip 0-95
peak_time  = f"{(peak_slot*15)//60:02d}:{(peak_slot*15)%60:02d}"
result     = f"{peak_power:.2f}@{peak_time}"
```

## Configuration (_config.json)

```json
{
  "data_source": "csv",
  "csv": {
    "data_path": "/workspace/2024_AI_BEMS/YiUmGoV2/dataset/data_colec_h_*.csv",
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
    "model_dir": "models",
    "dev_id": 2001,
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

## CLI Interface

```bash
# Training (development)
python peak_prediction/train_peak.py --csv

# Inference with CSV (development)
python peak_prediction/ai_peak_runner.py --csv

# Inference with DB (production)
python peak_prediction/ai_peak_runner.py
```
