# Peak Prediction Service - Complete

**All files in `peak_prediction/`:**

| File | Purpose |
|------|---------|
| `_config.json` | Configuration (336h window, LightGBM params, DB/CSV paths) |
| `config_peak_devices.csv` | Device list (B0019/2001) |
| `data_preprocessing.py` | ~40 hybrid features from 14 days of hourly data |
| `infer_peak.py` | Model loading, prediction, "207.33@13:15" formatting |
| `data_source.py` | CSV/DB data access layer |
| `train_peak.py` | Dual-model LightGBM training CLI |
| `ai_peak_runner.py` | Hourly inference runner (CSV/DB modes) |

## Training results (dev_id=2001)

- Power model: RMSE=27.05 kW, MAE=21.38 kW
- Time model: RMSE=5.79 slots (~68 min avg error)

## Inference verified

`154.58@12:45` written to `output/peak_results.csv` in correct PostgreSQL format.

## Quick Start

```bash
# Train models (CSV mode, development)
python peak_prediction/train_peak.py --csv

# Run inference (CSV mode, development)
python peak_prediction/ai_peak_runner.py --csv

# Run inference (DB mode, production)
python peak_prediction/ai_peak_runner.py
```

## Architecture

```
┌──────────────────────────┐     ┌──────────────────────────┐
│  CSV Files (dev mode)    │     │  PostgreSQL (prod mode)  │
│  dataset/*.csv           │     │  DATA_COLEC_H table      │
└────────────┬─────────────┘     └────────────┬─────────────┘
             └──────────┬─────────────────────┘
                        │
            data_source.py
            (dual-mode data access layer)
                        │
         ┌──────────────┴──────────────┐
         │                             │
  ai_peak_runner.py             train_peak.py
  (hourly cron inference)       (dual-model training)
         │                             │
         │  data_preprocessing.py      │  data_preprocessing.py
         │  (~40 hybrid features)      │  (~40 hybrid features)
         │                             │
         ▼                             ▼
  infer_peak.py                 models/
  (predict + format)            ({dev_id}_power.txt)
         │                      ({dev_id}_time.txt)
         ▼
  MAX_DMAND_FCST_H table
  (DLY_MAX_DMAND_FCST_INF = "207.33@13:15")
```

## Feature Engineering (~40 Features)

| Group | Count | Features |
|-------|-------|----------|
| Temporal | 7 | weekday, month, is_holiday, sin/cos encodings |
| Previous-day peaks | 4 | peak power/slot from prior 1-3 days |
| Same-weekday lag | 2 | peak power/slot from same weekday last week |
| Rolling peak stats 7d | 5 | mean, std, max, min peak; mean peak slot |
| Rolling peak stats 14d | 4 | mean, std, max peak; trend (slope) |
| Daily load shape | 4 | prev day mean, std, min, max/min ratio |
| Compressed profile prev day | 4 | morning/afternoon/evening/night means |
| Compressed profile prev week same day | 4 | morning/afternoon/evening/night means |
| Same-day partial | 4 | today_max_so_far, today_max_slot_so_far, today_mean_so_far, hours_elapsed |
| Trend | 2 | peak_trend_7d slope, weekday/weekend peak ratio |

## DB Tables

| Table | Direction | Description |
|-------|-----------|-------------|
| `DATA_COLEC_H` | Read | Sensor data (COLEC_DT, BLDG_ID, DEV_ID, TAG_CD, COLEC_VAL) |
| `DEV_USE_PURP_REL_R` | Read | Device config with peak prediction enablement |
| `MAX_DMAND_FCST_H` | Write | Peak results (USE_DT, BLDG_ID, DLY_MAX_DMAND_FCST_INF) |

## Configuration

All settings in `_config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `data_source` | `"csv"` | `"csv"` (development) or `"db"` (production) |
| `data.fetch_window_hours` | `336` | 14 days of history for lag features |
| `data.sampling_minutes` | `15` | Data collection interval |
| `training.min_history_days` | `14` | Minimum days needed for training |
| `model.test_size` | `0.15` | Train/test split ratio |
| `peak.dev_id` | `2001` | Main power meter device ID |
| `peak.result_table` | `"MAX_DMAND_FCST_H"` | DB table for peak prediction results |

## Cron Deployment

```bash
# Register hourly cron job
crontab -e
# Add:
0 * * * * cd /path/to/YiUmGoV2 && python peak_prediction/ai_peak_runner.py >> /var/log/ai_peak.log 2>&1
```
