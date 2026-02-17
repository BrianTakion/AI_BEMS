# Peak Power Prediction Service

Predicts today's **peak power demand (kW)** and **when it will occur (HH:MM)** for a building, using 14 days of 15-minute interval sensor data. Runs hourly and refines its prediction as more same-day data accumulates.

Output format: `"154.58@12:45"` (peak_kW@HH:MM)

## Directory Structure

```
peak_prediction/
├── _config.json                 # Central configuration (data source, LightGBM params, paths)
├── config_peak_devices.csv      # Enabled devices (B0019/2001)
├── data_source.py               # Dual-mode data access layer (CSV/PostgreSQL)
├── data_preprocessing.py        # Feature engineering (~40 features from 14-day window)
├── infer_peak.py                # Model loading, prediction, result formatting
├── train_peak.py                # CLI training script (dual LightGBM models)
├── ai_peak_runner.py            # Hourly inference runner (production entry point)
├── models/
│   ├── 2001_power.txt           # Trained LightGBM model: peak kW
│   └── 2001_time.txt            # Trained LightGBM model: peak time slot
├── output/
│   └── peak_results.csv         # Inference output (CSV mode)
└── docs/
    ├── PostgreSQL_DB_design.tsv
    ├── checkpoint_csv_to_db_migration.md
    └── plans/
        └── 2026-02-17-peak-prediction-design.md
```

## Quick Start

```bash
# Train models (CSV mode, skips if models already exist)
python peak_prediction/train_peak.py --csv

# Run inference once (CSV mode)
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
  (hourly cron inference)       (one-time training)
         │                             │
         │  data_preprocessing.py      │  data_preprocessing.py
         │  (~40 features)             │  (~40 features)
         │                             │
         ▼                             ▼
  infer_peak.py                 models/
  (predict + format)            {dev_id}_power.txt
         │                      {dev_id}_time.txt
         ▼
  "154.58@12:45"
  → output/peak_results.csv (CSV mode)
  → MAX_DMAND_FCST_H table  (DB mode)
```

### Data Flow

1. **Read** 14 days of 15-min power readings via `data_source.read_sensor_data()`
2. **Preprocess** into ~40 features via `data_preprocessing.preprocess_for_inference()`
   - Normalize to 15-min grid, interpolate gaps
   - Extract daily peaks (max kW + 15-min slot index 0-95)
   - Build temporal, lag, rolling, load-shape, and same-day partial features
3. **Predict** with two separate LightGBM models via `infer_peak.predict_peak()`
   - Power model: predicts peak kW
   - Time model: predicts peak slot (0-95, where slot = hour\*4 + minute//15)
4. **Format** as `"kW@HH:MM"` via `infer_peak.format_peak_result()`
5. **Write** to CSV file or DB table via `data_source.write_peak_result()`

### Key Design Decisions

- **Two separate models** for power and time rather than multi-output regression. Each model gets its own early stopping point.
- **Time as discrete slot index (0-95):** 96 fifteen-minute slots per day. Slot 51 = 12:45. The model output is rounded and clipped to [0, 95].
- **Same-day partial feature refinement:** `hours_elapsed`, `today_max_so_far`, `today_mean_so_far` improve as the day progresses. An 8 AM run sees ~8h of data; a 4 PM run sees ~16h.
- **Dual-mode data access:** `data_source.py` dispatches on `config["data_source"]` (`"csv"` or `"db"`). Core logic is mode-agnostic.
- **Path anchoring to script directory:** All relative paths resolve from `peak_prediction/`, not the caller's working directory.

## Feature Engineering (~40 Features)

| Group | Count | Features |
|-------|-------|----------|
| Temporal | 7 | weekday, month, is_holiday, sin/cos encodings |
| Previous-day peaks | 4 | peak power/slot from prior 1-3 days |
| Same-weekday lag | 2 | peak power/slot from same weekday last week |
| Rolling peak stats 7d | 5 | mean, std, max, min peak; mean peak slot |
| Rolling peak stats 14d | 4 | mean, std, max peak; trend (linear slope) |
| Daily load shape | 4 | prev day mean, std, min, max/min ratio |
| Compressed profile (prev day) | 4 | night/morning/afternoon/evening means |
| Compressed profile (same weekday last week) | 4 | night/morning/afternoon/evening means |
| Same-day partial | 4 | today_max_so_far, today_max_slot_so_far, today_mean_so_far, hours_elapsed |
| Trend | 2 | peak_trend_7d slope, weekday/weekend peak ratio |

Korean holidays (including Seollal/Chuseok extended periods and substitute holidays) are handled via the `holidays` library.

## Training Results (dev_id=2001)

| Model | RMSE | MAE |
|-------|------|-----|
| Power (kW) | 27.05 | 21.38 |
| Time (slots) | 5.79 (~87 min) | ~68 min |

Training uses all available history (`fetch_hours=0`), 15% test split, LightGBM with 500 rounds max and early stopping (patience=30).

**Note:** `train_peak.py` skips training if both model files already exist. Delete existing model files to retrain.

## Configuration

All settings in `_config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `data_source` | `"csv"` | `"csv"` (development) or `"db"` (production) |
| `data.fetch_window_hours` | `336` | 14 days of history for lag features |
| `data.sampling_minutes` | `15` | Data collection interval |
| `data.tag_cd` | `30001` | Power meter sensor tag |
| `training.min_history_days` | `14` | Minimum days needed for training |
| `model.num_boost_round` | `500` | Max boosting iterations |
| `model.early_stopping_rounds` | `30` | Early stopping patience |
| `model.test_size` | `0.15` | Train/test split ratio |
| `peak.dev_id` | `2001` | Main power meter device ID |
| `peak.result_table` | `MAX_DMAND_FCST_H` | DB table for results |

## DB Tables

| Table | Direction | Key Columns |
|-------|-----------|-------------|
| `DATA_COLEC_H` | Read | COLEC_DT, BLDG_ID, DEV_ID, TAG_CD, COLEC_VAL |
| `DEV_USE_PURP_REL_R` | Read | BLDG_ID, DEV_ID, FALT_PRCV_YN (device enablement) |
| `MAX_DMAND_FCST_H` | Write | USE_DT, BLDG_ID, DLY_MAX_DMAND_FCST_INF (e.g. `"154.58@12:45"`) |

## Dependencies

- Python 3.8+
- `lightgbm` - gradient boosting models
- `pandas` - data manipulation
- `numpy` - numerical operations
- `scikit-learn` - train/test split, metrics
- `holidays` - Korean holiday calendar
- `sqlalchemy`, `psycopg2` - PostgreSQL access (production mode only)

## Deployment

```bash
# Hourly cron job (production)
0 * * * * cd /path/to/YiUmGoV2 && /path/to/venv/bin/python peak_prediction/ai_peak_runner.py >> /var/log/ai_peak.log 2>&1
```

Use the absolute path to the virtualenv Python interpreter in cron.

## Module Reference

| Module | Public Functions |
|--------|-----------------|
| `data_source` | `create_data_source()`, `read_enabled_devices()`, `read_sensor_data()`, `write_peak_result()` |
| `data_preprocessing` | `preprocess_for_training()`, `preprocess_for_inference()` |
| `infer_peak` | `load_model()`, `predict_peak()`, `format_peak_result()` |
| `train_peak` | `train_model()` + CLI entry point |
| `ai_peak_runner` | `process_device()` + CLI entry point |
