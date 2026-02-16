# YiUmGoV2 - AI BEMS Anomaly Detection

LightGBM-based anomaly detection module for the BEMS (Building Energy Management System) platform.
Reads sensor data from PostgreSQL (production) or CSV files (development), engineers 44 time-series features, runs inference with pre-trained models, and writes anomaly scores (AD_SCORE 0-100) and descriptions back to the database.

## Quick Start

```bash
pip install -r requirements.txt

# Train models for enabled devices (reads config_anomaly_devices.csv, skips existing)
python train_anomaly.py --csv

# Run inference (CSV mode, writes to output/)
python ai_anomaly_runner.py --csv

# Run inference (DB mode, production)
python ai_anomaly_runner.py

# Run tests
python -m pytest tests/ -v
```

## Architecture

```
┌──────────────────────────┐     ┌──────────────────────────┐
│  CSV Files (dev mode)    │     │  PostgreSQL (prod mode)  │
│  dataset/*.csv           │     │  DATA_COLEC_H table      │
└────────────┬─────────────┘     └────────────┬─────────────┘
             └──────────┬─────────────────────┘
                        │
                   db_connection.py
            (dual-mode data access layer)
                        │
         ┌──────────────┴──────────────┐
         │                             │
  ai_anomaly_runner.py          train_anomaly.py
  (hourly cron inference)       (multi-device model training)
         │                             │
         └──────────────┬──────────────┘
                        │
              data_preprocessing.py
              (44 time-series features)
                        │
                 infer_anomaly.py
           (prediction, scoring, description)
                        │
              FALT_PRCV_FCST table
              (AD_SCORE, AD_DESC)
```

## Training / Inference Consistency

Training and inference both use identical **176h (7d+8h) windows** through `preprocess(window_df, config)`:

- **Training**: Randomly samples 176h windows from historical CSV data. Each window produces the last 8 samples (2h scoring window). Controlled by `training.max_steps` (default: 1000 windows).
- **Inference**: Fetches the most recent 176h of sensor data (time-based filtering for both CSV and DB modes), preprocesses, and scores the last 2h window.

This guarantees identical feature engineering (lags, rolling stats, seasonality, etc.) between training and inference.

## Feature Engineering (44 Features)

`data_preprocessing.py` transforms raw sensor values into 44 features:

| Group | Count | Features |
|-------|-------|----------|
| Cleansing | 1 | `is_missing` |
| Temporal | 4 | `hour`, `month`, `weekday`, `is_holiday` (Korean holidays) |
| Seasonality | 6 | `sin_month`, `cos_month`, `sine_day`, `cosine_day`, `sin_hour`, `cos_hour` |
| Lags | 5 | `lag_1p`, `lag_2p`, `lag_3p`, `lag_1d_0p`, `lag_1w_0p` |
| Rate of Change | 4 | `rate`, `rate_rate`, `rate_1d`, `rate_rate_1d` |
| 1h Rolling Stats | 4 | `ma_1h`, `max_1h`, `min_1h`, `std_1h` |
| 1d Rolling Stats | 4 | `ma_1d`, `max_1d`, `min_1d`, `std_1d` |
| Prior-Day Stats (1h) | 4 | `p1d_ma_1h`, `p1d_max_1h`, `p1d_min_1h`, `p1d_std_1h` |
| Prior-Week Stats (1h) | 4 | `p1w_ma_1h`, `p1w_max_1h`, `p1w_min_1h`, `p1w_std_1h` |
| MA Rates | 6 | `rate_ma_1h`, `rate_rate_ma_1h`, `rate_p1d_ma_1h`, `rate_rate_p1d_ma_1h`, `rate_p1w_ma_1h`, `rate_rate_p1w_ma_1h` |
| Season Decomp | 2 | `season_ma_1h`, `season_ma_1d` |

## Anomaly Scoring

- **AD_SCORE** (0-100): RMSE-based score normalized against mean of actual values. `100` = perfect match, `0` = maximum deviation. Score at or below `anomaly.score_threshold` (default: 50) indicates anomaly.
- **AD_DESC**: Korean-language statistical summary including status, window label, RMSE, mean, std, min, max. Truncated to 1000 characters (DB constraint).

## Configuration

All settings in `_config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `data_source` | `"csv"` | `"csv"` (development) or `"db"` (production) |
| `csv.data_path` | | Path to sensor data CSV |
| `csv.enabled_devices_path` | | Path to enabled devices CSV |
| `csv.result_path` | | Path for CSV output results |
| `db.host/port/database/user/password` | | PostgreSQL connection settings |
| `data.fetch_window_hours` | `176` | Data window size for training and inference |
| `data.scoring_window_hours` | `2` | Window for AD_SCORE calculation |
| `data.sampling_minutes` | `15` | Data collection interval |
| `data.tag_cd` | `30001` | Sensor tag code to process |
| `training.max_steps` | `1000` | Random windows to sample during training |
| `training.samples_per_window` | `8` | Samples taken from each window (= 2h at 15min) |
| `anomaly.score_threshold` | `50` | Anomaly threshold (score <= threshold = anomaly) |
| `model.*` | | LightGBM hyperparameters (objective, metric, num_leaves, etc.) |

## Project Structure

```
YiUmGoV2/
├── ai_anomaly_runner.py         # Main entry point (cron hourly)
├── train_anomaly.py             # Model training CLI (multi-device, random window sampling)
├── data_preprocessing.py        # 44 time-series features
├── infer_anomaly.py             # Inference, AD_SCORE, AD_DESC
├── db_connection.py             # CSV/DB dual-mode data access
├── utility.py                   # Device name lookup helpers
├── test_integration.py          # Standalone E2E pipeline test
├── _config.json                 # All configuration
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── tests/
│   ├── conftest.py              # Pytest fixtures (config, synthetic data)
│   ├── test_data_preprocessing.py  # Unit tests (44 features validation)
│   └── test_integration.py      # Integration tests (full pipeline)
├── dataset/
│   ├── data_colec_h_*.csv       # Sensor data (15-min intervals)
│   ├── config_anomaly_devices.csv  # Enabled devices (BLDG_ID, DEV_ID, FALT_PRCV_YN)
│   ├── devID_tagCD_map.csv      # Device ID to tag code mapping
│   └── dev_m_*.csv              # Device metadata (name, IP, location, etc.)
├── models/anomaly/              # Trained LightGBM models ({dev_id}.txt)
├── output/                      # CSV-mode inference results
└── docs/
    ├── PostgreSQL_DB_design.tsv # Database schema reference
    └── plans/                   # Design and implementation plans
```

## DB Tables

| Table | Direction | Description |
|-------|-----------|-------------|
| `DATA_COLEC_H` | Read | Sensor data (COLEC_DT, BLDG_ID, DEV_ID, TAG_CD, COLEC_VAL) |
| `DEV_USE_PURP_REL_R` | Read | Device config with `FALT_PRCV_YN` flag ('Y'/'N') |
| `FALT_PRCV_FCST` | Write | Anomaly results (USE_DT, BLDG_ID, DEV_ID, AD_SCORE, AD_DESC) |

## Cron Deployment

```
0 * * * * cd /path/to/YiUmGoV2 && python ai_anomaly_runner.py >> /var/log/ai_anomaly.log 2>&1
```

## Testing

```bash
# Unit tests (feature engineering validation)
python -m pytest tests/test_data_preprocessing.py -v

# Integration tests (full pipeline, CSV mode)
python -m pytest tests/test_integration.py -v

# Standalone E2E test
python test_integration.py

# All tests
python -m pytest tests/ -v
```
