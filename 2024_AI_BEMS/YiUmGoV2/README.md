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
                   data_source.py
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

## Modules

| File | Description |
|------|-------------|
| `ai_anomaly_runner.py` | Main entry point (hourly cron). Loops over enabled devices, loads models, fetches data, preprocesses, infers, and writes results. |
| `train_anomaly.py` | Training CLI (`--csv` required). Loads all historical sensor data via `data_source`, samples random 176h windows, trains one LightGBM model per device. Skips if model already exists. |
| `data_preprocessing.py` | Feature engineering. Transforms a raw 176h sensor window into 44 features (temporal, seasonal, lag, rolling stats, rates, decomposition). |
| `infer_anomaly.py` | Inference utilities. Loads a LightGBM Booster, runs predictions, computes AD_SCORE (0-100), and generates a Korean-language AD_DESC. |
| `data_source.py` | Dual-mode data access layer. Abstracts CSV vs PostgreSQL for reading devices, sensor data, and writing anomaly results. Supports `fetch_hours=0` (all history for training) and `fetch_hours=None` (config default for inference). |

### CSV Mode (`--csv`)

Development/testing mode. All data is read from local files, no database required.

```
train_anomaly.py --csv
  ├─ read enabled devices from   dataset/config_anomaly_devices.csv
  ├─ load sensor history from    dataset/data_colec_h_*.csv  (chunked, 500K rows)
  ├─ sample random 176h windows → data_preprocessing.preprocess()
  ├─ train LightGBM per device
  └─ save model to               models/anomaly/{dev_id}.txt

ai_anomaly_runner.py --csv
  ├─ read enabled devices from   dataset/config_anomaly_devices.csv
  ├─ load sensor data from       dataset/data_colec_h_*.csv  (last 176h by time)
  ├─ preprocess → infer → score last 2h window
  └─ write results to            output/anomaly_results.csv
```

### DB Mode (production)

Production mode. Reads from and writes to PostgreSQL.

```
ai_anomaly_runner.py  (no flag, hourly cron)
  ├─ read enabled devices from   DEV_USE_PURP_REL_R  (FALT_PRCV_YN = 'Y')
  ├─ query sensor data from      DATA_COLEC_H        (last 176h by COLEC_DT)
  ├─ preprocess → infer → score last 2h window
  └─ insert results into         FALT_PRCV_FCST      (AD_SCORE, AD_DESC)
```

## Training / Inference Consistency

Training and inference both use identical **176h (7d+8h) windows** through `preprocess(window_df, config)`:

- **Training**: Randomly samples 176h windows from historical CSV data. Each window produces the last 8 samples (2h scoring window). Controlled by `training.max_steps` (default: 1000 windows, ~8000 total samples per device).
- **Inference**: Fetches the most recent 176h of sensor data (time-based filtering for both CSV and DB modes), preprocesses, and scores the last 2h window.

Both paths use `fill_method="ffill"` (forward-fill then backward-fill) for missing value interpolation, ensuring identical feature engineering (lags, rolling stats, seasonality, etc.) between training and inference.

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
| Prior-Day Stats (±30min) | 4 | `p1d_ma_1h`, `p1d_max_1h`, `p1d_min_1h`, `p1d_std_1h` |
| Prior-Week Stats (±30min) | 4 | `p1w_ma_1h`, `p1w_max_1h`, `p1w_min_1h`, `p1w_std_1h` |
| MA Rates | 6 | `rate_ma_1h`, `rate_rate_ma_1h`, `rate_p1d_ma_1h`, `rate_rate_p1d_ma_1h`, `rate_p1w_ma_1h`, `rate_rate_p1w_ma_1h` |
| Season Decomp | 2 | `season_ma_1h`, `season_ma_1d` |

## Anomaly Scoring

- **AD_SCORE** (0-100): `score = max(0, 100 - (RMSE / mean(|y_actual|)) * 100)`. A score of `100` means perfect prediction match (no anomaly); `0` means maximum deviation. Score at or below `anomaly.score_threshold` (default: 50) indicates anomaly.
- **AD_DESC**: Korean-language statistical summary including status (정상/이상), window label, AD_SCORE, RMSE, mean, std, max, min. Truncated to 1000 characters (DB VARCHAR constraint).

## Configuration

All settings in `_config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `data_source` | `"csv"` | `"csv"` (development) or `"db"` (production) |
| `csv.data_path` | | Path to sensor data CSV |
| `csv.config_anomaly_devices_path` | | Path to enabled devices CSV |
| `csv.anomaly_results_path` | | Path for CSV output results |
| `db.host/port/database/user/password` | | PostgreSQL connection settings |
| `data.fetch_window_hours` | `176` | Data window size for feature engineering (7d + 8h for lags) |
| `data.scoring_window_hours` | `2` | Window for AD_SCORE calculation (last 8 points at 15min) |
| `data.sampling_minutes` | `15` | Data collection interval |
| `data.collection_table` | `"DATA_COLEC_H"` | DB table for sensor data |
| `data.tag_cd` | `30001` | Sensor tag code to process |
| `training.max_steps` | `1000` | Random windows to sample during training |
| `training.samples_per_window` | `8` | Samples taken from each window (= 2h at 15min) |
| `anomaly.score_threshold` | `50` | Anomaly threshold (score <= threshold = anomaly) |
| `anomaly.model_dir` | `"models/anomaly"` | Directory for trained model files |
| `anomaly.config_table` | `"DEV_USE_PURP_REL_R"` | DB table for device enablement config |
| `anomaly.result_table` | `"FALT_PRCV_FCST"` | DB table for anomaly detection results |
| `model.*` | | LightGBM hyperparameters (objective, metric, num_leaves, etc.) |

## Project Structure

```
YiUmGoV2/
├── ai_anomaly_runner.py         # Main entry point (cron hourly)
├── train_anomaly.py             # Model training CLI (multi-device, random window sampling)
├── data_preprocessing.py        # 44 time-series features
├── infer_anomaly.py             # Inference, AD_SCORE, AD_DESC
├── data_source.py               # CSV/DB dual-mode data access
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

# All tests
python -m pytest tests/ -v
```
