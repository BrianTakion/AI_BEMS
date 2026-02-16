# YiUmGoV2 - AI BEMS Anomaly Detection

LightGBM-based anomaly detection module for the BEMS platform.
Reads sensor data, runs inference, and writes anomaly scores (AD_SCORE 0-100) to PostgresDB.

## Quick Start

```bash
pip install -r requirements.txt

# Train a model for device 2001 (CSV-only, random 176h window sampling)
python train_anomaly.py --dev_id 2001

# Train with explicit date range
python train_anomaly.py --dev_id 2001 --start_date 2025-03-24 --end_date 2025-09-09

# Run inference (dry-run, no DB write)
python ai_anomaly_runner.py --dry-run

# Run inference (production, writes to DB)
python ai_anomaly_runner.py
```

## Training / Inference Consistency

Training and inference both use identical **176h (7d+8h) windows** through `preprocess(window_df, config)`:

- **Training**: Randomly samples 176h windows from historical CSV data. Each window produces the last 8 samples (2h scoring window). Controlled by `training.max_steps` in config.
- **Inference**: Fetches the most recent 176h of data (time-based filtering for both CSV and DB modes), preprocesses, and scores the last 2h window.

This guarantees identical feature engineering (lags, rolling stats, etc.) between training and inference.

## E2E Test (CSV-only, no DB required)

```bash
python manual_train_anomaly__run_e2e_test.py
```

Automatically processes **all** devices from `enabled_devices.csv` where `FALT_PRCV_YN='Y'`:

1. **Train** models for each enabled device (skips if `models/anomaly/{dev_id}.txt` already exists)
2. **Infer** — runs `ai_anomaly_runner.py` once (processes all enabled devices)
3. **Validate** — checks `anomaly_results.csv` for every enabled device (AD_SCORE, AD_DESC, USE_DT)

Prints PASS/FAIL summary at the end. Exits with code 0 on success, 1 on failure.

## Cron Deployment (hourly)

```
0 * * * * cd /path/to/YiUmGoV2 && python ai_anomaly_runner.py >> /var/log/ai_anomaly.log 2>&1
```

## Configuration

All settings in `_config.json`:

| Key | Description |
|-----|-------------|
| `data_source` | `"csv"` (dev) or `"db"` (production) |
| `data.fetch_window_hours` | Data window size for both training and inference (default: 176h) |
| `data.scoring_window_hours` | Anomaly scoring window (default: 2h) |
| `data.sampling_minutes` | Sampling interval (default: 15min) |
| `training.max_steps` | Number of random windows to sample during training (default: 1000) |
| `training.samples_per_window` | Samples taken from each window (default: 8, matching 2h) |
| `anomaly.score_threshold` | Anomaly threshold (default: 50, lower = anomaly) |
| `model.*` | LightGBM hyperparameters |

## Project Structure

```
YiUmGoV2/
├── ai_anomaly_runner.py      # Main entry point (cron)
├── db_connection.py          # CSV/DB dual-mode data access
├── data_preprocessing.py     # 44 time-series features
├── infer_anomaly.py          # Inference, AD_SCORE, AD_DESC
├── train_anomaly.py          # Model training CLI (CSV-only, random window sampling)
├── utility.py                # Device name lookup helpers
├── manual_train_anomaly__run_e2e_test.py  # E2E test (CSV-only, no DB)
├── test_integration.py       # End-to-end pipeline test
├── _config.json              # All configuration
├── requirements.txt          # Python dependencies
└── models/anomaly/           # Trained LightGBM models
```

## DB Tables

- **Reads**: `DATA_COLEC_H` (sensor data), `DEV_USE_PURP_REL_R` (device config)
- **Writes**: `FALT_PRCV_FCST` (AD_SCORE, AD_DESC per device)
