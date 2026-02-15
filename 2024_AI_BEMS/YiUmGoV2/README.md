# YiUmGoV2 - AI BEMS Anomaly Detection

LightGBM-based anomaly detection module for the BEMS platform.
Reads sensor data, runs inference, and writes anomaly scores (AD_SCORE 0-100) to PostgresDB.

## Quick Start

```bash
pip install -r requirements.txt

# Train a model for device 2001
python train_anomaly.py --dev_id 2001

# Run inference (dry-run, no DB write)
python ai_anomaly_runner.py --dry-run

# Run inference (production, writes to DB)
python ai_anomaly_runner.py
```

## Cron Deployment (hourly)

```
0 * * * * cd /path/to/YiUmGoV2 && python ai_anomaly_runner.py >> /var/log/ai_anomaly.log 2>&1
```

## Configuration

All settings in `_config.json`:

| Key | Description |
|-----|-------------|
| `data_source` | `"csv"` (dev) or `"db"` (production) |
| `data.input_interval_hours` | Scoring window (default: 4h) |
| `data.sampling_minutes` | Sampling interval (default: 15min) |
| `anomaly.score_threshold` | Anomaly threshold (default: 50, lower = anomaly) |
| `model.*` | LightGBM hyperparameters |

## Project Structure

```
YiUmGoV2/
├── ai_anomaly_runner.py      # Main entry point (cron)
├── db_connection.py          # CSV/DB dual-mode data access
├── data_preprocessing.py     # 60+ time-series features
├── infer_anomaly.py          # Inference, AD_SCORE, AD_DESC
├── train_anomaly.py          # Model training CLI
├── utility.py                # Device name lookup helpers
├── test_integration.py       # End-to-end pipeline test
├── _config.json              # All configuration
├── requirements.txt          # Python dependencies
└── models/anomaly/           # Trained LightGBM models
```

## DB Tables

- **Reads**: `DATA_COLEC_H` (sensor data), `DEV_USE_PURP_REL_R` (device config)
- **Writes**: `FALT_PRCV_FCST` (AD_SCORE, AD_DESC per device)
