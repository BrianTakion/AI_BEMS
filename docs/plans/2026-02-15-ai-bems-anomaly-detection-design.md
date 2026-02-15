# AI BEMS Anomaly Detection - Architecture Design

## Overview

Deploy an AI anomaly detection module that integrates with the existing Java/PostgresDB BEMS platform. The Python AI reads sensor data and configuration from PostgresDB, runs LightGBM-based anomaly detection, and writes results back to PostgresDB.

## Context

- **Existing system**: Java BEMS platform + PostgresDB on Ubuntu (production)
- **AI module**: Python with LightGBM, deployed as cron jobs on the same server
- **Scope**: AI anomaly detection only (power prediction and aircon control are future work)
- **Scale**: Single building (B0019), ~30 sensors, hourly inference

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JAVA BEMS PLATFORM                        │
│  - Collects sensor data every 15 min → DATA_COLEC_H        │
│  - User enables anomaly detection per sensor via Web UI     │
│  - Reads AI results from FALT_PRCV_FCST for display        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    PostgresDB (shared)
                           │
┌──────────────────────────┴──────────────────────────────────┐
│               PYTHON AI MODULE (Cron, hourly)                │
│                                                              │
│  1. Read DEV_USE_PURP_REL_R where FALT_PRCV_YN = 'Y'       │
│  2. For each enabled (BLDG_ID, DEV_ID):                     │
│     a. Read last 4h sensor data from DATA_COLEC_H           │
│     b. Preprocess (feature engineering)                      │
│     c. Load trained LightGBM model                          │
│     d. Predict expected values                               │
│     e. Compute AD_SCORE (0-100, <=50 = anomaly)             │
│     f. Generate AD_DESC (4h statistical summary)             │
│     g. INSERT into FALT_PRCV_FCST                           │
│  3. Exit                                                     │
└─────────────────────────────────────────────────────────────┘
```

## Database Tables

### Tables Python Reads (managed by Java)

**DATA_COLEC_H** - Sensor collection data (15-min intervals)
| Column | Type | Description |
|--------|------|-------------|
| COLEC_DT | TIMESTAMP | Collection datetime |
| BLDG_ID | VARCHAR(10) | Building ID |
| DEV_ID | VARCHAR(10) | Device ID |
| TAG_CD | VARCHAR(10) | Tag code |
| COLEC_VAL | NUMBER(15,2) | Measured value |

**DEV_USE_PURP_REL_R** - Device config with AI enable flags
| Column | Type | Description |
|--------|------|-------------|
| BLDG_ID (PK,FK) | VARCHAR(10) | Building ID |
| USE_PURP_ID (PK,FK) | VARCHAR(10) | Usage purpose ID |
| DEV_ID (PK) | VARCHAR(10) | Device ID |
| FALT_PRCV_YN | VARCHAR(1) | Anomaly detection enabled ('Y'/'N') |
| AI_APLY_YN | VARCHAR(1) | AI application enabled ('Y'/'N') |
| FOREC_BASE_QNT | NUMBER(15,2) | Forecasted base quantity |
| FOREC_RDUC_QNT | NUMBER(15,2) | Forecasted reduction quantity |
| LNCH_ST_TIME | VARCHAR(5) | Lunch start time |
| LNCH_ED_TIME | VARCHAR(5) | Lunch end time |
| COOL_BASE_TEMPR | NUMBER(10,2) | Cooling base temperature |
| HEAT_BASE_TEMPR | NUMBER(10,2) | Heating base temperature |

### Tables Python Writes

**FALT_PRCV_FCST** - Anomaly detection results
| Column | Type | Description |
|--------|------|-------------|
| USE_DT (PK) | TIMESTAMP | Prediction datetime |
| BLDG_ID (PK,FK) | VARCHAR(10) | Building ID |
| DEV_ID (PK,FK) | VARCHAR(10) | Device ID |
| AD_SCORE | NUMBER(15,2) | Anomaly score (0-100, <=50 = anomaly) |
| AD_DESC | VARCHAR(1000) | 4-hour statistical summary text |

## Project Structure

```
YiUmGoV2/
├── ai_anomaly_runner.py      # Main entry point (cron runs this)
├── db_connection.py          # PostgresDB connection via SQLAlchemy
├── data_preprocessing.py     # Feature engineering
├── utility.py                # Helper functions
├── _config.json              # All configuration
├── models/
│   └── anomaly/
│       ├── 2001.txt          # Trained model per DEV_ID
│       └── ...
├── train_anomaly.py          # Manual training script
└── infer_anomaly.py          # Core inference logic (importable)
```

## Configuration (_config.json)

```json
{
  "db": {
    "host": "localhost",
    "port": 5432,
    "database": "bems",
    "user": "ai_user",
    "password": "****"
  },
  "data": {
    "input_interval_hours": 4,
    "sampling_minutes": 15,
    "collection_table": "DATA_COLEC_H"
  },
  "anomaly": {
    "score_threshold": 50,
    "model_dir": "models/anomaly",
    "config_table": "DEV_USE_PURP_REL_R",
    "result_table": "FALT_PRCV_FCST"
  },
  "model": {
    "num_leaves": 50,
    "learning_rate": 0.1,
    "num_boost_round": 1000,
    "early_stopping_rounds": 30,
    "test_size": 0.03
  }
}
```

Both training and inference scripts read from `_config.json`. No hard-coded values for data window, sampling rate, model hyperparameters, or DB settings.

## Module Descriptions

### ai_anomaly_runner.py (main entry point)
- Loads `_config.json`
- Connects to PostgresDB
- Queries `DEV_USE_PURP_REL_R` for devices with `FALT_PRCV_YN = 'Y'`
- For each enabled device: calls `infer_anomaly.py` logic
- Writes results to `FALT_PRCV_FCST`
- Logs execution summary

### db_connection.py
- Creates SQLAlchemy engine from config
- `read_enabled_devices(engine, config)` -> list of (BLDG_ID, DEV_ID)
- `read_sensor_data(engine, config, bldg_id, dev_id)` -> DataFrame of last 4h data
- `write_anomaly_result(engine, config, result_row)` -> INSERT into FALT_PRCV_FCST

### data_preprocessing.py
- Cleaned-up version of `Data_PreProcessing_260207.py`
- `preprocess(df, config)` -> DataFrame with engineered features
- Uses `input_interval_hours` and `sampling_minutes` from config

### infer_anomaly.py
- `load_model(model_path)` -> LightGBM Booster
- `run_inference(model, X)` -> predictions
- `compute_ad_score(predicted, actual)` -> score (0-100)
- `generate_ad_desc(df_4h)` -> statistical summary text

### train_anomaly.py
- Manual training script (operator runs when needed)
- Reads training data from DB or CSV
- Uses model hyperparameters from `_config.json`
- Saves trained model to `models/anomaly/{DEV_ID}.txt`

### utility.py
- Cleaned-up version of `Utility_260207.py`
- Device name lookup, tag code helpers, visualization

## Deployment

- **Cron schedule**: `0 * * * * cd /path/to/YiUmGoV2 && python ai_anomaly_runner.py`
- **Runs hourly**, processes all enabled sensors sequentially
- **Manual retraining**: operator runs `python train_anomaly.py --dev_id 2001` as needed

## AD_SCORE Calculation

- Run LightGBM model on 4-hour window of data
- Compare predicted vs actual values
- Compute RMSE over the 4-hour window
- Normalize RMSE to 0-100 scale (100 = perfect match, 0 = maximum deviation)
- Score <= 50 triggers anomaly flag

## AD_DESC Format

4-hour statistical summary text including:
- Mean, std, min, max of actual values
- Mean prediction error
- Trend direction (increasing/decreasing/stable)
- Anomaly description if score <= 50
