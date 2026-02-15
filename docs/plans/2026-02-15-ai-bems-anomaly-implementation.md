# AI BEMS Anomaly Detection - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `YiUmGoV2/` — a production-ready Python module that reads sensor data from PostgresDB, runs LightGBM anomaly detection, and writes results back to PostgresDB.

**Architecture:** Single cron-triggered Python script (`ai_anomaly_runner.py`) orchestrates the pipeline: reads enabled device list from `DEV_USE_PURP_REL_R`, fetches 4h of sensor data from `DATA_COLEC_H`, runs LightGBM inference, computes anomaly scores, and inserts results into `FALT_PRCV_FCST`. All parameters come from `_config.json`.

**Tech Stack:** Python 3.10+, LightGBM, pandas, numpy, SQLAlchemy, psycopg2, scipy, pywt, holidays

**Reference design:** `docs/plans/2026-02-15-ai-bems-anomaly-detection-design.md`

**Source code reference (v1):**
- `YiUmGO/Anomaly_260207/Power_Prediction_train.py` — existing training script
- `YiUmGO/Anomaly_260207/Power_Prediction_inference.py` — existing inference script
- `YiUmGO/Data_PreProcessing_260207.py` — feature engineering (50+ features)
- `YiUmGO/Utility_260207.py` — helper functions
- `YiUmGO/PostgreSQL_DB_design.tsv` — DB schema specification

---

### Task 1: Project scaffold and configuration

**Files:**
- Create: `YiUmGoV2/_config.json`
- Create: `YiUmGoV2/requirements.txt`
- Create: `YiUmGoV2/models/anomaly/.gitkeep`

**Step 1: Create directory structure**

```bash
mkdir -p YiUmGoV2/models/anomaly
touch YiUmGoV2/models/anomaly/.gitkeep
```

**Step 2: Create `_config.json`**

```json
{
  "db": {
    "host": "localhost",
    "port": 5432,
    "database": "bems",
    "user": "ai_user",
    "password": "changeme"
  },
  "data": {
    "input_interval_hours": 4,
    "sampling_minutes": 15,
    "collection_table": "DATA_COLEC_H",
    "tag_cd": 30001
  },
  "anomaly": {
    "score_threshold": 50,
    "model_dir": "models/anomaly",
    "config_table": "DEV_USE_PURP_REL_R",
    "result_table": "FALT_PRCV_FCST"
  },
  "model": {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 50,
    "learning_rate": 0.1,
    "num_boost_round": 1000,
    "early_stopping_rounds": 30,
    "test_size": 0.03,
    "verbose": -1
  }
}
```

**Step 3: Create `requirements.txt`**

```
pandas>=2.0
numpy>=1.24
lightgbm>=4.0
scikit-learn>=1.3
sqlalchemy>=2.0
psycopg2-binary>=2.9
scipy>=1.11
PyWavelets>=1.4
holidays>=0.34
plotly>=5.15
```

**Step 4: Install dependencies**

Run: `pip install -r YiUmGoV2/requirements.txt`

**Step 5: Commit**

```bash
git add YiUmGoV2/
git commit -m "scaffold: create YiUmGoV2 project structure and config"
```

---

### Task 2: Utility module

Port the needed helper functions from `YiUmGO/Utility_260207.py`. Only include functions used by anomaly detection (drop visualization-only helpers).

**Files:**
- Create: `YiUmGoV2/utility.py`
- Reference: `YiUmGO/Utility_260207.py`

**Step 1: Create `utility.py`**

Port these functions from v1:
- `get_device_name(df_dev_map, devID)` — lookup device name from mapping DataFrame
- `select_devID_tagCD(df, devID, tagCD)` — filter DataFrame by device and tag
- `print_tagCD(df, devID)` — print available tag codes for a device

Keep signatures identical to v1. These functions are used by the preprocessing and training pipelines.

```python
import pandas as pd

def get_device_name(df_dev_map: pd.DataFrame, devID: int) -> str:
    """dev_id에 해당하는 dev_nm(장치명)을 매핑 DataFrame에서 조회"""
    row = df_dev_map[df_dev_map['dev_id'] == devID]
    if row.empty:
        raise ValueError(f"devID {devID}에 해당하는 장치를 찾을 수 없습니다.")
    return row.iloc[0]['dev_nm']

def print_tagCD(df: pd.DataFrame, devID: int) -> None:
    df_devID = df[df['dev_id'] == devID]
    print('tagCD list: ', df_devID['tag_cd'].unique())

def select_devID_tagCD(df: pd.DataFrame, devID: int, tagCD: int) -> pd.DataFrame:
    return df[(df['dev_id'] == devID) & (df['tag_cd'] == tagCD)]
```

**Step 2: Verify utility works with existing CSV data**

Run: `cd YiUmGoV2 && python -c "import utility; print('utility module OK')"`
Expected: `utility module OK`

**Step 3: Commit**

```bash
git add YiUmGoV2/utility.py
git commit -m "feat: add utility module with device lookup helpers"
```

---

### Task 3: Data preprocessing module

Port `YiUmGO/Data_PreProcessing_260207.py` into `YiUmGoV2/data_preprocessing.py`. Key change: replace hard-coded `points=4` and `freqInterval='15min'` defaults with values from `_config.json`.

**Files:**
- Create: `YiUmGoV2/data_preprocessing.py`
- Reference: `YiUmGO/Data_PreProcessing_260207.py` (entire file — 210 lines)

**Step 1: Create `data_preprocessing.py`**

Copy the `preprocess()` function from `YiUmGO/Data_PreProcessing_260207.py` with these changes:
- Add a `config` parameter: `preprocess(raw_df, config, only_cleansing=False, fill_method='zero')`
- Derive `points` from config: `points = 60 // config['data']['sampling_minutes']` (e.g., 60/15 = 4 points per hour)
- Derive `freqInterval` from config: `freqInterval = f"{config['data']['sampling_minutes']}min"`
- All feature engineering logic remains identical to v1

The function signature becomes:
```python
def preprocess(raw_df, config, only_cleansing=False, fill_method='zero'):
    points = 60 // config['data']['sampling_minutes']
    freqInterval = f"{config['data']['sampling_minutes']}min"
    # ... rest identical to v1's preprocess()
```

All imports from v1 must be preserved:
```python
import pandas as pd
import numpy as np
import re
import holidays
import datetime
from scipy.fftpack import fft
import pywt
from scipy.signal import find_peaks
from scipy import stats
```

**Step 2: Quick smoke test with existing data**

```python
# test_preprocessing.py (temporary, run from YiUmGoV2/)
import json, pandas as pd
import data_preprocessing as DP

with open('_config.json') as f:
    config = json.load(f)

# Create a small synthetic test DataFrame
idx = pd.date_range('2025-06-01', periods=96*8, freq='15min')  # 8 days
df = pd.DataFrame({'value': range(len(idx))}, index=idx)
X, y, nan_counts, missing_ratio = DP.preprocess(df, config)
print(f"Features: {X.shape[1]}, Samples: {len(X)}, Missing: {missing_ratio}")
assert X.shape[1] > 40, "Expected 40+ features"
print("preprocessing OK")
```

Run: `cd YiUmGoV2 && python test_preprocessing.py`
Expected: `Features: XX, Samples: XX, Missing: 0.0` followed by `preprocessing OK`

**Step 3: Remove temporary test file and commit**

```bash
rm YiUmGoV2/test_preprocessing.py
git add YiUmGoV2/data_preprocessing.py
git commit -m "feat: add data preprocessing module with config-driven parameters"
```

---

### Task 4: Database connection module

Create `db_connection.py` with SQLAlchemy-based functions to read/write from PostgresDB. Include a CSV fallback mode for development/testing without a live DB.

**Files:**
- Create: `YiUmGoV2/db_connection.py`
- Reference: `YiUmGO/PostgreSQL_DB_design.tsv` for table/column names

**Step 1: Create `db_connection.py`**

```python
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

def create_db_engine(config):
    """Create SQLAlchemy engine from config['db'] section."""
    db = config['db']
    url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"
    return create_engine(url)

def read_enabled_devices(engine, config):
    """
    Query DEV_USE_PURP_REL_R for devices with FALT_PRCV_YN = 'Y'.
    Returns list of dicts: [{'bldg_id': 'B0019', 'dev_id': '2001'}, ...]
    """
    table = config['anomaly']['config_table']
    query = text(f'SELECT "BLDG_ID", "DEV_ID" FROM "{table}" WHERE "FALT_PRCV_YN" = :flag')
    with engine.connect() as conn:
        result = conn.execute(query, {"flag": "Y"})
        return [{"bldg_id": row[0], "dev_id": row[1]} for row in result]

def read_sensor_data(engine, config, bldg_id, dev_id):
    """
    Read last N hours of sensor data from DATA_COLEC_H.
    Returns DataFrame with columns: colec_dt, colec_val
    """
    table = config['data']['collection_table']
    hours = config['data']['input_interval_hours']
    tag_cd = config['data']['tag_cd']
    cutoff = datetime.now() - timedelta(hours=hours)

    query = text(f"""
        SELECT "COLEC_DT", "COLEC_VAL"
        FROM "{table}"
        WHERE "BLDG_ID" = :bldg_id
          AND "DEV_ID" = :dev_id
          AND "TAG_CD" = :tag_cd
          AND "COLEC_DT" >= :cutoff
        ORDER BY "COLEC_DT"
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            "bldg_id": bldg_id, "dev_id": dev_id,
            "tag_cd": str(tag_cd), "cutoff": cutoff
        })
    df.columns = ['colec_dt', 'colec_val']
    df['colec_dt'] = pd.to_datetime(df['colec_dt'])
    return df

def write_anomaly_result(engine, config, bldg_id, dev_id, ad_score, ad_desc):
    """
    INSERT one row into FALT_PRCV_FCST.
    """
    table = config['anomaly']['result_table']
    query = text(f"""
        INSERT INTO "{table}" ("USE_DT", "BLDG_ID", "DEV_ID", "AD_SCORE", "AD_DESC")
        VALUES (:use_dt, :bldg_id, :dev_id, :ad_score, :ad_desc)
        ON CONFLICT ("USE_DT", "BLDG_ID", "DEV_ID") DO UPDATE
        SET "AD_SCORE" = :ad_score, "AD_DESC" = :ad_desc
    """)
    with engine.connect() as conn:
        conn.execute(query, {
            "use_dt": datetime.now(),
            "bldg_id": bldg_id,
            "dev_id": dev_id,
            "ad_score": float(ad_score),
            "ad_desc": str(ad_desc)[:1000]
        })
        conn.commit()
```

**Note on column name casing:** PostgresDB table/column names from the design spec are UPPERCASE. SQLAlchemy requires double-quoting to preserve case in PostgreSQL. If the actual DB uses lowercase, adjust the quotes accordingly. This should be verified during deployment.

**Step 2: Verify module imports cleanly**

Run: `cd YiUmGoV2 && python -c "import db_connection; print('db_connection module OK')"`
Expected: `db_connection module OK`

**Step 3: Commit**

```bash
git add YiUmGoV2/db_connection.py
git commit -m "feat: add DB connection module with read/write helpers"
```

---

### Task 5: Inference module

Create `infer_anomaly.py` with the core anomaly detection logic: load model, run prediction, compute AD_SCORE, generate AD_DESC.

**Files:**
- Create: `YiUmGoV2/infer_anomaly.py`
- Reference: `YiUmGO/Anomaly_260207/Power_Prediction_inference.py`

**Step 1: Create `infer_anomaly.py`**

```python
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

def load_model(model_path):
    """Load a trained LightGBM model from .txt file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return lgb.Booster(model_file=model_path)

def run_inference(model, X_df):
    """Run LightGBM prediction on feature DataFrame."""
    return model.predict(X_df)

def compute_ad_score(y_actual, y_predicted, config):
    """
    Compute anomaly detection score (0-100).
    100 = perfect match (no anomaly), 0 = maximum deviation.
    Score <= threshold (default 50) means anomaly detected.

    Method: RMSE-based scoring. Normalize RMSE against the mean of actual values.
    score = max(0, 100 - (rmse / mean_actual) * 100)
    """
    rmse = root_mean_squared_error(y_actual, y_predicted)
    mean_actual = np.abs(y_actual).mean()
    if mean_actual < 1e-6:
        return 100.0 if rmse < 1e-6 else 0.0
    normalized_error = rmse / mean_actual
    score = max(0.0, min(100.0, 100.0 - normalized_error * 100.0))
    return round(score, 2)

def generate_ad_desc(y_actual, y_predicted, ad_score, config):
    """
    Generate 4-hour statistical summary text for AD_DESC column.
    Max 1000 chars (VARCHAR(1000) limit).
    """
    rmse = root_mean_squared_error(y_actual, y_predicted)
    threshold = config['anomaly']['score_threshold']

    # Trend: compare first half vs second half mean
    mid = len(y_actual) // 2
    first_half_mean = np.mean(y_actual[:mid]) if mid > 0 else 0
    second_half_mean = np.mean(y_actual[mid:]) if mid > 0 else 0
    diff = second_half_mean - first_half_mean
    if abs(diff) < 0.01 * np.mean(np.abs(y_actual) + 1e-6):
        trend = "stable"
    elif diff > 0:
        trend = "increasing"
    else:
        trend = "decreasing"

    desc = (
        f"mean={np.mean(y_actual):.2f}, "
        f"std={np.std(y_actual):.2f}, "
        f"min={np.min(y_actual):.2f}, "
        f"max={np.max(y_actual):.2f}, "
        f"rmse={rmse:.2f}, "
        f"trend={trend}"
    )

    if ad_score <= threshold:
        desc += f", ANOMALY DETECTED (score={ad_score})"

    return desc[:1000]
```

**Step 2: Test inference module with synthetic data**

```python
# test_infer.py (temporary)
import numpy as np
from infer_anomaly import compute_ad_score, generate_ad_desc

config = {"anomaly": {"score_threshold": 50}}

# Normal case: predicted matches actual
actual = np.array([100, 102, 98, 101, 99, 100, 103, 97])
predicted = np.array([100, 101, 99, 100, 100, 101, 102, 98])
score = compute_ad_score(actual, predicted, config)
print(f"Normal score: {score}")
assert score > 50, f"Expected score > 50, got {score}"

# Anomaly case: predicted far from actual
predicted_bad = np.array([50, 50, 50, 50, 50, 50, 50, 50])
score_bad = compute_ad_score(actual, predicted_bad, config)
print(f"Anomaly score: {score_bad}")
assert score_bad <= 50, f"Expected score <= 50, got {score_bad}"

desc = generate_ad_desc(actual, predicted_bad, score_bad, config)
print(f"AD_DESC: {desc}")
assert "ANOMALY DETECTED" in desc
print("infer_anomaly OK")
```

Run: `cd YiUmGoV2 && python test_infer.py`
Expected: scores printed + `infer_anomaly OK`

**Step 3: Remove temp test and commit**

```bash
rm YiUmGoV2/test_infer.py
git add YiUmGoV2/infer_anomaly.py
git commit -m "feat: add inference module with AD_SCORE and AD_DESC computation"
```

---

### Task 6: Training script

Create `train_anomaly.py` that reads training data (from CSV or DB), trains a LightGBM model, and saves it to `models/anomaly/{DEV_ID}.txt`. Uses `_config.json` for all hyperparameters.

**Files:**
- Create: `YiUmGoV2/train_anomaly.py`
- Reference: `YiUmGO/Anomaly_260207/Power_Prediction_train.py`

**Step 1: Create `train_anomaly.py`**

The script accepts CLI arguments:
- `--dev_id` (required): Device ID to train for
- `--csv_path` (optional): CSV file path. If omitted, reads from PostgresDB.
- `--dev_map_path` (optional): Device mapping CSV path.

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_anomaly.py - LightGBM anomaly detection model training.
Reads from CSV or PostgresDB, trains model, saves to models/anomaly/{DEV_ID}.txt
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import plotly.graph_objects as go

import utility as Util
import data_preprocessing as DP

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config.json')
    with open(config_path) as f:
        return json.load(f)

def load_data_from_csv(csv_path, dev_id, tag_cd, start_date=None, end_date=None):
    """Load and filter sensor data from CSV file."""
    df = pd.read_csv(csv_path)
    df['colec_dt'] = pd.to_datetime(df['colec_dt']).dt.floor('min')
    if start_date:
        df = df[df['colec_dt'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['colec_dt'] <= pd.to_datetime(end_date)]
    df_filtered = df[(df['dev_id'] == dev_id) & (df['tag_cd'] == tag_cd)]
    return pd.DataFrame(
        data=df_filtered['colec_val'].values,
        index=df_filtered['colec_dt'].values,
        columns=['value']
    )

def load_data_from_db(config, dev_id, tag_cd):
    """Load training data from PostgresDB."""
    import db_connection as DB
    engine = DB.create_db_engine(config)
    # For training, read all available historical data (not just 4h)
    table = config['data']['collection_table']
    from sqlalchemy import text
    query = text(f"""
        SELECT "COLEC_DT", "COLEC_VAL"
        FROM "{table}"
        WHERE "DEV_ID" = :dev_id AND "TAG_CD" = :tag_cd
        ORDER BY "COLEC_DT"
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"dev_id": str(dev_id), "tag_cd": str(tag_cd)})
    df.columns = ['colec_dt', 'colec_val']
    df['colec_dt'] = pd.to_datetime(df['colec_dt'])
    return pd.DataFrame(data=df['colec_val'].values, index=df['colec_dt'].values, columns=['value'])

def train(df_raw, config, dev_id):
    """Train LightGBM model and return (model, rmse, X_test, y_test, y_pred)."""
    X_df, y_df, nan_counts_df, missing_ratio = DP.preprocess(df_raw, config)
    print(f"Features: {X_df.shape[1]}, Samples: {len(X_df)}, Missing ratio: {missing_ratio}")

    test_size = config['model']['test_size']
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=test_size, random_state=42, shuffle=False
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': config['model']['objective'],
        'metric': config['model']['metric'],
        'boosting_type': config['model']['boosting_type'],
        'learning_rate': config['model']['learning_rate'],
        'num_leaves': config['model']['num_leaves'],
        'verbose': config['model']['verbose']
    }

    model = lgb.train(
        params, train_data,
        valid_sets=[valid_data],
        num_boost_round=config['model']['num_boost_round'],
        valid_names=['validation'],
        callbacks=[lgb.early_stopping(stopping_rounds=config['model']['early_stopping_rounds'])]
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}")

    return model, rmse, X_test, y_test, y_pred

def save_model(model, config, dev_id):
    """Save trained model to models/anomaly/{dev_id}.txt"""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['anomaly']['model_dir'])
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{dev_id}.txt")
    model.save_model(model_path)
    print(f"Model saved: {model_path}")
    return model_path

def main():
    parser = argparse.ArgumentParser(description='LightGBM anomaly detection - training')
    parser.add_argument('--dev_id', type=int, required=True, help='Device ID (e.g., 2001)')
    parser.add_argument('--csv_path', type=str, default=None, help='CSV data file path (if omitted, reads from DB)')
    parser.add_argument('--dev_map_path', type=str, default=None, help='Device mapping CSV path')
    parser.add_argument('--start_date', type=str, default=None, help='Training data start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='Training data end date (YYYY-MM-DD)')
    args = parser.parse_args()

    config = load_config()
    tag_cd = config['data']['tag_cd']

    if args.csv_path:
        print(f"Loading data from CSV: {args.csv_path}")
        df_raw = load_data_from_csv(args.csv_path, args.dev_id, tag_cd, args.start_date, args.end_date)
    else:
        print("Loading data from PostgresDB...")
        df_raw = load_data_from_db(config, args.dev_id, tag_cd)

    print(f"Data loaded: {len(df_raw)} rows, {df_raw.index[0]} ~ {df_raw.index[-1]}")

    model, rmse, X_test, y_test, y_pred = train(df_raw, config, args.dev_id)
    save_model(model, config, args.dev_id)

if __name__ == '__main__':
    main()
```

**Step 2: Test training with existing CSV data**

Run: `cd YiUmGoV2 && python train_anomaly.py --dev_id 2001 --csv_path ../YiUmGO/data_colec_h_202509091411_B0019.csv --start_date 2025-03-24 --end_date 2025-09-09`

Expected:
- Prints feature count, sample count, RMSE
- Creates `models/anomaly/2001.txt`

**Step 3: Verify model file was created**

Run: `ls -la YiUmGoV2/models/anomaly/2001.txt`
Expected: file exists with non-zero size

**Step 4: Commit**

```bash
git add YiUmGoV2/train_anomaly.py
git commit -m "feat: add training script with CSV/DB data source support"
```

---

### Task 7: Main runner (ai_anomaly_runner.py)

Create the cron entry point that orchestrates the full pipeline: read config -> get enabled devices -> fetch data -> infer -> write results.

**Files:**
- Create: `YiUmGoV2/ai_anomaly_runner.py`

**Step 1: Create `ai_anomaly_runner.py`**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ai_anomaly_runner.py - Main entry point for AI anomaly detection.
Designed to run hourly via cron.

Usage:
    python ai_anomaly_runner.py              # Production: reads from PostgresDB
    python ai_anomaly_runner.py --dry-run    # Dry run: no DB writes
"""
import os, sys, json, time, argparse, logging
import numpy as np
import pandas as pd

import db_connection as DB
import data_preprocessing as DP
import infer_anomaly as Infer

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config.json')
    with open(config_path) as f:
        return json.load(f)

def process_device(engine, config, bldg_id, dev_id, dry_run=False):
    """Run anomaly detection for a single device. Returns (ad_score, ad_desc) or None on error."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['anomaly']['model_dir'])
    model_path = os.path.join(model_dir, f"{dev_id}.txt")

    # 1. Load model
    try:
        model = Infer.load_model(model_path)
    except FileNotFoundError:
        logger.warning(f"No model for dev_id={dev_id}, skipping")
        return None

    # 2. Read sensor data (last 4h)
    df_sensor = DB.read_sensor_data(engine, config, bldg_id, dev_id)
    if df_sensor.empty or len(df_sensor) < 2:
        logger.warning(f"Insufficient data for dev_id={dev_id} ({len(df_sensor)} rows), skipping")
        return None

    # 3. Preprocess into features
    df_raw = pd.DataFrame(
        data=df_sensor['colec_val'].values,
        index=pd.to_datetime(df_sensor['colec_dt']),
        columns=['value']
    )
    try:
        X_df, y_df, _, _ = DP.preprocess(df_raw, config, fill_method='ffill')
    except Exception as e:
        logger.error(f"Preprocessing failed for dev_id={dev_id}: {e}")
        return None

    if len(X_df) == 0:
        logger.warning(f"No samples after preprocessing for dev_id={dev_id}, skipping")
        return None

    # 4. Run inference
    y_pred = Infer.run_inference(model, X_df)
    y_actual = y_df.values

    # 5. Compute score and description
    ad_score = Infer.compute_ad_score(y_actual, y_pred, config)
    ad_desc = Infer.generate_ad_desc(y_actual, y_pred, ad_score, config)

    # 6. Write result to DB
    if not dry_run:
        DB.write_anomaly_result(engine, config, bldg_id, dev_id, ad_score, ad_desc)
        logger.info(f"dev_id={dev_id}: AD_SCORE={ad_score}, written to DB")
    else:
        logger.info(f"dev_id={dev_id}: AD_SCORE={ad_score} (dry-run, not written)")

    return ad_score, ad_desc

def main():
    parser = argparse.ArgumentParser(description='AI BEMS Anomaly Detection Runner')
    parser.add_argument('--dry-run', action='store_true', help='Run without writing to DB')
    args = parser.parse_args()

    config = load_config()
    logger.info("=== AI Anomaly Detection Start ===")
    start_time = time.time()

    engine = DB.create_db_engine(config)

    # Get enabled devices
    devices = DB.read_enabled_devices(engine, config)
    logger.info(f"Found {len(devices)} enabled device(s)")

    results = []
    for device in devices:
        bldg_id = device['bldg_id']
        dev_id = device['dev_id']
        logger.info(f"Processing: bldg_id={bldg_id}, dev_id={dev_id}")
        result = process_device(engine, config, bldg_id, dev_id, dry_run=args.dry_run)
        results.append({'bldg_id': bldg_id, 'dev_id': dev_id, 'result': result})

    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r['result'] is not None)
    logger.info(f"=== Done: {success_count}/{len(results)} devices processed in {elapsed:.1f}s ===")

if __name__ == '__main__':
    main()
```

**Step 2: Verify module imports**

Run: `cd YiUmGoV2 && python -c "import ai_anomaly_runner; print('runner module OK')"`
Expected: `runner module OK`

**Step 3: Commit**

```bash
git add YiUmGoV2/ai_anomaly_runner.py
git commit -m "feat: add main anomaly runner with dry-run support"
```

---

### Task 8: End-to-end integration test

Test the full pipeline using existing CSV data and a trained model (from Task 6). No live DB needed — we mock the DB calls.

**Files:**
- Create: `YiUmGoV2/test_integration.py` (kept for future use)

**Step 1: Create integration test**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_integration.py - End-to-end test of the anomaly detection pipeline.
Uses CSV data and trained model. Does NOT require a live PostgresDB.
"""
import os, json
import numpy as np
import pandas as pd
import data_preprocessing as DP
import infer_anomaly as Infer
import utility as Util

def test_full_pipeline():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_config.json')
    with open(config_path) as f:
        config = json.load(f)

    dev_id = 2001
    tag_cd = config['data']['tag_cd']

    # 1. Load CSV data (simulating DB read)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'YiUmGO', 'data_colec_h_202509091411_B0019.csv')
    if not os.path.exists(csv_path):
        print(f"SKIP: CSV file not found at {csv_path}")
        return

    df_all = pd.read_csv(csv_path)
    df_all['colec_dt'] = pd.to_datetime(df_all['colec_dt']).dt.floor('min')
    df_sensor = df_all[(df_all['dev_id'] == dev_id) & (df_all['tag_cd'] == tag_cd)]

    # Take last 4 hours of data
    sampling_min = config['data']['sampling_minutes']
    input_hours = config['data']['input_interval_hours']
    n_samples = (input_hours * 60) // sampling_min
    df_sensor = df_sensor.tail(n_samples * 4)  # extra data for feature engineering lags
    print(f"Sensor data: {len(df_sensor)} rows")

    # 2. Preprocess
    df_raw = pd.DataFrame(
        data=df_sensor['colec_val'].values,
        index=pd.to_datetime(df_sensor['colec_dt'].values),
        columns=['value']
    )
    X_df, y_df, _, missing_ratio = DP.preprocess(df_raw, config, fill_method='ffill')
    print(f"After preprocessing: {len(X_df)} samples, {X_df.shape[1]} features, missing={missing_ratio}")

    # 3. Load model and infer
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              config['anomaly']['model_dir'], f"{dev_id}.txt")
    if not os.path.exists(model_path):
        print(f"SKIP: Model file not found at {model_path}. Run train_anomaly.py first.")
        return

    model = Infer.load_model(model_path)
    y_pred = Infer.run_inference(model, X_df)
    print(f"Predictions: {len(y_pred)} values")

    # 4. Compute score
    ad_score = Infer.compute_ad_score(y_df.values, y_pred, config)
    ad_desc = Infer.generate_ad_desc(y_df.values, y_pred, ad_score, config)

    print(f"AD_SCORE: {ad_score}")
    print(f"AD_DESC: {ad_desc}")

    assert 0 <= ad_score <= 100, f"Score out of range: {ad_score}"
    assert len(ad_desc) <= 1000, f"DESC too long: {len(ad_desc)}"
    print("\n=== INTEGRATION TEST PASSED ===")

if __name__ == '__main__':
    test_full_pipeline()
```

**Step 2: Run integration test**

Run: `cd YiUmGoV2 && python test_integration.py`

Expected:
- Prints sensor data count, preprocessing results, prediction count
- Prints AD_SCORE and AD_DESC
- Prints `=== INTEGRATION TEST PASSED ===`

Note: Requires Task 6 to have been completed (model file must exist at `models/anomaly/2001.txt`).

**Step 3: Commit**

```bash
git add YiUmGoV2/test_integration.py
git commit -m "test: add end-to-end integration test for anomaly pipeline"
```

---

### Task 9: Final cleanup and documentation

**Files:**
- Create: `YiUmGoV2/README.md`

**Step 1: Create README.md**

```markdown
# YiUmGoV2 - AI BEMS Anomaly Detection

LightGBM-based anomaly detection for Building Energy Management System.
Reads sensor data from PostgresDB, runs inference, writes results back.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `_config.json` with your DB credentials and model parameters.

## Usage

### Training (manual)

```bash
# From CSV file:
python train_anomaly.py --dev_id 2001 --csv_path /path/to/data.csv

# From PostgresDB:
python train_anomaly.py --dev_id 2001
```

### Inference (cron, hourly)

```bash
python ai_anomaly_runner.py           # Production
python ai_anomaly_runner.py --dry-run # Test without DB writes
```

### Cron setup

```bash
0 * * * * cd /path/to/YiUmGoV2 && python ai_anomaly_runner.py >> /var/log/ai_bems.log 2>&1
```

### Integration test

```bash
python test_integration.py
```
```

**Step 2: Commit all**

```bash
git add YiUmGoV2/README.md
git commit -m "docs: add README for YiUmGoV2 anomaly detection module"
```

---

## Summary of Tasks

| # | Task | Files | Est. |
|---|------|-------|------|
| 1 | Project scaffold + config | `_config.json`, `requirements.txt` | Quick |
| 2 | Utility module | `utility.py` | Quick |
| 3 | Data preprocessing module | `data_preprocessing.py` | Medium |
| 4 | DB connection module | `db_connection.py` | Medium |
| 5 | Inference module | `infer_anomaly.py` | Medium |
| 6 | Training script | `train_anomaly.py` | Medium |
| 7 | Main runner | `ai_anomaly_runner.py` | Medium |
| 8 | Integration test | `test_integration.py` | Medium |
| 9 | README and cleanup | `README.md` | Quick |

## Dependencies Between Tasks

```
Task 1 (scaffold) ──> Task 2 (utility) ──> Task 3 (preprocessing)
                                                    │
Task 4 (db_connection) ─────────────────────────────┤
                                                    │
Task 5 (inference) ─────────────────────────────────┤
                                                    │
                                               Task 6 (training)
                                                    │
                                               Task 7 (runner)
                                                    │
                                               Task 8 (integration test)
                                                    │
                                               Task 9 (README)
```

Tasks 2, 4, 5 can be done in parallel after Task 1. Task 3 depends on Task 2. Tasks 6-9 are sequential.
