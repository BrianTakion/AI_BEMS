#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_anomaly.py
LightGBM-based anomaly detection model -- training script.

Reads sensor data from CSV (dev mode) or PostgreSQL (production mode),
performs feature engineering via data_preprocessing.preprocess(), trains
a LightGBM regression model, and saves it to models/anomaly/{dev_id}.txt.

Usage examples:
    # Train with explicit CSV path and date range
    python train_anomaly.py --dev_id 2001 \
        --csv_path ../YiUmGO/data_colec_h_202509091411_B0019.csv \
        --start_date 2025-03-24 --end_date 2025-09-09

    # Train using data_source and paths from _config.json
    python train_anomaly.py --dev_id 2001
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# ---------------------------------------------------------------------------
# Path setup -- ensure YiUmGoV2/ is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import data_preprocessing as DP


# ===================================================================
# CSV helpers
# ===================================================================

def _load_csv_chunked(csv_path, dev_id, tag_cd, start_date=None, end_date=None):
    """Read a large CSV in chunks, filtering by dev_id, tag_cd, and optional date range.

    Returns a DataFrame with columns ['colec_dt', 'colec_val'], sorted by colec_dt.
    """
    CHUNK_SIZE = 500_000
    usecols = ["dev_id", "tag_cd", "colec_dt", "colec_val"]
    chunks = []

    print(f"[DATA] Reading CSV: {csv_path}")
    print(f"[DATA] Filtering dev_id={dev_id}, tag_cd={tag_cd}")

    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype={"dev_id": str, "tag_cd": str, "colec_val": float},
        parse_dates=["colec_dt"],
        chunksize=CHUNK_SIZE,
    ):
        mask = (chunk["dev_id"] == str(dev_id)) & (chunk["tag_cd"] == str(tag_cd))
        filtered = chunk.loc[mask]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        raise ValueError(
            f"No data found in CSV for dev_id={dev_id}, tag_cd={tag_cd}"
        )

    df = pd.concat(chunks, ignore_index=True)
    df["colec_dt"] = pd.to_datetime(df["colec_dt"]).dt.floor("min")
    df = df.sort_values("colec_dt").reset_index(drop=True)

    # Apply date filters
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df = df[df["colec_dt"] >= start_dt]
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        df = df[df["colec_dt"] <= end_dt]

    df = df[["colec_dt", "colec_val"]].reset_index(drop=True)
    print(f"[DATA] Loaded {len(df)} rows  "
          f"({df['colec_dt'].iloc[0]} ~ {df['colec_dt'].iloc[-1]})")
    return df


def _load_db_all(config, dev_id, tag_cd, start_date=None, end_date=None):
    """Read ALL historical sensor data from PostgreSQL for training.

    Unlike db_connection.read_sensor_data() which only reads the last N hours,
    this function fetches all available data (with optional date filters).
    """
    from sqlalchemy import create_engine

    db_cfg = config["db"]
    url = (
        f"postgresql://{db_cfg['user']}:{db_cfg['password']}"
        f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
    )
    engine = create_engine(url)
    table = config["data"]["collection_table"]  # DATA_COLEC_H

    where_clauses = [
        '"DEV_ID" = %(dev_id)s',
        '"TAG_CD" = %(tag_cd)s',
    ]
    params = {"dev_id": str(dev_id), "tag_cd": str(tag_cd)}

    if start_date is not None:
        where_clauses.append('"COLEC_DT" >= %(start_dt)s')
        params["start_dt"] = pd.to_datetime(start_date)
    if end_date is not None:
        where_clauses.append('"COLEC_DT" <= %(end_dt)s')
        params["end_dt"] = pd.to_datetime(end_date)

    where_sql = " AND ".join(where_clauses)
    query = (
        f'SELECT "COLEC_DT" AS colec_dt, "COLEC_VAL" AS colec_val '
        f'FROM "{table}" '
        f"WHERE {where_sql} "
        f'ORDER BY "COLEC_DT" ASC'
    )

    print(f"[DATA] Querying DB table {table} for dev_id={dev_id}, tag_cd={tag_cd} ...")
    df = pd.read_sql(query, engine, params=params)
    df["colec_dt"] = pd.to_datetime(df["colec_dt"])
    df["colec_val"] = df["colec_val"].astype(float)
    print(f"[DATA] Loaded {len(df)} rows from DB")

    if df.empty:
        raise ValueError(
            f"No data found in DB for dev_id={dev_id}, tag_cd={tag_cd}"
        )
    return df


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM anomaly detection model for a single device"
    )
    parser.add_argument(
        "--dev_id", type=int, required=True,
        help="Device ID to train for (e.g. 2001)"
    )
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="CSV data file path. Overrides data_source setting in config."
    )
    parser.add_argument(
        "--start_date", type=str, default=None,
        help="Training data start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date", type=str, default=None,
        help="Training data end date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    config_path = os.path.join(SCRIPT_DIR, "_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    tag_cd = config["data"]["tag_cd"]  # 30001
    model_cfg = config["model"]

    print("=" * 60)
    print(f"[TRAIN] Device ID : {args.dev_id}")
    print(f"[TRAIN] Tag CD    : {tag_cd}")
    print(f"[TRAIN] Date range: {args.start_date or '(all)'} ~ {args.end_date or '(all)'}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    t0 = time.time()

    if args.csv_path is not None:
        # Explicit CSV path from CLI
        csv_abs = os.path.normpath(os.path.join(SCRIPT_DIR, args.csv_path))
        if not os.path.isfile(csv_abs):
            # Try as-is (absolute path)
            csv_abs = args.csv_path
        df_sensor = _load_csv_chunked(
            csv_abs, args.dev_id, tag_cd, args.start_date, args.end_date
        )
    elif config.get("data_source", "csv") == "csv":
        # CSV mode from config
        csv_rel = config["csv"]["data_path"]
        csv_abs = os.path.normpath(os.path.join(SCRIPT_DIR, csv_rel))
        df_sensor = _load_csv_chunked(
            csv_abs, args.dev_id, tag_cd, args.start_date, args.end_date
        )
    else:
        # DB mode -- fetch ALL historical data
        df_sensor = _load_db_all(
            config, args.dev_id, tag_cd, args.start_date, args.end_date
        )

    load_time = time.time() - t0
    print(f"[TRAIN] Data loaded in {load_time:.1f}s  ({len(df_sensor)} rows)")

    # ------------------------------------------------------------------
    # 3. Build raw DataFrame:  index=colec_dt, columns=['value']
    # ------------------------------------------------------------------
    df_raw = pd.DataFrame(
        data=df_sensor["colec_val"].values,
        index=df_sensor["colec_dt"],
        columns=["value"],
    )
    print(f"[TRAIN] Raw data range: {df_raw.index[0]} ~ {df_raw.index[-1]}")

    # ------------------------------------------------------------------
    # 4. Feature engineering via data_preprocessing
    # ------------------------------------------------------------------
    X_df, y_df, nan_counts_df, missing_ratio = DP.preprocess(df_raw, config)
    print(f"[TRAIN] Features: {X_df.shape[1]},  Samples: {len(X_df)}")
    print(f"[TRAIN] NaN counts max: {nan_counts_df.max()},  Missing ratio: {missing_ratio}")

    if len(X_df) == 0:
        print("[ERROR] No samples after preprocessing. Check data quality / date range.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Train / test split (time-ordered, no shuffle)
    # ------------------------------------------------------------------
    test_size = model_cfg["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df,
        test_size=test_size,
        random_state=42,
        shuffle=False,
    )
    print(f"[TRAIN] Train size: {len(X_train)},  Test size: {len(X_test)}")

    # ------------------------------------------------------------------
    # 6. Train LightGBM
    # ------------------------------------------------------------------
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": model_cfg["objective"],
        "metric": model_cfg["metric"],
        "boosting_type": model_cfg["boosting_type"],
        "learning_rate": model_cfg["learning_rate"],
        "num_leaves": model_cfg["num_leaves"],
        "verbose": model_cfg["verbose"],
    }

    print(f"[TRAIN] LightGBM params: {params}")
    print(f"[TRAIN] num_boost_round={model_cfg['num_boost_round']}, "
          f"early_stopping_rounds={model_cfg['early_stopping_rounds']}")

    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        valid_names=["validation"],
        num_boost_round=model_cfg["num_boost_round"],
        callbacks=[lgb.early_stopping(stopping_rounds=model_cfg["early_stopping_rounds"])],
    )
    train_time = time.time() - t0

    # ------------------------------------------------------------------
    # 7. Evaluate on test set
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"[TRAIN] Training completed in {train_time:.1f}s")
    print(f"[TRAIN] Best iteration: {model.best_iteration}")
    print(f"[TRAIN] Test RMSE: {rmse:.4f}  ({X_df.shape[1]} features)")

    # ------------------------------------------------------------------
    # 8. Save model
    # ------------------------------------------------------------------
    model_dir = os.path.join(SCRIPT_DIR, config["anomaly"]["model_dir"])
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{args.dev_id}.txt")
    model.save_model(model_path)
    print(f"[TRAIN] Model saved: {model_path}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Device ID        : {args.dev_id}")
    print(f"  Tag CD           : {tag_cd}")
    print(f"  Data rows (raw)  : {len(df_sensor)}")
    print(f"  Samples (after FE): {len(X_df)}")
    print(f"  Features         : {X_df.shape[1]}")
    print(f"  Train / Test     : {len(X_train)} / {len(X_test)}")
    print(f"  Test RMSE        : {rmse:.4f}")
    print(f"  Best iteration   : {model.best_iteration}")
    print(f"  Model file       : {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
