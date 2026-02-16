#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_anomaly.py
LightGBM-based anomaly detection model -- training script.

Loads ALL historical sensor data from CSV, then samples random 176-hour
windows (controlled by training.max_steps in _config.json) to build a
diverse training set.  Each window is preprocessed via
data_preprocessing.preprocess() and the last `samples_per_window` rows
are collected.  A LightGBM regression model is trained on the pooled
samples and saved to models/anomaly/{dev_id}.txt.

Reads enabled devices from config_anomaly_devices.csv (FALT_PRCV_YN='Y').
Skips training when a model already exists.

Usage:
    python train_anomaly.py --csv
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

def _load_csv_chunked(csv_path, dev_id, tag_cd):
    """Read a large CSV in chunks, filtering by dev_id and tag_cd.

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

    df = df[["colec_dt", "colec_val"]].reset_index(drop=True)
    print(f"[DATA] Loaded {len(df)} rows  "
          f"({df['colec_dt'].iloc[0]} ~ {df['colec_dt'].iloc[-1]})")
    return df


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM anomaly detection models"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        required=True,
        help="Read enabled devices from config_anomaly_devices.csv and train each",
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
    fetch_hours = config["data"]["fetch_window_hours"]  # 176
    max_steps = config["training"]["max_steps"]
    samples_per_window = config["training"]["samples_per_window"]

    # Shared paths -- resolved once, before the device loop
    model_dir = os.path.join(SCRIPT_DIR, config["anomaly"]["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    csv_path = config["csv"]["data_path"]
    csv_abs = os.path.normpath(os.path.join(SCRIPT_DIR, csv_path))
    if not os.path.isfile(csv_abs):
        csv_abs = csv_path

    # ==================================================================
    # Read enabled devices from CSV
    # ==================================================================
    devices_path = os.path.normpath(
        os.path.join(SCRIPT_DIR, config["csv"]["config_anomaly_devices_path"])
    )
    devices_df = pd.read_csv(
        devices_path, dtype={"BLDG_ID": str, "DEV_ID": int, "FALT_PRCV_YN": str}
    )
    enabled = devices_df[devices_df["FALT_PRCV_YN"] == "Y"]
    dev_ids = enabled["DEV_ID"].tolist()

    print(f"[TRAIN] Enabled devices ({len(dev_ids)}): {dev_ids}")
    print(f"[TRAIN] Devices CSV: {devices_path}")

    # ==================================================================
    # Device loop
    # ==================================================================
    for dev_id in dev_ids:
        model_path = os.path.join(model_dir, f"{dev_id}.txt")

        # Skip if model already exists
        if os.path.isfile(model_path):
            print(f"[SKIP] Model already exists: {model_path}")
            continue

        print("=" * 60)
        print(f"[TRAIN] Device ID          : {dev_id}")
        print(f"[TRAIN] Tag CD             : {tag_cd}")
        print(f"[TRAIN] Window size        : {fetch_hours}h")
        print(f"[TRAIN] Max steps          : {max_steps}")
        print(f"[TRAIN] Samples per window : {samples_per_window}")
        print("=" * 60)

        # --------------------------------------------------------------
        # 2. Load ALL historical CSV data
        # --------------------------------------------------------------
        t0 = time.time()

        try:
            df_sensor = _load_csv_chunked(csv_abs, dev_id, tag_cd)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            continue

        load_time = time.time() - t0
        print(f"[TRAIN] Data loaded in {load_time:.1f}s  ({len(df_sensor)} rows)")

        # --------------------------------------------------------------
        # 3. Build full historical DataFrame:  index=colec_dt, columns=['value']
        # --------------------------------------------------------------
        df_all = pd.DataFrame(
            data=df_sensor["colec_val"].values,
            index=df_sensor["colec_dt"],
            columns=["value"],
        )
        print(f"[TRAIN] Raw data range: {df_all.index[0]} ~ {df_all.index[-1]}")

        # --------------------------------------------------------------
        # 4. Random window sampling + feature engineering
        # --------------------------------------------------------------
        window_td = pd.Timedelta(hours=fetch_hours)
        min_end = df_all.index[0] + window_td
        max_end = df_all.index[-1]

        if min_end > max_end:
            print(f"[ERROR] Not enough data for a single {fetch_hours}h window. "
                  f"Need data spanning at least {fetch_hours}h.")
            continue

        # Use nanosecond arithmetic to avoid timezone-sensitive timestamp() round-trip
        min_end_ns = min_end.value
        max_end_ns = max_end.value

        np.random.seed(42)

        X_list = []
        y_list = []
        windows_sampled = 0
        windows_skipped = 0

        print(f"[TRAIN] Sampling {max_steps} random {fetch_hours}h windows ...")

        for step in range(max_steps):
            # Pick a random end_time within the valid range (tz-free nanosecond arithmetic)
            rand_ns = np.random.uniform(min_end_ns, max_end_ns)
            end_time = pd.Timestamp(int(rand_ns))
            start_time = end_time - window_td

            # Slice the window from the full dataset
            window_df = df_all.loc[start_time:end_time].copy()

            # Skip windows with insufficient raw data
            if len(window_df) < 2:
                windows_skipped += 1
                continue

            # Preprocess the window
            try:
                X_df, y_df, nan_counts_df, missing_ratio = DP.preprocess(
                    window_df, config, fill_method="ffill"
                )
            except Exception as exc:
                windows_skipped += 1
                if windows_skipped == 1 or (step + 1) % 100 == 0:
                    print(f"  [WARN] Step {step + 1}: preprocess failed ({exc})")
                continue

            # Need at least samples_per_window rows after preprocessing
            if len(X_df) < samples_per_window:
                windows_skipped += 1
                continue

            # Take the last samples_per_window rows from this window
            X_list.append(X_df.iloc[-samples_per_window:])
            y_list.append(y_df.iloc[-samples_per_window:])
            windows_sampled += 1

            if (step + 1) % 100 == 0:
                print(f"  [PROGRESS] Step {step + 1}/{max_steps}  "
                      f"sampled={windows_sampled}  skipped={windows_skipped}")

        print(f"[TRAIN] Window sampling complete: "
              f"{windows_sampled} sampled, {windows_skipped} skipped")

        if windows_sampled == 0:
            print("[ERROR] No valid windows produced samples. "
                  "Check data quality.")
            continue

        # Concatenate all collected samples
        X_all = pd.concat(X_list, ignore_index=True)
        y_all = pd.concat(y_list, ignore_index=True)

        n_features = X_all.shape[1]
        n_samples = len(X_all)
        print(f"[TRAIN] Total samples: {n_samples}  Features: {n_features}")

        # --------------------------------------------------------------
        # 5. Train / test split (shuffle=True -- samples from random windows)
        # --------------------------------------------------------------
        test_size = model_cfg["test_size"]
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=test_size,
            random_state=42,
            shuffle=True,
        )
        print(f"[TRAIN] Train size: {len(X_train)},  Test size: {len(X_test)}")

        # --------------------------------------------------------------
        # 6. Train LightGBM
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # 7. Evaluate on test set
        # --------------------------------------------------------------
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        rmse = root_mean_squared_error(y_test, y_pred)

        print(f"[TRAIN] Training completed in {train_time:.1f}s")
        print(f"[TRAIN] Best iteration: {model.best_iteration}")
        print(f"[TRAIN] Test RMSE: {rmse:.4f}  ({n_features} features)")

        # --------------------------------------------------------------
        # 8. Save model
        # --------------------------------------------------------------
        model.save_model(model_path)
        print(f"[TRAIN] Model saved: {model_path}")

        # --------------------------------------------------------------
        # 9. Summary
        # --------------------------------------------------------------
        print()
        print("=" * 60)
        print("  Training Summary")
        print("=" * 60)
        print(f"  Device ID          : {dev_id}")
        print(f"  Tag CD             : {tag_cd}")
        print(f"  Data rows (raw)    : {len(df_sensor)}")
        print(f"  Window size        : {fetch_hours}h")
        print(f"  Windows sampled    : {windows_sampled}")
        print(f"  Windows skipped    : {windows_skipped}")
        print(f"  Samples per window : {samples_per_window}")
        print(f"  Total samples      : {n_samples}")
        print(f"  Features           : {n_features}")
        print(f"  Train / Test       : {len(X_train)} / {len(X_test)}")
        print(f"  Test RMSE          : {rmse:.4f}")
        print(f"  Best iteration     : {model.best_iteration}")
        print(f"  Model file         : {model_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
