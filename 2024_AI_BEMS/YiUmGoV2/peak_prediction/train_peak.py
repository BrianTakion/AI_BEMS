#!/usr/bin/env python
"""
train_peak.py  --  Train LightGBM models for daily peak prediction.

Trains two models per device:
  - {dev_id}_power.txt: predicts daily peak power (kW)
  - {dev_id}_time.txt: predicts daily peak time slot (0-95)

Reads all historical sensor data, builds daily feature vectors via
data_preprocessing.preprocess_for_training(), then trains LightGBM
regression models.

Usage:
    python peak_prediction/train_peak.py --csv
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import data_preprocessing as DP
import data_source as DS


def train_model(X_train, y_train, X_test, y_test, model_cfg, model_path, target_name):
    """Train a single LightGBM model and save it.

    Returns:
        Tuple (model, metrics_dict).
    """
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

    print(f"  [TRAIN-{target_name}] LightGBM params: {params}")

    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        valid_names=["validation"],
        num_boost_round=model_cfg["num_boost_round"],
        callbacks=[lgb.early_stopping(stopping_rounds=model_cfg["early_stopping_rounds"])],
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    model.save_model(model_path)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "best_iteration": model.best_iteration,
        "train_time": elapsed,
    }

    print(f"  [TRAIN-{target_name}] Done in {elapsed:.1f}s | "
          f"RMSE={rmse:.4f} MAE={mae:.4f} | "
          f"best_iter={model.best_iteration} | saved: {model_path}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM peak prediction models")
    parser.add_argument("--csv", action="store_true", required=True,
                        help="Train using CSV data source")
    args = parser.parse_args()

    # 1. Load configuration
    config_path = os.path.join(SCRIPT_DIR, "_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["data_source"] = "csv"

    model_cfg = config["model"]
    model_dir = os.path.join(SCRIPT_DIR, config["peak"]["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    # 2. Read enabled devices
    source = DS.create_data_source(config)
    devices = DS.read_enabled_devices(source, config)
    print(f"[TRAIN] Enabled devices ({len(devices)}): {[d['dev_id'] for d in devices]}")

    # 3. Device loop
    for device in devices:
        bldg_id = device["bldg_id"]
        dev_id = device["dev_id"]
        power_model_path = os.path.join(model_dir, f"{dev_id}_power.txt")
        time_model_path = os.path.join(model_dir, f"{dev_id}_time.txt")

        # Skip if both models already exist
        if os.path.isfile(power_model_path) and os.path.isfile(time_model_path):
            print(f"[SKIP] Models already exist: {power_model_path}, {time_model_path}")
            continue

        print("=" * 60)
        print(f"[TRAIN] Device: bldg_id={bldg_id}, dev_id={dev_id}")
        print("=" * 60)

        # 4. Load ALL historical sensor data
        t0 = time.time()
        df_sensor = DS.read_sensor_data(source, config, bldg_id, dev_id, fetch_hours=0)
        if df_sensor.empty:
            print(f"[ERROR] No data found for dev_id={dev_id}")
            continue
        print(f"[TRAIN] Data loaded in {time.time() - t0:.1f}s ({len(df_sensor)} rows)")

        # 5. Build daily training dataset
        t0 = time.time()
        try:
            X_df, y_power, y_slot = DP.preprocess_for_training(df_sensor, config)
        except ValueError as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            continue
        print(f"[TRAIN] Features built in {time.time() - t0:.1f}s: "
              f"{len(X_df)} days, {X_df.shape[1]} features")

        # 6. Train/test split
        test_size = model_cfg["test_size"]
        X_train, X_test, y_pow_train, y_pow_test, y_slot_train, y_slot_test = (
            train_test_split(
                X_df, y_power, y_slot,
                test_size=test_size, random_state=42, shuffle=True,
            )
        )
        print(f"[TRAIN] Train: {len(X_train)} days, Test: {len(X_test)} days")

        # 7. Train peak power model
        print(f"\n--- Training POWER model ---")
        power_model, power_metrics = train_model(
            X_train, y_pow_train, X_test, y_pow_test,
            model_cfg, power_model_path, "POWER"
        )

        # 8. Train peak time model
        print(f"\n--- Training TIME model ---")
        time_model, time_metrics = train_model(
            X_train, y_slot_train, X_test, y_slot_test,
            model_cfg, time_model_path, "TIME"
        )

        # 9. Summary
        # Convert test slot predictions to hours for interpretability
        y_slot_pred = time_model.predict(X_test, num_iteration=time_model.best_iteration)
        slot_errors = np.abs(y_slot_test.values - np.round(y_slot_pred))
        time_error_minutes = slot_errors.mean() * 15  # Each slot = 15 min

        print()
        print("=" * 60)
        print("  Training Summary")
        print("=" * 60)
        print(f"  Device              : {bldg_id}/{dev_id}")
        print(f"  Training days       : {len(X_train)}")
        print(f"  Test days           : {len(X_test)}")
        print(f"  Features            : {X_df.shape[1]}")
        print(f"  --- Power Model ---")
        print(f"  RMSE                : {power_metrics['rmse']:.4f} kW")
        print(f"  MAE                 : {power_metrics['mae']:.4f} kW")
        print(f"  Best iteration      : {power_metrics['best_iteration']}")
        print(f"  --- Time Model ---")
        print(f"  RMSE (slots)        : {time_metrics['rmse']:.4f}")
        print(f"  MAE (slots)         : {time_metrics['mae']:.4f}")
        print(f"  Avg time error      : {time_error_minutes:.0f} minutes")
        print(f"  Best iteration      : {time_metrics['best_iteration']}")
        print(f"  --- Files ---")
        print(f"  Power model         : {power_model_path}")
        print(f"  Time model          : {time_model_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
