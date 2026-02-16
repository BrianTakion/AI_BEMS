#!/usr/bin/env python
"""
run_e2e_test.py
End-to-end test for the anomaly detection pipeline using CSV files only (no PostgreSQL).

Runs 3 steps:
  1. Read enabled_devices.csv, train models for devices missing a model file
  2. Run the anomaly detection service (simulating cron) for all enabled devices
  3. Validate output in anomaly_results.csv for every enabled device

Usage:
    python run_e2e_test.py
"""

import json
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    config_path = os.path.join(SCRIPT_DIR, "_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(rel_path):
    return os.path.normpath(os.path.join(SCRIPT_DIR, rel_path))


def run_command(cmd, description):
    """Run a shell command, printing output in real-time. Returns exit code."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n[FAIL] Command exited with code {result.returncode}")
    return result.returncode


def read_enabled_devices(config):
    """Read enabled_devices.csv and return list of (BLDG_ID, DEV_ID) with FALT_PRCV_YN='Y'."""
    csv_path = resolve_path(config["csv"]["enabled_devices_path"])
    if not os.path.isfile(csv_path):
        print(f"[FAIL] Enabled devices CSV not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path, dtype={"BLDG_ID": str, "DEV_ID": int, "FALT_PRCV_YN": str})
    enabled = df[df["FALT_PRCV_YN"] == "Y"]

    if enabled.empty:
        print(f"[FAIL] No devices with FALT_PRCV_YN='Y' in {csv_path}")
        return []

    devices = list(zip(enabled["BLDG_ID"], enabled["DEV_ID"]))
    print(f"[OK] Found {len(devices)} enabled device(s): {[d for _, d in devices]}")
    return devices


def step1_train(config, devices):
    """Step 1: Train models for devices that don't have one yet."""
    print(f"\n{'#' * 60}")
    print(f"  [STEP 1/3] Training models (skip if model exists)")
    print(f"{'#' * 60}")

    model_dir = config["anomaly"]["model_dir"]
    all_ok = True

    for bldg_id, dev_id in devices:
        model_path = resolve_path(f"{model_dir}/{dev_id}.txt")

        if os.path.isfile(model_path):
            size_kb = os.path.getsize(model_path) / 1024
            print(f"\n[SKIP] Model already exists for DEV_ID={dev_id}: {model_path} ({size_kb:.1f} KB)")
            continue

        rc = run_command(
            [sys.executable, "train_anomaly.py", "--dev_id", str(dev_id)],
            f"Training LightGBM model for DEV_ID={dev_id}",
        )
        if rc != 0:
            print(f"[FAIL] Training failed for DEV_ID={dev_id}")
            all_ok = False
            continue

        if not os.path.isfile(model_path):
            print(f"[FAIL] Model file not created: {model_path}")
            all_ok = False
            continue

        size_kb = os.path.getsize(model_path) / 1024
        print(f"\n[OK] Model saved: {model_path} ({size_kb:.1f} KB)")

    return all_ok


def step2_run_inference():
    """Step 2: Run the anomaly detection service for all enabled devices."""
    print(f"\n{'#' * 60}")
    print(f"  [STEP 2/3] Running anomaly detection service")
    print(f"{'#' * 60}")

    rc = run_command(
        [sys.executable, "ai_anomaly_runner.py"],
        "Running ai_anomaly_runner.py (simulating cron)",
    )
    return rc == 0


def step3_validate_output(config, devices):
    """Step 3: Validate the output CSV file for every enabled device."""
    print(f"\n{'#' * 60}")
    print(f"  [STEP 3/3] Validating output for all devices")
    print(f"{'#' * 60}")

    result_path = resolve_path(config["csv"]["result_path"])

    if not os.path.isfile(result_path):
        print(f"[FAIL] Result CSV not found: {result_path}")
        return False

    df = pd.read_csv(result_path, dtype={"DEV_ID": str})
    all_ok = True

    for bldg_id, dev_id in devices:
        dev_rows = df[df["DEV_ID"] == str(dev_id)]

        if dev_rows.empty:
            print(f"\n[FAIL] No results for DEV_ID={dev_id} in {result_path}")
            all_ok = False
            continue

        row = dev_rows.iloc[-1]
        errors = []

        ad_score = row["AD_SCORE"]
        if not (0 <= ad_score <= 100):
            errors.append(f"AD_SCORE={ad_score} out of range [0, 100]")

        ad_desc = str(row["AD_DESC"])
        if not ad_desc or ad_desc == "nan":
            errors.append("AD_DESC is empty")
        if len(ad_desc) > 1000:
            errors.append(f"AD_DESC length={len(ad_desc)} exceeds 1000")

        use_dt = str(row["USE_DT"])
        if not use_dt or use_dt == "nan":
            errors.append("USE_DT is empty")

        if errors:
            for e in errors:
                print(f"[FAIL] DEV_ID={dev_id}: {e}")
            all_ok = False
            continue

        print(f"\n[OK] Result validated for DEV_ID={dev_id}:")
        print(f"     USE_DT    = {row['USE_DT']}")
        print(f"     BLDG_ID   = {row['BLDG_ID']}")
        print(f"     DEV_ID    = {row['DEV_ID']}")
        print(f"     AD_SCORE  = {row['AD_SCORE']}")
        print(f"     AD_DESC   = {str(row['AD_DESC'])[:80]}...")

    return all_ok


def main():
    config = load_config()

    devices = read_enabled_devices(config)
    if not devices:
        print("\n[ABORT] No enabled devices found.")
        sys.exit(1)

    dev_ids = [d for _, d in devices]
    print(f"\n{'=' * 60}")
    print(f"  E2E Anomaly Detection Test")
    print(f"  Devices: {dev_ids}")
    print(f"  Mode: CSV (no PostgreSQL)")
    print(f"{'=' * 60}")

    results = {}

    # Step 1: Train (skip if model exists)
    results["train"] = step1_train(config, devices)
    if not results["train"]:
        print("\n[ABORT] Training failed for one or more devices. Cannot continue.")
        sys.exit(1)

    # Step 2: Run inference
    results["inference"] = step2_run_inference()
    if not results["inference"]:
        print("\n[ABORT] Inference failed. Cannot continue.")
        sys.exit(1)

    # Step 3: Validate output
    results["validate"] = step3_validate_output(config, devices)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  E2E Test Summary")
    print(f"{'=' * 60}")
    all_pass = all(results.values())
    for step_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {step_name}")
    print(f"{'=' * 60}")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
