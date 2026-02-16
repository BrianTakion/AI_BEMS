#!/usr/bin/env python
"""
run_e2e_test.py
End-to-end test for the anomaly detection pipeline using CSV files only (no PostgreSQL).

Runs 4 steps:
  1. Train a model for DEV_ID=2002
  2. Verify device is registered in enabled_devices.csv
  3. Run the anomaly detection service (simulating cron)
  4. Validate output in anomaly_results.csv

Usage:
    python run_e2e_test.py
"""

import json
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEV_ID = 2002
BLDG_ID = "B0019"


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


def step1_train():
    """Step 1: Train model for DEV_ID."""
    print(f"\n{'#' * 60}")
    print(f"  [STEP 1/4] Training model for DEV_ID={DEV_ID}")
    print(f"{'#' * 60}")

    model_path = resolve_path(f"models/anomaly/{DEV_ID}.txt")

    rc = run_command(
        [sys.executable, "train_anomaly.py", "--dev_id", str(DEV_ID)],
        f"Training LightGBM model for DEV_ID={DEV_ID}",
    )
    if rc != 0:
        return False

    if not os.path.isfile(model_path):
        print(f"[FAIL] Model file not created: {model_path}")
        return False

    size_kb = os.path.getsize(model_path) / 1024
    print(f"\n[OK] Model saved: {model_path} ({size_kb:.1f} KB)")
    return True


def step2_verify_device(config):
    """Step 2: Verify device is registered in enabled_devices.csv."""
    print(f"\n{'#' * 60}")
    print(f"  [STEP 2/4] Verifying device registration")
    print(f"{'#' * 60}")

    csv_path = resolve_path(config["csv"]["enabled_devices_path"])

    if not os.path.isfile(csv_path):
        print(f"[FAIL] Enabled devices CSV not found: {csv_path}")
        return False

    df = pd.read_csv(csv_path, dtype={"BLDG_ID": str, "DEV_ID": int, "FALT_PRCV_YN": str})
    enabled = df[(df["DEV_ID"] == DEV_ID) & (df["FALT_PRCV_YN"] == "Y")]

    if enabled.empty:
        print(f"[FAIL] DEV_ID={DEV_ID} not found with FALT_PRCV_YN='Y' in {csv_path}")
        return False

    print(f"\n[OK] DEV_ID={DEV_ID} is registered and enabled in {csv_path}")
    print(f"     BLDG_ID={enabled.iloc[0]['BLDG_ID']}")
    return True


def step3_run_inference():
    """Step 3: Run the anomaly detection service."""
    print(f"\n{'#' * 60}")
    print(f"  [STEP 3/4] Running anomaly detection service")
    print(f"{'#' * 60}")

    rc = run_command(
        [sys.executable, "ai_anomaly_runner.py"],
        "Running ai_anomaly_runner.py (simulating cron)",
    )
    return rc == 0


def step4_validate_output(config):
    """Step 4: Validate the output CSV file."""
    print(f"\n{'#' * 60}")
    print(f"  [STEP 4/4] Validating output")
    print(f"{'#' * 60}")

    result_path = resolve_path(config["csv"]["result_path"])

    if not os.path.isfile(result_path):
        print(f"[FAIL] Result CSV not found: {result_path}")
        return False

    df = pd.read_csv(result_path, dtype={"DEV_ID": str})
    dev_rows = df[df["DEV_ID"] == str(DEV_ID)]

    if dev_rows.empty:
        print(f"[FAIL] No results for DEV_ID={DEV_ID} in {result_path}")
        return False

    # Validate the most recent row
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
            print(f"[FAIL] {e}")
        return False

    print(f"\n[OK] Result validated for DEV_ID={DEV_ID}:")
    print(f"     USE_DT    = {row['USE_DT']}")
    print(f"     BLDG_ID   = {row['BLDG_ID']}")
    print(f"     DEV_ID    = {row['DEV_ID']}")
    print(f"     AD_SCORE  = {row['AD_SCORE']}")
    print(f"     AD_DESC   = {row['AD_DESC'][:80]}...")
    return True


def main():
    print(f"\n{'=' * 60}")
    print(f"  E2E Anomaly Detection Test")
    print(f"  DEV_ID={DEV_ID}, BLDG_ID={BLDG_ID}")
    print(f"  Mode: CSV (no PostgreSQL)")
    print(f"{'=' * 60}")

    config = load_config()
    results = {}

    # Step 1: Train
    results["train"] = step1_train()
    if not results["train"]:
        print("\n[ABORT] Training failed. Cannot continue.")
        sys.exit(1)

    # Step 2: Verify device registration
    results["register"] = step2_verify_device(config)
    if not results["register"]:
        print("\n[ABORT] Device not registered. Cannot continue.")
        sys.exit(1)

    # Step 3: Run inference
    results["inference"] = step3_run_inference()
    if not results["inference"]:
        print("\n[ABORT] Inference failed. Cannot continue.")
        sys.exit(1)

    # Step 4: Validate output
    results["validate"] = step4_validate_output(config)

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
