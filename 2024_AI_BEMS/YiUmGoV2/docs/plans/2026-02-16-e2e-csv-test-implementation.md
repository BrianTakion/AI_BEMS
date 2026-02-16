# E2E CSV-Based Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable full E2E anomaly detection testing for DEV_ID=2002 using only CSV files (no PostgreSQL).

**Architecture:** Extend `db_connection.py` CSV mode to read device config from a CSV file and write results to a CSV file. Create a manual E2E script that trains a model, runs the inference pipeline, and validates output.

**Tech Stack:** Python 3, pandas, LightGBM, subprocess

---

### Task 1: Update `_config.json` with new CSV paths

**Files:**
- Modify: `_config.json:2-5`

**Step 1: Add enabled_devices_path and result_path to csv section**

Edit `_config.json` to change the `csv` section from:

```json
"csv": {
    "data_path": "/workspace/2024_AI_BEMS/YiUmGoV2/dataset/data_colec_h_250730_260212_B0019.csv"
}
```

to:

```json
"csv": {
    "data_path": "/workspace/2024_AI_BEMS/YiUmGoV2/dataset/data_colec_h_250730_260212_B0019.csv",
    "enabled_devices_path": "dataset/enabled_devices.csv",
    "result_path": "output/anomaly_results.csv"
}
```

Note: `enabled_devices_path` and `result_path` are relative to the script directory (resolved by `_resolve_path()` in `db_connection.py`). `data_path` stays absolute because `read_sensor_data()` already resolves it.

**Step 2: Commit**

```bash
git add _config.json
git commit -m "config: add enabled_devices_path and result_path to csv section"
```

---

### Task 2: Create `dataset/enabled_devices.csv`

**Files:**
- Create: `dataset/enabled_devices.csv`

**Step 1: Create the CSV file**

Create `dataset/enabled_devices.csv` with contents:

```csv
BLDG_ID,DEV_ID,FALT_PRCV_YN
B0019,2001,Y
B0019,2002,Y
```

This mimics the `DEV_USE_PURP_REL_R` database table. Only rows with `FALT_PRCV_YN=Y` will be processed.

**Step 2: Commit**

```bash
git add dataset/enabled_devices.csv
git commit -m "data: add enabled_devices.csv for CSV-mode device config"
```

---

### Task 3: Update `read_enabled_devices()` in `db_connection.py`

**Files:**
- Modify: `db_connection.py:100-103`
- Test: Run existing integration test to verify no regression

**Step 1: Replace the hardcoded device list with CSV reading**

In `db_connection.py`, replace the CSV mode block in `read_enabled_devices()` (lines 100-103):

```python
    # CSV mode -- hardcoded test devices
    devices = [{"bldg_id": "B0019", "dev_id": 2001}]
    logger.info("CSV mode: returning hardcoded device list (%d devices)", len(devices))
    return devices
```

with:

```python
    # CSV mode -- read from enabled_devices CSV file
    csv_path = config["csv"].get("enabled_devices_path")
    if csv_path:
        abs_path = _resolve_path(csv_path)
        if os.path.isfile(abs_path):
            df = pd.read_csv(abs_path, dtype={"BLDG_ID": str, "DEV_ID": int, "FALT_PRCV_YN": str})
            df = df[df["FALT_PRCV_YN"] == "Y"]
            devices = [{"bldg_id": row["BLDG_ID"], "dev_id": row["DEV_ID"]} for _, row in df.iterrows()]
            logger.info("CSV mode: read %d enabled devices from %s", len(devices), abs_path)
            return devices
        logger.warning("CSV mode: enabled_devices file not found: %s, using fallback", abs_path)

    # Fallback: hardcoded test devices (backwards compat)
    devices = [{"bldg_id": "B0019", "dev_id": 2001}]
    logger.info("CSV mode: returning hardcoded device list (%d devices)", len(devices))
    return devices
```

Key points:
- Falls back to hardcoded list if `enabled_devices_path` is not in config or file doesn't exist
- Uses `_resolve_path()` to resolve relative paths (already defined in db_connection.py)
- Filters `FALT_PRCV_YN == 'Y'` to match DB query behavior

**Step 2: Run integration test to verify no regression**

```bash
cd /workspace/2024_AI_BEMS/YiUmGoV2
python -m pytest tests/test_integration.py -v
```

Expected: PASS (the enabled_devices.csv file exists now, so it should read both 2001 and 2002. The test should still pass for 2001 since its model exists.)

**Step 3: Commit**

```bash
git add db_connection.py
git commit -m "feat: read enabled devices from CSV file in CSV mode"
```

---

### Task 4: Update `write_anomaly_result()` in `db_connection.py`

**Files:**
- Modify: `db_connection.py:241-250`

**Step 1: Add CSV file writing to the CSV mode block**

In `db_connection.py`, replace the CSV mode block in `write_anomaly_result()` (lines 241-250):

```python
    # CSV mode -- just log the result
    status = "ANOMALY" if ad_score <= config["anomaly"]["score_threshold"] else "NORMAL"
    msg = (
        f"[{now:%Y-%m-%d %H:%M:%S}] "
        f"bldg_id={bldg_id}, dev_id={dev_id}, "
        f"ad_score={ad_score:.2f} ({status}), "
        f"ad_desc={ad_desc}"
    )
    print(msg)
    logger.info("CSV mode (write_anomaly_result): %s", msg)
```

with:

```python
    # CSV mode -- write to result CSV file and also log
    status = "ANOMALY" if ad_score <= config["anomaly"]["score_threshold"] else "NORMAL"
    msg = (
        f"[{now:%Y-%m-%d %H:%M:%S}] "
        f"bldg_id={bldg_id}, dev_id={dev_id}, "
        f"ad_score={ad_score:.2f} ({status}), "
        f"ad_desc={ad_desc}"
    )
    print(msg)
    logger.info("CSV mode (write_anomaly_result): %s", msg)

    # Append to result CSV if path is configured
    result_path = config["csv"].get("result_path")
    if result_path:
        abs_path = _resolve_path(result_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        row = pd.DataFrame([{
            "USE_DT": now.strftime("%Y-%m-%d %H:%M:%S"),
            "BLDG_ID": bldg_id,
            "DEV_ID": str(dev_id),
            "AD_SCORE": round(ad_score, 2),
            "AD_DESC": ad_desc,
        }])
        write_header = not os.path.isfile(abs_path)
        row.to_csv(abs_path, mode="a", header=write_header, index=False)
        logger.info("CSV mode: result appended to %s", abs_path)
```

Key points:
- Creates `output/` directory if it doesn't exist
- Writes header only on first write (when file doesn't exist)
- Appends on subsequent writes
- Preserves existing print/log behavior
- Does nothing extra if `result_path` is not in config (backwards compat)

**Step 2: Quick manual verification**

```bash
cd /workspace/2024_AI_BEMS/YiUmGoV2
python -c "
import json, db_connection as DB
with open('_config.json') as f:
    config = json.load(f)
source = DB.create_data_source(config)
DB.write_anomaly_result(source, config, 'B0019', 9999, 75.0, 'test desc')
import pandas as pd
print(pd.read_csv('output/anomaly_results.csv'))
"
```

Expected: A CSV file with one row for dev_id=9999, ad_score=75.0.

**Step 3: Clean up test output and commit**

```bash
rm -f output/anomaly_results.csv
git add db_connection.py
git commit -m "feat: write anomaly results to CSV file in CSV mode"
```

---

### Task 5: Create `run_e2e_test.py`

**Files:**
- Create: `run_e2e_test.py`

**Step 1: Write the E2E test script**

Create `run_e2e_test.py` with the following content:

```python
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
```

**Step 2: Commit**

```bash
git add run_e2e_test.py
git commit -m "feat: add E2E test script for CSV-mode anomaly detection"
```

---

### Task 6: Run full E2E test

**Files:**
- None (execution only)

**Step 1: Run the E2E test**

```bash
cd /workspace/2024_AI_BEMS/YiUmGoV2
python run_e2e_test.py
```

Expected output:
- Step 1: Model trained and saved to `models/anomaly/2002.txt`
- Step 2: DEV_ID=2002 found in `dataset/enabled_devices.csv`
- Step 3: Runner processes both 2001 and 2002, writes results
- Step 4: Validation passes for DEV_ID=2002 row in `output/anomaly_results.csv`
- Summary: All 4 steps PASS

**Step 2: Verify output file**

```bash
cat output/anomaly_results.csv
```

Expected: CSV with columns `USE_DT,BLDG_ID,DEV_ID,AD_SCORE,AD_DESC` and rows for both 2001 and 2002.

**Step 3: Run existing unit tests to confirm no regression**

```bash
python -m pytest tests/ -v
```

Expected: All 44+ tests pass.

**Step 4: Commit the trained model**

```bash
git add models/anomaly/2002.txt
git commit -m "feat: add trained model for DEV_ID=2002"
```

---

### Task 7: Final commit and cleanup

**Files:**
- None (git only)

**Step 1: Add output/ to .gitignore**

The `output/` directory contains runtime results and should not be tracked. Create or update `.gitignore`:

```
output/
```

**Step 2: Final commit**

```bash
git add .gitignore
git commit -m "chore: add output/ to gitignore"
```
