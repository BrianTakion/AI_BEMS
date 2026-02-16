# E2E CSV-Based Anomaly Detection Test Design

**Date:** 2026-02-16
**Status:** Approved

## Goal

Test the end-to-end anomaly detection pipeline for DEV_ID=2002 without PostgreSQL, using CSV files as stand-ins for all three DB operations: device configuration reads, sensor data reads, and result writes.

## Approach: Thin CSV Adapter Layer

Extend `db_connection.py` so that CSV mode handles all three data operations via files, then create a manual E2E script that exercises the full pipeline.

## Changes

### 1. `db_connection.py`

**`read_enabled_devices()` (CSV mode):**
- Replace hardcoded device list with reading from `dataset/enabled_devices.csv`
- CSV schema: `BLDG_ID,DEV_ID,FALT_PRCV_YN`
- Filter for `FALT_PRCV_YN='Y'` (mirrors the DB query logic)

**`write_anomaly_result()` (CSV mode):**
- Instead of only logging, append a row to `output/anomaly_results.csv`
- CSV schema: `USE_DT,BLDG_ID,DEV_ID,AD_SCORE,AD_DESC`
- Create the `output/` directory and file on first write; append on subsequent writes

**`read_sensor_data()` — no changes.** Already works in CSV mode.

### 2. `_config.json`

Add two new fields to the `csv` section:

```json
"csv": {
    "data_path": "dataset/data_colec_h_250730_260212_B0019.csv",
    "enabled_devices_path": "dataset/enabled_devices.csv",
    "result_path": "output/anomaly_results.csv"
}
```

### 3. `dataset/enabled_devices.csv`

New file mimicking `DEV_USE_PURP_REL_R`:

```csv
BLDG_ID,DEV_ID,FALT_PRCV_YN
B0019,2001,Y
B0019,2002,Y
```

### 4. `run_e2e_test.py`

Manual E2E script with 4 steps:

1. **Train DEV_ID=2002** — calls `python train_anomaly.py --dev_id 2002` via subprocess
2. **Register device** — creates/verifies `dataset/enabled_devices.csv` with DEV_ID=2002
3. **Run inference** — calls `python ai_anomaly_runner.py` via subprocess (simulates cron)
4. **Verify output** — reads `output/anomaly_results.csv` and checks:
   - Row exists for DEV_ID=2002
   - AD_SCORE in [0, 100]
   - AD_DESC is non-empty and <= 1000 chars
   - USE_DT is recent

Prints PASS/FAIL summary.

## Edge Cases

- **Training time:** Uses default `max_steps` from config (~10+ min). Reduce in config for faster testing.
- **Existing results:** Append mode; E2E script validates only the most recent DEV_ID=2002 row.
- **Existing model:** Training overwrites `models/anomaly/2002.txt` if it exists.
- **Directory creation:** `output/` created automatically by `write_anomaly_result()`.

## Files Modified

| File | Change |
|------|--------|
| `db_connection.py` | CSV-based `read_enabled_devices()` and `write_anomaly_result()` |
| `_config.json` | Add `enabled_devices_path` and `result_path` to `csv` section |

## Files Created

| File | Purpose |
|------|---------|
| `dataset/enabled_devices.csv` | Device config (mimics DEV_USE_PURP_REL_R) |
| `output/anomaly_results.csv` | Results output (mimics FALT_PRCV_FCST) |
| `run_e2e_test.py` | Manual E2E test script |
