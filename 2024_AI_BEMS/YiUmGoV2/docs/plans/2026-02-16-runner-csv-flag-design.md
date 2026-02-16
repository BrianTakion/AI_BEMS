# Design: ai_anomaly_runner.py --csv Flag

**Date:** 2026-02-16

## Goal

Replace `--dry-run` with `--csv` flag that forces full CSV mode (read devices from CSV, read sensor data from CSV, write results to CSV output file). Default (no flag) = DB production mode.

## Changes

### 1. CLI Arguments

**Remove:** `--dry-run`

**Add:** `--csv` (`action="store_true"`) — forces `data_source="csv"`

Default (no flag): `data_source="db"` (production)

### 2. Config

Update `_config.json`: `csv.enabled_devices_path` → `dataset/config_anomaly_devices.csv`

### 3. `process_device` Simplification

Remove `dry_run` parameter. CSV vs DB write is already handled by `db_connection.write_anomaly_result()` based on `data_source`.

### 4. `main()` Changes

- Replace `args.dry_run` with `args.csv`
- After `load_config()`: `config["data_source"] = "csv" if args.csv else "db"`
- Remove `dry_run=` from `process_device()` call
- Update log messages

### 5. README.md

Update Quick Start and docstring to reflect `--csv` usage.

## Approach

Minimal change. The `--csv` flag maps directly to the existing `data_source` toggle in `db_connection.py`. All CSV/DB branching already exists.
