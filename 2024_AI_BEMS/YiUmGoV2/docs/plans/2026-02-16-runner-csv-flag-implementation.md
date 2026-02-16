# ai_anomaly_runner.py --csv Flag — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `--dry-run` with `--csv` flag that forces full CSV mode, and update config to point to the correct device list file.

**Architecture:** The `--csv` flag overrides `config["data_source"]` to `"csv"` after loading config. Default (no flag) sets it to `"db"`. All CSV/DB branching already exists in `db_connection.py`. The `dry_run` parameter on `process_device` is removed since CSV mode writes to a local file (not DB).

**Tech Stack:** Python 3, argparse

---

### Task 1: Update `_config.json` — fix enabled_devices_path

**Files:**
- Modify: `_config.json:5`

**Step 1: Change the path**

In `_config.json`, change line 5:
```json
"enabled_devices_path": "dataset/enabled_devices.csv",
```
to:
```json
"enabled_devices_path": "dataset/config_anomaly_devices.csv",
```

**Step 2: Commit**

```bash
git add _config.json
git commit -m "fix: point enabled_devices_path to config_anomaly_devices.csv"
```

---

### Task 2: Refactor ai_anomaly_runner.py — replace --dry-run with --csv

**Files:**
- Modify: `ai_anomaly_runner.py`

**Step 1: Update module docstring (lines 2-12)**

Replace:
```python
"""
ai_anomaly_runner.py
Main entry point for AI BEMS anomaly detection.
Designed to be executed hourly via cron.

Flow: read config -> get enabled devices -> fetch data -> preprocess -> infer -> write results

Usage:
    python ai_anomaly_runner.py             # normal run (writes results)
    python ai_anomaly_runner.py --dry-run   # dry-run (logs only, no DB write)
"""
```

With:
```python
"""
ai_anomaly_runner.py
Main entry point for AI BEMS anomaly detection.
Designed to be executed hourly via cron.

Flow: read config -> get enabled devices -> fetch data -> preprocess -> infer -> write results

Usage:
    python ai_anomaly_runner.py          # DB mode (production, writes to DB)
    python ai_anomaly_runner.py --csv    # CSV mode (development, writes to output/)
"""
```

**Step 2: Simplify `process_device` (lines 50-126)**

Remove the `dry_run` parameter and the dry-run conditional block. Replace:

```python
def process_device(source, config, bldg_id, dev_id, dry_run=False):
    """Process a single device: load model, fetch data, preprocess, infer, write result.

    Args:
        source: Data source handle (DB engine or config dict for CSV mode).
        config: Configuration dict from _config.json.
        bldg_id: Building identifier string.
        dev_id: Device identifier (int).
        dry_run: If True, skip writing results to DB.

    Returns:
        Tuple (ad_score, ad_desc) on success, or None on skip/error.
    """
```

With:
```python
def process_device(source, config, bldg_id, dev_id):
    """Process a single device: load model, fetch data, preprocess, infer, write result.

    Args:
        source: Data source handle (DB engine or config dict for CSV mode).
        config: Configuration dict from _config.json.
        bldg_id: Building identifier string.
        dev_id: Device identifier (int).

    Returns:
        Tuple (ad_score, ad_desc) on success, or None on skip/error.
    """
```

And replace the dry-run conditional block (lines 120-124):
```python
    # 11. Write result (unless dry-run)
    if dry_run:
        logger.info("(dry-run, not written) bldg_id=%s, dev_id=%s, ad_score=%.2f", bldg_id, dev_id, ad_score)
    else:
        db_connection.write_anomaly_result(source, config, bldg_id, dev_id, ad_score, ad_desc)
```

With:
```python
    # 11. Write result (DB mode writes to DB, CSV mode writes to output file)
    db_connection.write_anomaly_result(source, config, bldg_id, dev_id, ad_score, ad_desc)
```

**Step 3: Rewrite `main()` CLI args and mode logic (lines 129-182)**

Replace the argument parser block (lines 131-137):
```python
    parser = argparse.ArgumentParser(description="AI BEMS Anomaly Detection Runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing results to DB (just logs)",
    )
    args = parser.parse_args()
```

With:
```python
    parser = argparse.ArgumentParser(description="AI BEMS Anomaly Detection Runner")
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Run in CSV mode (read from CSV files, write results to output/)",
    )
    args = parser.parse_args()
```

Replace the mode logging block (lines 145-148):
```python
    # 4. Log start
    logger.info("=== AI Anomaly Detection Start ===")
    if args.dry_run:
        logger.info("Mode: DRY-RUN (results will NOT be written)")
```

With:
```python
    # 4. Override data_source based on CLI flag
    config["data_source"] = "csv" if args.csv else "db"

    logger.info("=== AI Anomaly Detection Start ===")
    logger.info("Mode: %s", config["data_source"].upper())
```

Replace the process_device call (line 171):
```python
            result = process_device(source, config, bldg_id, dev_id, dry_run=args.dry_run)
```

With:
```python
            result = process_device(source, config, bldg_id, dev_id)
```

**Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('ai_anomaly_runner.py').read()); print('OK')"`
Expected: `OK`

**Step 5: Verify help output**

Run: `python ai_anomaly_runner.py --help`
Expected: Shows `--csv` flag, no `--dry-run`

**Step 6: Commit**

```bash
git add ai_anomaly_runner.py
git commit -m "feat: replace --dry-run with --csv flag for full CSV mode"
```

---

### Task 3: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Update Quick Start inference examples (lines 15-19)**

Replace:
```markdown
# Run inference (dry-run, no DB write)
python ai_anomaly_runner.py --dry-run

# Run inference (production, writes to DB)
python ai_anomaly_runner.py
```

With:
```markdown
# Run inference (CSV mode, writes to output/)
python ai_anomaly_runner.py --csv

# Run inference (DB mode, production)
python ai_anomaly_runner.py
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for --csv flag"
```

---

### Task 4: Final verification

**Step 1: Verify syntax**

Run: `python -c "import ast; ast.parse(open('ai_anomaly_runner.py').read()); print('OK')"`
Expected: `OK`

**Step 2: Verify help**

Run: `python ai_anomaly_runner.py --help`
Expected: `--csv` flag shown, no `--dry-run`

**Step 3: Run existing tests**

Run: `python -m pytest tests/ -v`
Expected: All existing tests pass
