# Design: train_anomaly.py Multi-Device Refactor

**Date:** 2026-02-16

## Goal

Simplify `train_anomaly.py` by removing date range arguments and adding multi-device training support with automatic skip when a model already exists.

## Changes

### 1. CLI Arguments

**Remove:** `--start_date`, `--end_date`

**Modify:** `--dev_id` from `type=int` to `type=int, nargs="+"` (one or more device IDs)

Usage after:
```
python train_anomaly.py --dev_id 2001
python train_anomaly.py --dev_id 2001 2002 2003
```

### 2. Skip Logic

Before loading data for each device, check if `models/anomaly/{dev_id}.txt` exists. If so, print `[SKIP] Model already exists: {path}` and continue to the next device.

### 3. `_load_csv_chunked` Simplification

Remove `start_date` and `end_date` parameters. The function loads all data for the given `dev_id`/`tag_cd`.

### 4. Loop Structure

Wrap existing training logic (steps 2-9 in current `main()`) in a `for dev_id in args.dev_id:` loop. Each device gets its own banner and summary output.

### 5. README.md

- Remove "Train with explicit date range" example
- Update training example to show multi-device usage

## Approach

Minimal loop refactor (Approach A) — smallest change that satisfies all requirements. No function extraction needed.
