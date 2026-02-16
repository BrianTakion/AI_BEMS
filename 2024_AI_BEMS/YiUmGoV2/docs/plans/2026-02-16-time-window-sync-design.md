# Training/Inference Time-Window Synchronization Design

**Date:** 2026-02-16
**Status:** Approved

## Problem

Training and inference use different data window strategies, causing feature engineering inconsistency:
- Training processes the full historical dataset at once (months of data)
- Inference processes only the last 176h
- Features (lags, rolling stats) computed on different-sized inputs produce different values for the same time point

Additionally, CSV-mode inference selects data by row count (`tail(704)`) while DB-mode uses time-based filtering, causing window size variation when duplicates/gaps exist.

## Design Decisions

1. **Training: CSV-only** with random 176h window sampling (no DB connection for training)
2. **Inference: CSV (dev) / DB (production)** -- no change to dual-mode support
3. **preprocess()** always receives a fixed 176h window (`window_df` parameter)
4. **Scoring window: 2h (8 points)** for both training samples and inference scoring
5. **Max training steps: 1000** random windows (= 8,000 total training samples)
6. **Current dedup/fill logic is sufficient** -- no changes to data quality handling

## Architecture

```
Training Flow (CSV only):
  CSV file -> load all historical data -> random 176h window sampling (x1000)
    -> preprocess(window_df) -> take last 8 points -> collect all samples
    -> train LightGBM once -> save model

Inference Flow (CSV dev / DB production):
  CSV or DB -> load last 176h (TIME-based)
    -> preprocess(window_df) -> take last 8 points (2h scoring window)
    -> model.predict() -> compute AD_SCORE -> output
```

## Config Changes (_config.json)

```json
{
  "data": {
    "scoring_window_hours": 2
  },
  "training": {
    "max_steps": 1000,
    "samples_per_window": 8
  }
}
```

- `scoring_window_hours`: 4 -> 2 (match training sample size)
- `training.max_steps`: new, number of random 176h windows to sample
- `training.samples_per_window`: new, last N points per window (2h = 8 points at 15min)

## File Changes

### data_preprocessing.py
- Rename parameter `raw_df` -> `window_df` (all references)
- No logic changes

### train_anomaly.py (major refactor)
- Remove full-dataset-at-once training approach
- New flow:
  1. Load all historical CSV data (same `_load_csv_chunked`)
  2. Compute valid random window range: [min_time + 176h, max_time]
  3. Loop max_steps times:
     - Pick random end_time within valid range
     - Slice [end_time - 176h, end_time] by TIME from historical data
     - Build window_df (index=datetime, column='value')
     - Call preprocess(window_df, config, fill_method="ffill")
     - Take last samples_per_window rows from X_df, y_df
     - Append to collection lists
  4. Concatenate all collected samples
  5. Train LightGBM once on full collected dataset
  6. Evaluate and save model
- Remove DB training path (CSV-only per requirement)

### db_connection.py
- Fix read_sensor_data() CSV mode: replace row-count filter with TIME-based filter:
  ```python
  cutoff = df["colec_dt"].iloc[-1] - timedelta(hours=fetch_hours)
  df = df[df["colec_dt"] >= cutoff]
  ```

### ai_anomaly_runner.py
- Update preprocess() call for renamed parameter
- Scoring window automatically uses new scoring_window_hours=2 from config

### infer_anomaly.py
- No changes needed

### test_integration.py
- Update for renamed parameter

## Time-Window Synchronization Guarantees

| Property           | Training                  | Inference                          | Match |
|--------------------|---------------------------|------------------------------------|-------|
| Window size        | 176h (config)             | 176h (config)                      | Yes   |
| Window selection   | TIME-based random         | TIME-based (DB cutoff / CSV fix)   | Yes   |
| preprocess input   | window_df (176h)          | window_df (176h)                   | Yes   |
| fill_method        | "ffill"                   | "ffill"                            | Yes   |
| sampling_minutes   | 15 (config)               | 15 (config)                        | Yes   |
| Output samples     | last 8 points (2h)        | last 8 points (2h)                 | Yes   |
| Feature context    | ~168h warmup + ~8h valid  | ~168h warmup + ~8h valid           | Yes   |

## What Does NOT Change

- preprocess() internal logic (feature engineering, dedup, fill)
- infer_anomaly.py (scoring functions)
- Model architecture (LightGBM regression)
- DB mode for inference (already time-based)
- Data quality handling (existing dedup/fill is sufficient)
