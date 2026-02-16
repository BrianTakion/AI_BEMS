#!/usr/bin/env python
"""
ai_anomaly_runner.py
Main entry point for AI BEMS anomaly detection.
Designed to be executed hourly via cron.

Flow: read config -> get enabled devices -> fetch data -> preprocess -> infer -> write results

Usage:
    python ai_anomaly_runner.py             # normal run (writes results)
    python ai_anomaly_runner.py --dry-run   # dry-run (logs only, no DB write)
"""

import argparse
import json
import logging
import os
import time

import db_connection
import data_preprocessing
import infer_anomaly

# ---------------------------------------------------------------------------
# Resolve paths relative to this script's directory (YiUmGoV2/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger("ai_anomaly_runner")


def setup_logging():
    """Configure root logger with INFO level and timestamp format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config():
    """Load _config.json from the script directory."""
    config_path = os.path.join(SCRIPT_DIR, "_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Config loaded from %s", config_path)
    return config


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
    # 1. Determine model path
    model_dir = config["anomaly"]["model_dir"]
    model_path = os.path.join(SCRIPT_DIR, model_dir, f"{dev_id}.txt")

    # 2. Load model -- skip if not found
    try:
        model = infer_anomaly.load_model(model_path)
    except FileNotFoundError:
        logger.warning("No model file for dev_id=%s (expected: %s) -- skipping", dev_id, model_path)
        return None
    logger.info("Model loaded: %s", model_path)

    # 3. Read sensor data
    raw_df = db_connection.read_sensor_data(source, config, bldg_id, dev_id)

    # 4. Check if data is sufficient (at least 2 rows)
    if len(raw_df) < 2:
        logger.warning("Insufficient data for dev_id=%s (%d rows) -- skipping", dev_id, len(raw_df))
        return None

    # 5. Build df_raw: index=colec_dt, columns=['value']
    df_raw = raw_df.set_index("colec_dt")[["colec_val"]].rename(columns={"colec_val": "value"})

    # 6. Preprocess
    result = data_preprocessing.preprocess(df_raw, config, fill_method="ffill")
    X_df, y_df, nan_counts_df, missing_ratio = result

    # 7. Skip if X_df is empty after preprocessing
    if X_df.empty:
        logger.warning("Empty features after preprocessing for dev_id=%s -- skipping", dev_id)
        return None
    logger.info("Preprocessed: %d samples, missing_ratio=%.1f", len(X_df), missing_ratio)

    # 8. Run inference
    y_predicted = infer_anomaly.run_inference(model, X_df)
    y_actual = y_df.values

    # 9. Slice the last scoring window for AD (feature engineering uses
    #    a larger historical window, but AD_SCORE reflects only the most
    #    recent input_interval_hours).
    sampling_min = config["data"]["sampling_minutes"]
    input_hours = config["data"]["input_interval_hours"]
    window_size = (60 // sampling_min) * input_hours
    y_actual_window = y_actual[-window_size:]
    y_predicted_window = y_predicted[-window_size:]

    # 10. Compute anomaly score and description on the scoring window
    ad_score = infer_anomaly.compute_ad_score(y_actual_window, y_predicted_window, config)
    ad_desc = infer_anomaly.generate_ad_desc(y_actual_window, y_predicted_window, ad_score, config)

    threshold = config["anomaly"]["score_threshold"]
    status = "ANOMALY" if ad_score <= threshold else "NORMAL"
    logger.info(
        "dev_id=%s => AD_SCORE=%.2f (%s) [%d/%d samples in %dh window] | %s",
        dev_id, ad_score, status, len(y_actual_window), len(y_actual), input_hours, ad_desc,
    )

    # 11. Write result (unless dry-run)
    if dry_run:
        logger.info("(dry-run, not written) bldg_id=%s, dev_id=%s, ad_score=%.2f", bldg_id, dev_id, ad_score)
    else:
        db_connection.write_anomaly_result(source, config, bldg_id, dev_id, ad_score, ad_desc)

    return (ad_score, ad_desc)


def main():
    """Main entry point: parse args, load config, process all enabled devices."""
    # 1. Parse CLI args
    parser = argparse.ArgumentParser(description="AI BEMS Anomaly Detection Runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing results to DB (just logs)",
    )
    args = parser.parse_args()

    # 2. Load config
    # 3. Set up logging
    setup_logging()
    config = load_config()

    # 4. Log start
    logger.info("=== AI Anomaly Detection Start ===")
    if args.dry_run:
        logger.info("Mode: DRY-RUN (results will NOT be written)")

    start_time = time.time()

    # 5. Create data source
    source = db_connection.create_data_source(config)

    # 6. Get enabled devices
    devices = db_connection.read_enabled_devices(source, config)

    # 7. Log number of enabled devices
    logger.info("Enabled devices: %d", len(devices))

    # 8. Process each device
    success_count = 0
    total_count = len(devices)

    for device in devices:
        bldg_id = device["bldg_id"]
        dev_id = device["dev_id"]
        logger.info("Processing: bldg_id=%s, dev_id=%s", bldg_id, dev_id)

        try:
            result = process_device(source, config, bldg_id, dev_id, dry_run=args.dry_run)
            if result is not None:
                success_count += 1
        except Exception:
            logger.exception("Error processing bldg_id=%s, dev_id=%s", bldg_id, dev_id)

    # 9. Log summary
    elapsed = time.time() - start_time
    logger.info(
        "=== Done: %d/%d devices processed in %.1fs ===",
        success_count, total_count, elapsed,
    )


if __name__ == "__main__":
    main()
