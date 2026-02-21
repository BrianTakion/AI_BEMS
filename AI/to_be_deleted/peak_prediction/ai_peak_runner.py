#!/usr/bin/env python
"""
ai_peak_runner.py  --  Peak Prediction Inference Runner

Predicts daily peak power and peak time for enabled devices.
Designed to run hourly via cron, refining predictions as same-day data arrives.

Usage:
    python peak_prediction/ai_peak_runner.py          # DB mode (production)
    python peak_prediction/ai_peak_runner.py --csv    # CSV mode (development)
"""

import argparse
import json
import logging
import os
import time

import data_source
import data_preprocessing
import infer_peak

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("ai_peak_runner")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config():
    config_path = os.path.join(SCRIPT_DIR, "_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Config loaded from %s", config_path)
    return config


def process_device(source, config, bldg_id, dev_id):
    """Process a single device: load models, fetch data, preprocess, predict, write result.

    Returns:
        Formatted peak_info string on success, or None on skip/error.
    """
    model_dir = config["peak"]["model_dir"]
    power_model_path = os.path.join(SCRIPT_DIR, model_dir, f"{dev_id}_power.txt")
    time_model_path = os.path.join(SCRIPT_DIR, model_dir, f"{dev_id}_time.txt")

    # 1. Load models
    try:
        model_power = infer_peak.load_model(power_model_path)
        model_time = infer_peak.load_model(time_model_path)
    except FileNotFoundError as e:
        logger.warning("Model not found for dev_id=%s: %s -- skipping", dev_id, e)
        return None
    logger.info("Models loaded: %s, %s", power_model_path, time_model_path)

    # 2. Read sensor data (last 14 days = 336 hours)
    raw_df = data_source.read_sensor_data(source, config, bldg_id, dev_id)
    if len(raw_df) < 2:
        logger.warning("Insufficient data for dev_id=%s (%d rows) -- skipping",
                       dev_id, len(raw_df))
        return None

    # 3. Preprocess for inference
    try:
        X_df = data_preprocessing.preprocess_for_inference(raw_df, config)
    except (ValueError, Exception) as e:
        logger.error("Preprocessing failed for dev_id=%s: %s", dev_id, e)
        return None
    logger.info("Features built: %d columns", X_df.shape[1])

    # 4. Predict
    peak_power, peak_slot = infer_peak.predict_peak(model_power, model_time, X_df)

    # 5. Format result
    peak_info = infer_peak.format_peak_result(peak_power, peak_slot)
    logger.info("dev_id=%s => %s", dev_id, peak_info)

    # 6. Write result
    data_source.write_peak_result(source, config, bldg_id, peak_info)

    return peak_info


def main():
    parser = argparse.ArgumentParser(description="AI BEMS Peak Prediction Runner")
    parser.add_argument("--csv", action="store_true",
                        help="Run in CSV mode (development)")
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    config["data_source"] = "csv" if args.csv else "db"

    logger.info("=== AI Peak Prediction Start ===")
    logger.info("Mode: %s", config["data_source"].upper())

    start_time = time.time()

    source = data_source.create_data_source(config)
    devices = data_source.read_enabled_devices(source, config)
    logger.info("Enabled devices: %d", len(devices))

    success_count = 0
    for device in devices:
        bldg_id = device["bldg_id"]
        dev_id = device["dev_id"]
        logger.info("Processing: bldg_id=%s, dev_id=%s", bldg_id, dev_id)

        try:
            result = process_device(source, config, bldg_id, dev_id)
            if result is not None:
                success_count += 1
        except Exception:
            logger.exception("Error processing bldg_id=%s, dev_id=%s", bldg_id, dev_id)

    elapsed = time.time() - start_time
    logger.info("=== Done: %d/%d devices in %.1fs ===",
                success_count, len(devices), elapsed)


if __name__ == "__main__":
    main()
