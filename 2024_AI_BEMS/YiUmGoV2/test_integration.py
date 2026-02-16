"""
test_integration.py
End-to-end integration test for the anomaly detection pipeline.

Validates the full pipeline (data loading -> preprocessing -> inference -> scoring)
using CSV mode. No live database connection is required.

Usage:
    cd YiUmGoV2 && python test_integration.py
"""

import os
import sys
import json

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve paths relative to this script's directory (YiUmGoV2/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_full_pipeline():
    # ------------------------------------------------------------------
    # 1. Load _config.json
    # ------------------------------------------------------------------
    config_path = os.path.join(SCRIPT_DIR, "_config.json")
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"  data_source = {config['data_source']}")

    # ------------------------------------------------------------------
    # 2. Create data source via db_connection
    # ------------------------------------------------------------------
    import db_connection as DB

    source = DB.create_data_source(config)
    print("Data source created.")

    # ------------------------------------------------------------------
    # 3. Get enabled devices
    # ------------------------------------------------------------------
    devices = DB.read_enabled_devices(source, config)
    print(f"Enabled devices: {devices}")

    # ------------------------------------------------------------------
    # 4. Process dev_id 2001
    # ------------------------------------------------------------------
    dev_id = 2001
    bldg_id = "B0019"

    # 4a. Check that the CSV data file exists
    csv_rel_path = config["csv"]["data_path"]
    csv_abs_path = os.path.normpath(os.path.join(SCRIPT_DIR, csv_rel_path))
    if not os.path.isfile(csv_abs_path):
        print(f"SKIP: CSV data file not found: {csv_abs_path}")
        return

    # 4a. Read sensor data
    print(f"\nReading sensor data for bldg_id={bldg_id}, dev_id={dev_id} ...")
    print("  (This may take ~1 min for chunked CSV reading of a large file)")
    df_sensor = DB.read_sensor_data(source, config, bldg_id, dev_id)

    # 4b. Print row count
    print(f"  Sensor data rows: {len(df_sensor)}")
    if df_sensor.empty:
        print("SKIP: No sensor data returned for dev_id=2001")
        return

    # 4c. Build df_raw: index=colec_dt, columns=['value']
    df_raw = df_sensor[["colec_dt", "colec_val"]].copy()
    df_raw = df_raw.rename(columns={"colec_val": "value"})
    df_raw = df_raw.set_index("colec_dt")
    df_raw.index.name = None
    print(f"  df_raw shape: {df_raw.shape}")

    # 4d. Preprocess
    import data_preprocessing

    X_df, y_df, nan_counts, missing_ratio = data_preprocessing.preprocess(
        df_raw, config, fill_method="ffill"
    )

    # 4e. Print feature count, sample count
    print(f"  Feature count: {X_df.shape[1]}")
    print(f"  Sample count:  {X_df.shape[0]}")
    print(f"  Missing ratio: {missing_ratio}")

    if X_df.empty:
        print("SKIP: Preprocessing returned 0 samples (not enough data for lags)")
        return

    # 4f. Load model
    import infer_anomaly

    model_dir = config["anomaly"]["model_dir"]
    model_path = os.path.join(SCRIPT_DIR, model_dir, f"{dev_id}.txt")
    if not os.path.isfile(model_path):
        print(f"SKIP: Model file not found: {model_path}")
        return

    print(f"  Loading model from: {model_path}")
    model = infer_anomaly.load_model(model_path)

    # 4g. Run inference
    y_pred = infer_anomaly.run_inference(model, X_df)
    print(f"  Predictions count: {len(y_pred)}")

    # 4h. Slice last scoring window
    sampling_min = config["data"]["sampling_minutes"]
    input_hours = config["data"]["input_interval_hours"]
    window_size = (60 // sampling_min) * input_hours
    y_actual_window = y_df.values[-window_size:]
    y_pred_window = y_pred[-window_size:]
    print(f"  Scoring window ({input_hours}h): last {len(y_actual_window)} of {len(y_pred)} samples")

    # 4i. Compute ad_score and ad_desc on scoring window
    ad_score = infer_anomaly.compute_ad_score(y_actual_window, y_pred_window, config)
    ad_desc = infer_anomaly.generate_ad_desc(y_actual_window, y_pred_window, ad_score, config)

    # 4j. Print AD_SCORE and AD_DESC
    print(f"\n  AD_SCORE: {ad_score:.2f}")
    print(f"  AD_DESC:  {ad_desc}")

    # 4k. Assert 0 <= ad_score <= 100
    assert 0 <= ad_score <= 100, (
        f"ad_score out of range: {ad_score}"
    )

    # 4l. Assert len(ad_desc) <= 1000
    assert len(ad_desc) <= 1000, (
        f"ad_desc too long: {len(ad_desc)} chars (max 1000)"
    )

    # ------------------------------------------------------------------
    # 5. Done
    # ------------------------------------------------------------------
    print("\n=== INTEGRATION TEST PASSED ===")


if __name__ == "__main__":
    test_full_pipeline()
