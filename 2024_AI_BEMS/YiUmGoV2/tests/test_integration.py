"""
End-to-end integration test for the anomaly detection pipeline (pytest version).

Validates the full pipeline (data loading -> preprocessing -> inference -> scoring)
using CSV mode. No live database connection is required.

Usage:
    cd YiUmGoV2 && python -m pytest tests/test_integration.py -v
"""

import os

import pandas as pd
import pytest

from anomaly_detection import data_source as DS
from anomaly_detection import data_preprocessing
from anomaly_detection import infer_anomaly

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def source(config):
    return DS.create_data_source(config)


@pytest.fixture
def devices(source, config):
    return DS.read_enabled_devices(source, config)


@pytest.fixture
def sensor_data(source, config):
    """Load sensor data for dev_id=2001. Skip if CSV file is missing."""
    csv_path = os.path.normpath(
        os.path.join(SCRIPT_DIR, config["csv"]["data_path"])
    )
    if not os.path.isfile(csv_path):
        pytest.skip(f"CSV data file not found: {csv_path}")

    df = DS.read_sensor_data(source, config, bldg_id="B0019", dev_id=2001)
    if df.empty:
        pytest.skip("No sensor data returned for dev_id=2001")
    return df


@pytest.fixture
def window_df(sensor_data):
    """Build window_df: index=colec_dt, columns=['value']."""
    df = sensor_data[["colec_dt", "colec_val"]].copy()
    df = df.rename(columns={"colec_val": "value"})
    df = df.set_index("colec_dt")
    df.index.name = None
    return df


@pytest.fixture
def preprocessed(window_df, config):
    X_df, y_df, nan_counts, missing_ratio = data_preprocessing.preprocess(
        window_df, config, fill_method="ffill"
    )
    if X_df.empty:
        pytest.skip("Preprocessing returned 0 samples (not enough data for lags)")
    return X_df, y_df, nan_counts, missing_ratio


@pytest.fixture
def model(config):
    model_dir = config["anomaly"]["model_dir"]
    model_path = os.path.join(SCRIPT_DIR, "anomaly_detection", model_dir, "2001.txt")
    if not os.path.isfile(model_path):
        pytest.skip(f"Model file not found: {model_path}")
    return infer_anomaly.load_model(model_path)


@pytest.fixture
def scoring_results(preprocessed, model, config):
    """Run inference and compute anomaly score/description."""
    X_df, y_df, _, _ = preprocessed

    y_pred = infer_anomaly.run_inference(model, X_df)

    sampling_min = config["data"]["sampling_minutes"]
    scoring_hours = config["data"]["scoring_window_hours"]
    window_size = (60 // sampling_min) * scoring_hours

    y_actual_window = y_df.values[-window_size:]
    y_pred_window = y_pred[-window_size:]

    ad_score = infer_anomaly.compute_ad_score(y_actual_window, y_pred_window, config)
    ad_desc = infer_anomaly.generate_ad_desc(
        y_actual_window, y_pred_window, ad_score, config
    )
    return ad_score, ad_desc, y_pred


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_config_has_data_source(self, config):
        assert "data_source" in config

    def test_config_data_section(self, config):
        assert "data" in config
        assert "fetch_window_hours" in config["data"]
        assert "scoring_window_hours" in config["data"]
        assert "sampling_minutes" in config["data"]


class TestDataSource:
    def test_create_data_source(self, source):
        assert source is not None

    def test_enabled_devices_not_empty(self, devices):
        assert len(devices) > 0

    def test_device_has_required_keys(self, devices):
        for dev in devices:
            assert "bldg_id" in dev
            assert "dev_id" in dev


class TestSensorData:
    def test_sensor_data_not_empty(self, sensor_data):
        assert len(sensor_data) > 0

    def test_sensor_data_columns(self, sensor_data):
        assert "colec_dt" in sensor_data.columns
        assert "colec_val" in sensor_data.columns


class TestPreprocessing:
    def test_feature_count(self, preprocessed):
        X_df, _, _, _ = preprocessed
        assert X_df.shape[1] == 44

    def test_sample_count_positive(self, preprocessed):
        X_df, _, _, _ = preprocessed
        assert X_df.shape[0] > 0

    def test_missing_ratio_valid(self, preprocessed):
        _, _, _, missing_ratio = preprocessed
        assert 0.0 <= missing_ratio <= 1.0


class TestInference:
    def test_predictions_count(self, preprocessed, scoring_results):
        X_df, _, _, _ = preprocessed
        _, _, y_pred = scoring_results
        assert len(y_pred) == len(X_df)


class TestAnomalyScoring:
    def test_ad_score_range(self, scoring_results):
        ad_score, _, _ = scoring_results
        assert 0 <= ad_score <= 100, f"ad_score out of range: {ad_score}"

    def test_ad_desc_length(self, scoring_results):
        _, ad_desc, _ = scoring_results
        assert len(ad_desc) <= 1000, (
            f"ad_desc too long: {len(ad_desc)} chars (max 1000)"
        )

    def test_ad_desc_not_empty(self, scoring_results):
        _, ad_desc, _ = scoring_results
        assert len(ad_desc) > 0
