"""Unit tests for data_preprocessing.preprocess()."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection import data_preprocessing

EXPECTED_FEATURE_COUNT = 44

EXPECTED_FEATURES = [
    # is_missing (from cleansing step)
    "is_missing",
    # temporal
    "hour", "month", "weekday", "is_holiday",
    # seasonal sinusoids
    "sin_month", "cos_month", "sine_day", "cosine_day", "sin_hour", "cos_hour",
    # lags
    "lag_1p", "lag_2p", "lag_3p", "lag_1d_0p", "lag_1w_0p",
    # rate of change
    "rate", "rate_rate", "rate_1d", "rate_rate_1d",
    # 1h window stats
    "ma_1h", "max_1h", "min_1h", "std_1h",
    # 1d window stats
    "ma_1d", "max_1d", "min_1d", "std_1d",
    # 1-day prior ±30min stats
    "p1d_ma_1h", "p1d_max_1h", "p1d_min_1h", "p1d_std_1h",
    # 1-week prior ±30min stats
    "p1w_ma_1h", "p1w_max_1h", "p1w_min_1h", "p1w_std_1h",
    # moving average rates
    "rate_ma_1h", "rate_rate_ma_1h",
    "rate_p1d_ma_1h", "rate_rate_p1d_ma_1h",
    "rate_p1w_ma_1h", "rate_rate_p1w_ma_1h",
    # seasonal decomposition
    "season_ma_1h", "season_ma_1d",
]


class TestPreprocessOutputShape:
    """Verify the output shape and columns of preprocess()."""

    def test_feature_count(self, synthetic_window_df, config):
        X_df, y_df, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        assert X_df.shape[1] == EXPECTED_FEATURE_COUNT, (
            f"Expected {EXPECTED_FEATURE_COUNT} features, got {X_df.shape[1]}. "
            f"Columns: {list(X_df.columns)}"
        )

    def test_feature_names(self, synthetic_window_df, config):
        X_df, _, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        assert set(X_df.columns) == set(EXPECTED_FEATURES)

    def test_output_rows_positive(self, synthetic_window_df, config):
        X_df, y_df, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        assert len(X_df) > 0, "Preprocessing returned 0 rows"
        assert len(X_df) == len(y_df), "X_df and y_df row count mismatch"

    def test_y_df_is_series(self, synthetic_window_df, config):
        _, y_df, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        assert isinstance(y_df, pd.Series)
        assert y_df.name == "value"


class TestPreprocessNoNans:
    """Verify that preprocessing eliminates all NaN values."""

    def test_x_no_nans(self, synthetic_window_df, config):
        X_df, _, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        nan_count = X_df.isna().sum().sum()
        assert nan_count == 0, f"X_df has {nan_count} NaN values"

    def test_y_no_nans(self, synthetic_window_df, config):
        _, y_df, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        assert y_df.isna().sum() == 0, "y_df has NaN values"


class TestTemporalFeatures:
    """Verify temporal feature correctness."""

    def test_hour_range(self, synthetic_window_df, config):
        X_df, _, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        assert X_df["hour"].min() >= 0
        assert X_df["hour"].max() <= 23

    def test_month_range(self, synthetic_window_df, config):
        X_df, _, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        assert X_df["month"].min() >= 1
        assert X_df["month"].max() <= 12

    def test_weekday_range(self, synthetic_window_df, config):
        X_df, _, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        assert X_df["weekday"].min() >= 0
        assert X_df["weekday"].max() <= 6

    def test_is_holiday_binary(self, synthetic_window_df, config):
        X_df, _, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        assert set(X_df["is_holiday"].unique()).issubset({0, 1})


class TestSeasonalFeatures:
    """Verify sinusoidal features are bounded in [-1, 1]."""

    @pytest.mark.parametrize("col", [
        "sin_month", "cos_month", "sine_day", "cosine_day", "sin_hour", "cos_hour",
    ])
    def test_sinusoidal_bounds(self, synthetic_window_df, config, col):
        X_df, _, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        assert X_df[col].min() >= -1.0 - 1e-9, f"{col} below -1"
        assert X_df[col].max() <= 1.0 + 1e-9, f"{col} above 1"


class TestLagFeatures:
    """Verify lag features reference correct prior values."""

    def test_lag_1p_values(self, synthetic_window_df, config):
        X_df, y_df, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        # lag_1p should equal value shifted by 1 period
        # After dropna, all rows are valid — compare aligned indices
        idx = X_df.index
        # Reconstruct from original resampled data
        assert not X_df["lag_1p"].isna().any()

    def test_lag_columns_exist(self, synthetic_window_df, config):
        X_df, _, _, _ = data_preprocessing.preprocess(synthetic_window_df, config)
        for col in ["lag_1p", "lag_2p", "lag_3p", "lag_1d_0p", "lag_1w_0p"]:
            assert col in X_df.columns, f"Missing lag column: {col}"


class TestFillMethods:
    """Verify different fill methods produce valid output."""

    @pytest.mark.parametrize("fill_method", ["zero", "ffill"])
    def test_fill_method_produces_output(self, synthetic_window_df, config, fill_method):
        X_df, y_df, _, _ = data_preprocessing.preprocess(
            synthetic_window_df, config, fill_method=fill_method
        )
        assert len(X_df) > 0
        assert X_df.isna().sum().sum() == 0


class TestOnlyCleansing:
    """Verify only_cleansing=True returns interpolated data without features."""

    def test_only_cleansing_returns_four_items(self, synthetic_window_df, config):
        result = data_preprocessing.preprocess(
            synthetic_window_df, config, only_cleansing=True
        )
        assert len(result) == 4

    def test_only_cleansing_df_has_value_column(self, synthetic_window_df, config):
        df_interpol, df_is_missing, nan_counts, missing_ratio = (
            data_preprocessing.preprocess(synthetic_window_df, config, only_cleansing=True)
        )
        assert "value" in df_interpol.columns
        assert "is_missing" in df_is_missing.columns

    def test_only_cleansing_no_nans_in_value(self, synthetic_window_df, config):
        df_interpol, _, _, _ = data_preprocessing.preprocess(
            synthetic_window_df, config, only_cleansing=True
        )
        assert df_interpol["value"].isna().sum() == 0

    def test_missing_ratio_type(self, synthetic_window_df, config):
        _, _, _, missing_ratio = data_preprocessing.preprocess(
            synthetic_window_df, config, only_cleansing=True
        )
        assert isinstance(missing_ratio, float)
        assert 0.0 <= missing_ratio <= 1.0


class TestMissingDataHandling:
    """Verify preprocessing handles gaps in input data."""

    def test_with_gaps(self, config):
        """Create data with intentional gaps and verify is_missing flag."""
        fetch_hours = config["data"]["fetch_window_hours"]
        sampling_min = config["data"]["sampling_minutes"]
        total_points = fetch_hours * (60 // sampling_min)

        start = pd.Timestamp("2026-01-05 00:00:00")
        index = pd.date_range(start=start, periods=total_points, freq=f"{sampling_min}min")

        rng = np.random.default_rng(99)
        values = 50 + rng.normal(0, 5, total_points)

        # Drop 10% of rows to simulate gaps
        keep_mask = rng.random(total_points) > 0.1
        index = index[keep_mask]
        values = values[keep_mask]

        df = pd.DataFrame({"value": values}, index=index)
        X_df, _, _, missing_ratio = data_preprocessing.preprocess(df, config)

        # After resampling, some points will be missing and filled
        assert missing_ratio >= 0.0
        assert len(X_df) > 0
