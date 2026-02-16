import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error


def load_model(model_path):
    """Load LightGBM Booster from a .txt file.

    Args:
        model_path: Path to the LightGBM model text file.

    Returns:
        lgb.Booster instance loaded from the file.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = lgb.Booster(model_file=model_path)
    return model


def run_inference(model, X_df):
    """Run prediction using a LightGBM Booster model.

    Args:
        model: lgb.Booster instance.
        X_df: pandas DataFrame or array-like of input features.

    Returns:
        numpy array of predicted values.
    """
    predictions = model.predict(X_df)
    return np.array(predictions)


def compute_ad_score(y_actual, y_predicted, config):
    """Compute anomaly detection score on a 0-100 scale.

    Uses RMSE-based scoring normalized against the mean of actual values.
    A score of 100 means a perfect match (no anomaly), while a score
    at or below the threshold (default 50) indicates an anomaly.

    Args:
        y_actual: Array-like of actual observed values.
        y_predicted: Array-like of predicted values.
        config: Configuration dict (config['anomaly']['score_threshold']
                is used by the caller, not by this function).

    Returns:
        Float anomaly score between 0 and 100.
    """
    y_actual = np.asarray(y_actual, dtype=np.float64)
    y_predicted = np.asarray(y_predicted, dtype=np.float64)

    rmse = root_mean_squared_error(y_actual, y_predicted)
    mean_actual = np.mean(np.abs(y_actual))

    if mean_actual < 1e-8:
        # When actual values are near zero, even small RMSE is significant
        score = 0.0 if rmse > 1e-8 else 100.0
    else:
        score = max(0.0, 100.0 - (rmse / mean_actual) * 100.0)

    return float(score)


def generate_ad_desc(y_actual, y_predicted, ad_score, config):
    """Generate a statistical summary text description for anomaly detection.

    Includes mean, std, min, max of actual values, RMSE, trend direction
    (comparing first half vs second half of actual values), and an anomaly
    alert if the score is at or below the configured threshold.

    Args:
        y_actual: Array-like of actual observed values.
        y_predicted: Array-like of predicted values.
        ad_score: Float anomaly score (0-100).
        config: Configuration dict with config['anomaly']['score_threshold'].

    Returns:
        String description, truncated to 1000 characters max.
    """
    y_actual = np.asarray(y_actual, dtype=np.float64)
    y_predicted = np.asarray(y_predicted, dtype=np.float64)

    rmse = root_mean_squared_error(y_actual, y_predicted)

    mean_val = np.mean(y_actual)
    std_val = np.std(y_actual)
    min_val = np.min(y_actual)
    max_val = np.max(y_actual)

    # Trend direction: compare mean of first half vs second half
    mid = len(y_actual) // 2
    if mid > 0:
        first_half_mean = np.mean(y_actual[:mid])
        second_half_mean = np.mean(y_actual[mid:])
        if second_half_mean > first_half_mean + 1e-8:
            trend = "INCREASING"
        elif second_half_mean < first_half_mean - 1e-8:
            trend = "DECREASING"
        else:
            trend = "STABLE"
    else:
        trend = "STABLE"

    scoring_hours = config["data"]["scoring_window_hours"]
    if scoring_hours >= 24:
        window_label = f"{scoring_hours // 24}D Stats"
    else:
        window_label = f"{scoring_hours}H Stats"

    desc = (
        f"{window_label} | "
        f"Mean: {mean_val:.2f}, Std: {std_val:.2f}, "
        f"Min: {min_val:.2f}, Max: {max_val:.2f} | "
        f"RMSE: {rmse:.2f} | "
        f"AD_SCORE: {ad_score:.1f} | "
        f"Trend: {trend}"
    )

    threshold = config['anomaly']['score_threshold']
    if ad_score <= threshold:
        desc += " | ANOMALY DETECTED"

    # Truncate to 1000 characters (VARCHAR(1000) in DB)
    if len(desc) > 1000:
        desc = desc[:1000]

    return desc
