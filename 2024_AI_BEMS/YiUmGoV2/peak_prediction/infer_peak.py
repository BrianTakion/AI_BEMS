"""
infer_peak.py  --  Peak Prediction Inference Utilities

Provides model loading, prediction, and result formatting for peak
power and peak time prediction.

Two models:
  - Power model: predicts daily peak power (kW)
  - Time model: predicts daily peak time slot (0-95, each = 15 min)

Output format: "207.33@13:15" (peak_power@peak_time)
"""

import os
import numpy as np
import lightgbm as lgb


def load_model(model_path):
    """Load a LightGBM Booster from a .txt file.

    Args:
        model_path: Path to the LightGBM model text file.

    Returns:
        lgb.Booster instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return lgb.Booster(model_file=model_path)


def predict_peak(model_power, model_time, X_df):
    """Predict peak power and peak time from feature DataFrame.

    Args:
        model_power: LightGBM Booster for peak power prediction.
        model_time: LightGBM Booster for peak time slot prediction.
        X_df: pandas DataFrame with feature columns (single row for inference).

    Returns:
        Tuple (peak_power, peak_slot):
          - peak_power: float, predicted peak power in kW
          - peak_slot: int, predicted 15-min slot index (0-95)
    """
    power_pred = model_power.predict(X_df)
    time_pred = model_time.predict(X_df)

    # Take the last prediction (most relevant for single-row inference)
    peak_power = float(power_pred[-1])
    peak_slot = int(np.clip(np.round(time_pred[-1]), 0, 95))

    # Ensure peak_power is non-negative
    peak_power = max(0.0, peak_power)

    return peak_power, peak_slot


def format_peak_result(peak_power, peak_slot):
    """Format peak prediction as "power@HH:MM" string.

    Args:
        peak_power: Predicted peak power (float).
        peak_slot: Predicted 15-min slot index 0-95.

    Returns:
        String in format "207.33@13:15".
    """
    total_minutes = peak_slot * 15
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{peak_power:.2f}@{hours:02d}:{minutes:02d}"
