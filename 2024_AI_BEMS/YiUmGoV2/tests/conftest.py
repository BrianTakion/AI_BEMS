"""Shared pytest fixtures for YiUmGoV2 tests."""

import os
import json

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def config():
    """Load _config.json from the project root."""
    config_path = os.path.join(SCRIPT_DIR, "_config.json")
    with open(config_path, "r") as f:
        return json.load(f)


@pytest.fixture
def synthetic_window_df(config):
    """Create a synthetic 176-hour window DataFrame for testing preprocessing.

    Generates 15-min interval data with a realistic daily sinusoidal pattern
    plus random noise, covering the full fetch_window_hours (176h).
    """
    fetch_hours = config["data"]["fetch_window_hours"]
    sampling_min = config["data"]["sampling_minutes"]
    points_per_hour = 60 // sampling_min
    total_points = fetch_hours * points_per_hour  # 176 * 4 = 704

    start = pd.Timestamp("2026-01-05 00:00:00")
    index = pd.date_range(start=start, periods=total_points, freq=f"{sampling_min}min")

    rng = np.random.default_rng(42)
    hours = np.arange(total_points) / points_per_hour
    # Daily sinusoidal pattern (peak at 14:00) + baseline + noise
    values = 50 + 30 * np.sin(2 * np.pi * (hours - 6) / 24) + rng.normal(0, 3, total_points)
    values = np.clip(values, 0, None)  # no negatives

    df = pd.DataFrame({"value": values}, index=index)
    df.index.name = None
    return df
