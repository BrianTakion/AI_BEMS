"""
data_preprocessing.py  --  Peak Prediction Feature Engineering

Transforms raw 15-min sensor data into daily feature vectors for peak
power and peak time prediction.

Pipeline:
  1. Resample raw 15-min data to 1-hour means
  2. Extract daily peaks (power value + 15-min slot index) as targets
  3. Build ~57 hybrid features per day from 14-day hourly history

Feature categories (~57 total):
  - Temporal (7): weekday, month, is_holiday, sin/cos encodings
  - Previous-day peaks (4): peak power/slot from prior 1-3 days
  - Same-weekday lag (2): peak power/slot from same weekday last week
  - Rolling peak stats 7d (5): mean, std, max, min peak; mean peak slot
  - Rolling peak quantiles 7d (2): 75th, 90th percentile
  - Peak time distribution 7d (4): slot std, slot median, most common period, same weekday slot
  - Rolling peak stats 14d (4): mean, std, max peak; trend (slope)
  - Rolling peak quantiles 14d (2): 75th, 90th percentile
  - Daily load shape (4): prev day mean, std, min, max/min ratio
  - Compressed profile prev day (4): morning/afternoon/evening/night means
  - Load ramp features (2): morning ramp, peak-to-mean ratio
  - Compressed profile prev week same day (4): morning/afternoon/evening/night means
  - Same-day partial (4): today_max_so_far, today_max_slot_so_far, today_mean_so_far, hours_elapsed
  - Trend (2): peak_trend_7d slope, weekday/weekend peak ratio
  - Day-type interaction (3): weekday x mean peak, prev day holiday, consecutive workdays
"""

import numpy as np
import pandas as pd
import holidays
import datetime
from collections import Counter


def _get_korean_holidays(year_range):
    """Build a set of Korean holiday dates including extended major holidays."""
    kr_holidays = holidays.KR(years=year_range)
    holiday_set = set(kr_holidays.keys())

    # 설날/추석: add day before and after
    for date in list(kr_holidays.keys()):
        name = kr_holidays[date]
        if name in ["설날", "추석"]:
            holiday_set.add(date - datetime.timedelta(days=1))
            holiday_set.add(date + datetime.timedelta(days=1))

    # 대체공휴일: if holiday falls on weekend, next weekday is substitute
    for date in list(kr_holidays.keys()):
        if date.weekday() in (5, 6):  # Saturday or Sunday
            replacement = date + datetime.timedelta(days=1)
            while replacement.weekday() in (5, 6) or replacement in kr_holidays:
                replacement += datetime.timedelta(days=1)
            holiday_set.add(replacement)

    return holiday_set


def _extract_daily_peaks(df_15min):
    """Extract daily peak power and peak slot from 15-min data.

    Args:
        df_15min: DataFrame with DatetimeIndex and column 'value', at 15-min freq.

    Returns:
        DataFrame indexed by date with columns:
          - peak_power: max value of the day
          - peak_slot: 15-min slot index (0-95) when peak occurred
    """
    daily_records = []
    for date, group in df_15min.groupby(df_15min.index.date):
        if group.empty:
            continue
        peak_idx = group["value"].idxmax()
        peak_power = group["value"].max()
        # slot index: (hour * 4) + (minute // 15)
        peak_slot = peak_idx.hour * 4 + peak_idx.minute // 15
        daily_records.append({
            "date": pd.Timestamp(date),
            "peak_power": peak_power,
            "peak_slot": peak_slot,
        })
    if not daily_records:
        return pd.DataFrame(columns=["peak_power", "peak_slot"])
    result = pd.DataFrame(daily_records).set_index("date")
    return result


def _resample_to_hourly(df_15min):
    """Resample 15-min data to hourly means.

    Args:
        df_15min: DataFrame with DatetimeIndex and column 'value'.

    Returns:
        DataFrame with hourly DatetimeIndex and column 'value'.
    """
    hourly = df_15min[["value"]].resample("1h").mean()
    return hourly


def _build_compressed_profile(hourly_df, target_date):
    """Build 4-period compressed profile for a given date from hourly data.

    Periods: night (0-6h), morning (6-12h), afternoon (12-18h), evening (18-24h).

    Args:
        hourly_df: DataFrame with hourly DatetimeIndex and column 'value'.
        target_date: date to extract profile for.

    Returns:
        Dict with keys: night_mean, morning_mean, afternoon_mean, evening_mean.
        Returns zeros if no data available for that date.
    """
    day_start = pd.Timestamp(target_date)
    day_data = hourly_df.loc[day_start:day_start + pd.Timedelta(hours=23)]

    if day_data.empty:
        return {"night_mean": 0.0, "morning_mean": 0.0,
                "afternoon_mean": 0.0, "evening_mean": 0.0}

    hours = day_data.index.hour
    result = {}
    for name, (h_start, h_end) in [("night", (0, 6)), ("morning", (6, 12)),
                                     ("afternoon", (12, 18)), ("evening", (18, 24))]:
        mask = (hours >= h_start) & (hours < h_end)
        subset = day_data.loc[mask, "value"]
        result[f"{name}_mean"] = subset.mean() if len(subset) > 0 else 0.0

    return result


def preprocess_for_training(df_raw, config):
    """Build training dataset: daily feature vectors + targets from raw 15-min data.

    Args:
        df_raw: DataFrame with columns ['colec_dt', 'colec_val'] at 15-min intervals.
        config: Configuration dict.

    Returns:
        Tuple (X_df, y_power, y_slot):
          - X_df: DataFrame of ~40 features per day
          - y_power: Series of daily peak power values
          - y_slot: Series of daily peak slot indices (0-95)
    """
    min_history_days = config["training"]["min_history_days"]  # 14

    # Build a value-indexed time series
    df = pd.DataFrame({
        "value": df_raw["colec_val"].values,
    }, index=pd.to_datetime(df_raw["colec_dt"]))
    df.index = df.index.round("15min")
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("15min").asfreq()
    df["value"] = df["value"].interpolate(method="time")
    df["value"] = df["value"].ffill().bfill()

    # Extract daily peaks from 15-min data
    daily_peaks = _extract_daily_peaks(df)
    if len(daily_peaks) < min_history_days + 1:
        raise ValueError(
            f"Need at least {min_history_days + 1} days of data, "
            f"got {len(daily_peaks)}"
        )

    # Resample to hourly for feature construction
    hourly = _resample_to_hourly(df)
    hourly["value"] = hourly["value"].interpolate(method="time").ffill().bfill()

    # Build features for each day (skip first min_history_days for lag availability)
    all_dates = daily_peaks.index.sort_values()
    year_range = range(all_dates[0].year, all_dates[-1].year + 1)
    holiday_set = _get_korean_holidays(year_range)

    # Multi-window training: for each day, create samples at 8 different
    # hours_elapsed values to simulate predictions at different times of day.
    # All data is real — only the same-day feature cutoff varies.
    TRAIN_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

    feature_rows = []
    target_power = []
    target_slot = []

    for i in range(min_history_days, len(all_dates)):
        date = all_dates[i]
        for hours_elapsed in TRAIN_HOURS:
            features = _build_features_for_date(
                date, daily_peaks, hourly, holiday_set,
                hours_elapsed=hours_elapsed, today_raw_15min=df
            )
            if features is not None:
                feature_rows.append(features)
                target_power.append(daily_peaks.loc[date, "peak_power"])
                target_slot.append(daily_peaks.loc[date, "peak_slot"])

    X_df = pd.DataFrame(feature_rows)
    y_power = pd.Series(target_power, name="peak_power")
    y_slot = pd.Series(target_slot, name="peak_slot")

    return X_df, y_power, y_slot


def preprocess_for_inference(df_raw, config):
    """Build feature vector for today's peak prediction from raw 15-min data.

    Args:
        df_raw: DataFrame with columns ['colec_dt', 'colec_val'], last 14+ days.
        config: Configuration dict.

    Returns:
        DataFrame with a single row of ~40 features for today's prediction.
    """
    # Build a value-indexed time series
    df = pd.DataFrame({
        "value": df_raw["colec_val"].values,
    }, index=pd.to_datetime(df_raw["colec_dt"]))
    df.index = df.index.round("15min")
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("15min").asfreq()
    df["value"] = df["value"].ffill().bfill()

    # Extract daily peaks from 15-min data
    daily_peaks = _extract_daily_peaks(df)

    # Resample to hourly
    hourly = _resample_to_hourly(df)
    hourly["value"] = hourly["value"].ffill().bfill()

    # Today = the last date in the data
    now = df.index[-1]
    today = pd.Timestamp(now.date())

    # hours_elapsed = how many full hours of today's data we have
    today_start = today
    today_data = df.loc[today_start:now]
    hours_elapsed = max(0, (now - today_start).total_seconds() / 3600)

    all_dates = daily_peaks.index.sort_values()
    year_range = range(all_dates[0].year, all_dates[-1].year + 1)
    holiday_set = _get_korean_holidays(year_range)

    features = _build_features_for_date(
        today, daily_peaks, hourly, holiday_set,
        hours_elapsed=hours_elapsed, today_raw_15min=df
    )

    if features is None:
        raise ValueError("Could not build features for today's prediction")

    return pd.DataFrame([features])


def _build_features_for_date(date, daily_peaks, hourly, holiday_set,
                              hours_elapsed=24, today_raw_15min=None):
    """Build the ~40 hybrid feature vector for a single date.

    Args:
        date: pd.Timestamp of the target date.
        daily_peaks: DataFrame indexed by date with columns peak_power, peak_slot.
        hourly: DataFrame with hourly DatetimeIndex and column 'value'.
        holiday_set: Set of holiday dates.
        hours_elapsed: Hours of today's data available (24 for training, partial for inference).
        today_raw_15min: If provided, raw 15-min data for computing same-day partial features.

    Returns:
        Dict of feature name -> value, or None if insufficient history.
    """
    features = {}
    d = date

    # =====================================================================
    # 1. Temporal features (7)
    # =====================================================================
    features["weekday"] = d.weekday()
    features["month"] = d.month
    features["is_holiday"] = 1 if d.date() in holiday_set or d.weekday() >= 5 else 0
    features["sin_month"] = np.sin(2 * np.pi * d.month / 12)
    features["cos_month"] = np.cos(2 * np.pi * d.month / 12)
    features["sin_weekday"] = np.sin(2 * np.pi * d.weekday() / 7)
    features["cos_weekday"] = np.cos(2 * np.pi * d.weekday() / 7)

    # =====================================================================
    # 2. Previous-day peaks (4)
    # =====================================================================
    for lag in [1, 2, 3]:
        lag_date = d - pd.Timedelta(days=lag)
        if lag_date in daily_peaks.index:
            features[f"prev_{lag}d_peak_power"] = daily_peaks.loc[lag_date, "peak_power"]
            if lag == 1:
                features["prev_1d_peak_slot"] = daily_peaks.loc[lag_date, "peak_slot"]
        else:
            features[f"prev_{lag}d_peak_power"] = 0.0
            if lag == 1:
                features["prev_1d_peak_slot"] = 0.0

    # =====================================================================
    # 3. Same-weekday lag (2)
    # =====================================================================
    same_weekday_date = d - pd.Timedelta(days=7)
    if same_weekday_date in daily_peaks.index:
        features["prev_week_same_day_peak_power"] = daily_peaks.loc[same_weekday_date, "peak_power"]
        features["prev_week_same_day_peak_slot"] = daily_peaks.loc[same_weekday_date, "peak_slot"]
    else:
        features["prev_week_same_day_peak_power"] = 0.0
        features["prev_week_same_day_peak_slot"] = 0.0

    # =====================================================================
    # 4. Rolling peak stats - 7 day (5)
    # =====================================================================
    past_7d = []
    past_7d_slots = []
    for lag in range(1, 8):
        lag_date = d - pd.Timedelta(days=lag)
        if lag_date in daily_peaks.index:
            past_7d.append(daily_peaks.loc[lag_date, "peak_power"])
            past_7d_slots.append(daily_peaks.loc[lag_date, "peak_slot"])

    if len(past_7d) >= 3:  # Need at least 3 days for meaningful stats
        features["mean_peak_7d"] = np.mean(past_7d)
        features["std_peak_7d"] = np.std(past_7d)
        features["max_peak_7d"] = np.max(past_7d)
        features["min_peak_7d"] = np.min(past_7d)
        features["mean_peak_slot_7d"] = np.mean(past_7d_slots)
    else:
        return None  # Insufficient history

    # Rolling peak quantiles - 7 day (2)
    features["peak_75pct_7d"] = np.percentile(past_7d, 75)
    features["peak_90pct_7d"] = np.percentile(past_7d, 90)

    # Peak time distribution features (4)
    features["peak_slot_std_7d"] = np.std(past_7d_slots) if len(past_7d_slots) >= 3 else 0.0
    features["peak_slot_median_7d"] = np.median(past_7d_slots)

    # Most common peak period (0=night 0-6, 1=morning 6-12, 2=afternoon 12-18, 3=evening 18-24)
    slot_periods = [s // 24 for s in past_7d_slots]  # 96 slots / 4 periods = 24 slots each
    if slot_periods:
        features["most_common_peak_period"] = Counter(slot_periods).most_common(1)[0][0]
    else:
        features["most_common_peak_period"] = 2  # default afternoon

    # Same weekday peak slot over last 2 weeks
    same_wd_slots = []
    for w in [7, 14]:
        wd_date = d - pd.Timedelta(days=w)
        if wd_date in daily_peaks.index:
            same_wd_slots.append(daily_peaks.loc[wd_date, "peak_slot"])
    features["same_weekday_peak_slot_2w"] = np.mean(same_wd_slots) if same_wd_slots else features.get("mean_peak_slot_7d", 0.0)

    # =====================================================================
    # 5. Rolling peak stats - 14 day (4)
    # =====================================================================
    past_14d = list(past_7d)  # start with 7d data
    for lag in range(8, 15):
        lag_date = d - pd.Timedelta(days=lag)
        if lag_date in daily_peaks.index:
            past_14d.append(daily_peaks.loc[lag_date, "peak_power"])

    features["mean_peak_14d"] = np.mean(past_14d)
    features["std_peak_14d"] = np.std(past_14d)
    features["max_peak_14d"] = np.max(past_14d)

    # Linear trend over 14 days (slope of peak power)
    if len(past_14d) >= 7:
        x = np.arange(len(past_14d))
        slope = np.polyfit(x, past_14d[::-1], 1)[0]  # reversed so oldest first
        features["trend_peak_14d"] = slope
    else:
        features["trend_peak_14d"] = 0.0

    # Rolling peak quantiles - 14 day (2)
    features["peak_75pct_14d"] = np.percentile(past_14d, 75)
    features["peak_90pct_14d"] = np.percentile(past_14d, 90)

    # =====================================================================
    # 6. Daily load shape - prev day (4)
    # =====================================================================
    prev_day = d - pd.Timedelta(days=1)
    prev_day_start = pd.Timestamp(prev_day)
    prev_day_end = prev_day_start + pd.Timedelta(hours=23)
    prev_day_hourly = hourly.loc[prev_day_start:prev_day_end, "value"]

    if len(prev_day_hourly) > 0:
        features["prev_day_mean"] = prev_day_hourly.mean()
        features["prev_day_std"] = prev_day_hourly.std() if len(prev_day_hourly) > 1 else 0.0
        features["prev_day_min"] = prev_day_hourly.min()
        min_val = prev_day_hourly.min()
        features["prev_day_max_min_ratio"] = (
            prev_day_hourly.max() / min_val if min_val > 0.01 else 0.0
        )
    else:
        features["prev_day_mean"] = 0.0
        features["prev_day_std"] = 0.0
        features["prev_day_min"] = 0.0
        features["prev_day_max_min_ratio"] = 0.0

    # =====================================================================
    # 7. Compressed profile - prev day (4)
    # =====================================================================
    profile = _build_compressed_profile(hourly, prev_day)
    features["prev_day_night_mean"] = profile["night_mean"]
    features["prev_day_morning_mean"] = profile["morning_mean"]
    features["prev_day_afternoon_mean"] = profile["afternoon_mean"]
    features["prev_day_evening_mean"] = profile["evening_mean"]

    # Load ramp features (2)
    features["prev_day_morning_ramp"] = profile["morning_mean"] - profile["night_mean"]
    prev_day_mean_val = features.get("prev_day_mean", 0.0)
    features["prev_day_peak_to_mean_ratio"] = (
        features["prev_1d_peak_power"] / prev_day_mean_val
        if prev_day_mean_val > 0.01 else 0.0
    )

    # =====================================================================
    # 8. Compressed profile - prev week same day (4)
    # =====================================================================
    pw_profile = _build_compressed_profile(hourly, same_weekday_date)
    features["pw_night_mean"] = pw_profile["night_mean"]
    features["pw_morning_mean"] = pw_profile["morning_mean"]
    features["pw_afternoon_mean"] = pw_profile["afternoon_mean"]
    features["pw_evening_mean"] = pw_profile["evening_mean"]

    # =====================================================================
    # 9. Same-day partial features (4)
    # =====================================================================
    features["hours_elapsed"] = hours_elapsed

    if hours_elapsed > 0 and today_raw_15min is not None:
        today_start = pd.Timestamp(d)
        now = today_start + pd.Timedelta(hours=hours_elapsed)
        today_data = today_raw_15min.loc[today_start:now, "value"]
        if len(today_data) > 0:
            features["today_max_so_far"] = today_data.max()
            max_idx = today_data.idxmax()
            features["today_max_slot_so_far"] = max_idx.hour * 4 + max_idx.minute // 15
            features["today_mean_so_far"] = today_data.mean()
        else:
            features["today_max_so_far"] = 0.0
            features["today_max_slot_so_far"] = 0.0
            features["today_mean_so_far"] = 0.0
    elif hours_elapsed > 0:
        # Training mode: use hourly data for partial features
        today_start = pd.Timestamp(d)
        today_end = today_start + pd.Timedelta(hours=min(hours_elapsed, 23))
        today_hourly = hourly.loc[today_start:today_end, "value"]
        if len(today_hourly) > 0:
            features["today_max_so_far"] = today_hourly.max()
            max_idx = today_hourly.idxmax()
            features["today_max_slot_so_far"] = max_idx.hour * 4
            features["today_mean_so_far"] = today_hourly.mean()
        else:
            features["today_max_so_far"] = 0.0
            features["today_max_slot_so_far"] = 0.0
            features["today_mean_so_far"] = 0.0
    else:
        features["today_max_so_far"] = 0.0
        features["today_max_slot_so_far"] = 0.0
        features["today_mean_so_far"] = 0.0

    # =====================================================================
    # 10. Trend features (2)
    # =====================================================================
    # 7-day slope (reuse past_7d, reversed to chronological order)
    if len(past_7d) >= 3:
        x = np.arange(len(past_7d))
        features["peak_trend_7d"] = np.polyfit(x, past_7d[::-1], 1)[0]
    else:
        features["peak_trend_7d"] = 0.0

    # Weekday vs weekend peak ratio
    weekday_peaks = []
    weekend_peaks = []
    for lag in range(1, 15):
        lag_date = d - pd.Timedelta(days=lag)
        if lag_date in daily_peaks.index:
            pp = daily_peaks.loc[lag_date, "peak_power"]
            if lag_date.weekday() < 5:
                weekday_peaks.append(pp)
            else:
                weekend_peaks.append(pp)

    if weekday_peaks and weekend_peaks and np.mean(weekend_peaks) > 0.01:
        features["weekday_weekend_peak_ratio"] = np.mean(weekday_peaks) / np.mean(weekend_peaks)
    else:
        features["weekday_weekend_peak_ratio"] = 1.0

    # =====================================================================
    # 11. Day-type interaction features (3)
    # =====================================================================
    features["is_weekday_x_mean_peak_7d"] = (1 if d.weekday() < 5 else 0) * features["mean_peak_7d"]

    prev_day_date = d - pd.Timedelta(days=1)
    features["prev_day_was_holiday"] = 1 if (prev_day_date.date() in holiday_set or prev_day_date.weekday() >= 5) else 0

    # Consecutive workdays before this date
    consec = 0
    for back in range(1, 15):
        check_date = d - pd.Timedelta(days=back)
        if check_date.date() in holiday_set or check_date.weekday() >= 5:
            break
        consec += 1
    features["consecutive_workdays"] = consec

    return features
