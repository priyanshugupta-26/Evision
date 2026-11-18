"""
Feature Engineering Module for EVision.
Generates robust time-series features for Machine Learning (XGBoost Regressor):
- Autoregressive Lags (1, 2, 3, 24, 48, 168)
- Shifted Rolling Windows (Rolling Mean, Rolling Std without future leakage)
- Calendar & Diurnal indicators (Hour, Day of Week, Is Weekend, Month)
- Cyclical trigonometric encodings (Sine and Cosine transformations)
- Strict prevention of future information leakage
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("EVision.FeatureEngineering")


def generate_time_series_features(
    df_demand: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 24, 48, 168],
    rolling_windows: List[int] = [6, 24],
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Constructs leak-free feature matrix from a univariate demand time series.
    
    Parameters:
    - df_demand: DataFrame with 'time' and 'demand' columns.
    - lags: Lag orders to construct (default: 1, 2, 3, 24, 48, 168).
    - rolling_windows: Window sizes for rolling statistics.
    - drop_na: Whether to drop initial rows with NaN due to lags.
    
    Anti-Leakage Guarantee:
    - All rolling calculations use `shift(1)`, guaranteeing that the current
      target observation y_t is never included in features for time t.
    """
    df = df_demand.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by="time").reset_index(drop=True)

    # 1. Autoregressive Lag Features
    for lag in lags:
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    # 2. Shifted Rolling Window Statistics (Strictly Past Information)
    # Using shift(1) ensures the current observation at t is never part of the feature
    demand_past = df["demand"].shift(1)
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = demand_past.rolling(window=window, min_periods=window // 2).mean()
        df[f"rolling_std_{window}"] = demand_past.rolling(window=window, min_periods=window // 2).std()

    # 3. Calendar & Diurnal Features
    hour = df["time"].dt.hour
    dayofweek = df["time"].dt.day_of_week
    month = df["time"].dt.month

    df["hour"] = hour
    df["dayofweek"] = dayofweek
    df["is_weekend"] = dayofweek.isin([5, 6]).astype(int)
    df["month"] = month

    # 4. Cyclical Trigonometric Encodings (Smooth continuous circular variables)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    df["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)

    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    if drop_na:
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped_len = initial_len - len(df)
        logger.debug(f"Dropped {dropped_len} initial rows due to lag generation (max lag={max(lags)})")

    return df


def get_feature_names() -> List[str]:
    """Returns the standardized list of ML feature column names."""
    return [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "lag_48",
        "lag_168",
        "rolling_mean_6",
        "rolling_std_6",
        "rolling_mean_24",
        "rolling_std_24",
        "hour",
        "dayofweek",
        "is_weekend",
        "month",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
    ]


def prepare_train_val_test_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Constructs features across concatenated continuous series to avoid boundary gaps,
    then slices back along the original exact split timestamps.
    """
    combined = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    featured_df = generate_time_series_features(combined, drop_na=False)

    train_end_time = train_df["time"].max()
    val_end_time = val_df["time"].max()

    # Slice back cleanly
    feat_train = featured_df[featured_df["time"] <= train_end_time].dropna().reset_index(drop=True)
    feat_val = featured_df[(featured_df["time"] > train_end_time) & (featured_df["time"] <= val_end_time)].reset_index(drop=True)
    feat_test = featured_df[featured_df["time"] > val_end_time].reset_index(drop=True)

    return feat_train, feat_val, feat_test
