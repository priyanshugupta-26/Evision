"""
Unit tests for feature_engineering module.
Tests feature construction and verifies strict prevention of future information leakage.
"""

import numpy as np
import pandas as pd
import pytest

from evision.feature_engineering import generate_time_series_features, get_feature_names


def test_feature_generation_columns():
    times = pd.date_range("2022-09-01", periods=250, freq="h")
    df = pd.DataFrame({"time": times, "demand": np.random.uniform(10, 50, size=250)})

    feat_df = generate_time_series_features(df, drop_na=True)
    expected_cols = get_feature_names()

    for col in expected_cols:
        assert col in feat_df.columns, f"Missing feature: {col}"

    assert len(feat_df) == 250 - 168  # 168 rows dropped for max lag
    assert not feat_df[expected_cols].isna().any().any()


def test_cyclical_features_bounds():
    times = pd.date_range("2022-09-01", periods=100, freq="h")
    df = pd.DataFrame({"time": times, "demand": np.ones(100)})
    feat_df = generate_time_series_features(df, drop_na=False)

    for col in ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]:
        vals = feat_df[col].values
        assert vals.min() >= -1.0 - 1e-6
        assert vals.max() <= 1.0 + 1e-6


def test_strict_anti_leakage():
    """
    Leakage Test: Modifying target y_t MUST NOT change features X_t.
    All features at time t must depend exclusively on history up to t-1.
    """
    times = pd.date_range("2022-09-01", periods=200, freq="h")
    base_demand = np.random.uniform(20, 60, size=200)

    df1 = pd.DataFrame({"time": times, "demand": base_demand.copy()})
    feat1 = generate_time_series_features(df1, drop_na=False)

    # Change value at index 180 by +1000
    df2 = pd.DataFrame({"time": times, "demand": base_demand.copy()})
    df2.loc[180, "demand"] += 1000.0
    feat2 = generate_time_series_features(df2, drop_na=False)

    feature_cols = get_feature_names()

    # Features AT index 180 must be IDENTICAL between df1 and df2!
    row1 = feat1.loc[180, feature_cols].values
    row2 = feat2.loc[180, feature_cols].values

    assert np.allclose(row1, row2), "LEAKAGE DETECTED: Feature vector at index t was affected by target y_t!"

    # Features at index 181 SHOULD change (because index 180 is now in its past)
    row1_next = feat1.loc[181, ["lag_1"]].values
    row2_next = feat2.loc[181, ["lag_1"]].values
    assert not np.allclose(row1_next, row2_next)
