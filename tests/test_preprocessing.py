"""
Unit tests for preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest

from evision.preprocessing import (
    chronological_split,
    compute_dataset_summary,
    get_series_for_zone,
    get_zone_list,
)


@pytest.fixture
def sample_df():
    times = pd.date_range("2022-09-01", periods=100, freq="h")
    return pd.DataFrame({
        "time": times,
        "102": np.random.uniform(10, 50, size=100),
        "104": np.random.uniform(5, 25, size=100),
        "105": np.random.uniform(20, 80, size=100),
    })


def test_get_zone_list(sample_df):
    zones = get_zone_list(sample_df)
    assert zones == ["102", "104", "105"]


def test_get_series_for_individual_zone(sample_df):
    s = get_series_for_zone(sample_df, "102")
    assert list(s.columns) == ["time", "demand"]
    assert len(s) == 100
    assert np.allclose(s["demand"].values, sample_df["102"].values)


def test_get_series_for_all_zones_combined(sample_df):
    s_all = get_series_for_zone(sample_df, "All Zones")
    expected_sum = sample_df["102"] + sample_df["104"] + sample_df["105"]
    assert np.allclose(s_all["demand"].values, expected_sum.values)


def test_chronological_split(sample_df):
    train_df, val_df, test_df = chronological_split(sample_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Strict chronological guarantee: no overlap and strictly increasing
    assert train_df["time"].max() < val_df["time"].min()
    assert val_df["time"].max() < test_df["time"].min()


def test_compute_dataset_summary(sample_df):
    summary = compute_dataset_summary(sample_df)
    assert summary["total_observations"] == 100
    assert summary["number_of_zones"] == 3
    assert summary["avg_hourly_zone_demand"] > 0
    assert len(summary["top_10_zones"]) == 3
