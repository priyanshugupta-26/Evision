"""
Unit tests for data_loader module.
"""

from io import BytesIO
import numpy as np
import pandas as pd
import pytest

from evision.data_loader import (
    clean_and_validate_time_series,
    detect_timestamp_column,
    load_volume_data,
    load_weather_data,
    load_zone_metadata,
)


def test_detect_timestamp_column():
    df1 = pd.DataFrame({"TIME": ["2022-09-01 00:00:00"], "102": [45.0]})
    assert detect_timestamp_column(df1) == "TIME"

    df2 = pd.DataFrame({"datetime": ["2022-09-01 00:00:00"], "zone_1": [12.0]})
    assert detect_timestamp_column(df2) == "datetime"

    df3 = pd.DataFrame({"custom_date": ["2022-09-01 01:00:00"], "demand": [10.0]})
    assert detect_timestamp_column(df3) == "custom_date"


def test_clean_and_validate_time_series():
    raw_data = {
        "time": [
            "2022-09-01 02:00:00",
            "2022-09-01 00:00:00",  # unsorted
            "2022-09-01 01:00:00",
            "2022-09-01 01:00:00",  # duplicate
            "invalid_date",          # invalid
        ],
        "zone_A": [30.0, -5.0, 15.0, 25.0, 10.0],  # negative value
        "zone_B": [10.0, np.nan, 20.0, 20.0, 5.0],  # missing value
    }
    df = pd.DataFrame(raw_data)
    clean_df, stats = clean_and_validate_time_series(df, timestamp_col="time")

    # Chronologically sorted
    assert clean_df["time"].is_monotonic_increasing
    # Duplicates aggregated
    assert stats["duplicate_timestamps"] == 1
    # Invalid dropped
    assert stats["null_timestamps"] == 1
    # Non-negative clipped
    assert (clean_df["zone_A"] >= 0).all()
    # No NaNs remaining
    assert clean_df["zone_B"].isna().sum() == 0


def test_load_volume_data_local():
    df, stats = load_volume_data()
    assert len(df) == 4344
    assert stats["zone_count"] == 275
    assert "time" in df.columns
    assert df["time"].is_monotonic_increasing


def test_load_volume_from_bytes_io():
    sample_csv = "time,101,102\n2022-09-01 00:00:00,10.5,20.0\n2022-09-01 01:00:00,12.0,22.5\n"
    buf = BytesIO(sample_csv.encode("utf-8"))
    df, stats = load_volume_data(buf)
    assert len(df) == 2
    assert stats["zone_count"] == 2
    assert "101" in df.columns


def test_load_weather_and_metadata():
    weather = load_weather_data()
    if weather is not None:
        assert "time" in weather.columns
        assert len(weather) > 0

    meta = load_zone_metadata()
    if meta is not None:
        assert "zone_id" in meta.columns
        assert "station_count" in meta.columns
        assert len(meta) == 275
