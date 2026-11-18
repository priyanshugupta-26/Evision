"""
Dataset Validation Script for EVision.
Performs data quality checks, schema verification, temporal ordering tests,
and reports dataset integrity.
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evision.data_loader import load_volume_data, load_weather_data, load_zone_metadata
from evision.preprocessing import compute_dataset_summary, get_zone_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EVision.ValidateData")


def validate_dataset():
    logger.info("Starting EVision dataset validation...")

    # 1. Volume dataset check
    df_vol, stats = load_volume_data()
    logger.info("Volume dataset loaded successfully.")
    print("\n--- Volume Dataset Validation ---")
    print(f"Total Rows (Hours): {len(df_vol)}")
    print(f"Total Columns: {len(df_vol.columns)}")
    print(f"Total Zones: {stats['zone_count']}")
    print(f"Null Timestamps: {stats['null_timestamps']}")
    print(f"Duplicate Timestamps: {stats['duplicate_timestamps']}")
    print(f"Imputed Missing Values: {stats['imputed_missing_values']}")
    print(f"Clipped Negative Values: {stats['negative_values_clipped']}")
    print(f"Start Timestamp: {stats['date_start']}")
    print(f"End Timestamp: {stats['date_end']}")

    assert len(df_vol) > 0, "Volume dataset is empty!"
    assert "time" in df_vol.columns, "Timestamp column 'time' missing!"
    assert df_vol["time"].is_monotonic_increasing, "Timestamps are not chronologically sorted!"
    assert stats["zone_count"] >= 10, f"Expected at least 10 zones, got {stats['zone_count']}"

    # 2. Weather dataset check
    df_weather = load_weather_data()
    if df_weather is not None:
        print("\n--- Weather Dataset Validation ---")
        print(f"Weather Rows: {len(df_weather)}")
        print(f"Weather Columns: {df_weather.columns.tolist()}")
        assert "time" in df_weather.columns, "Weather missing 'time' column"
    else:
        print("\n--- Weather Dataset: Optional / Not Loaded ---")

    # 3. Zone metadata check
    df_meta = load_zone_metadata()
    if df_meta is not None:
        print("\n--- Zone Metadata Validation ---")
        print(f"Metadata Zones: {len(df_meta)}")
        print(f"Metadata Columns: {df_meta.columns.tolist()}")
        print(f"Total Charging Stations: {df_meta['station_count'].sum()}")
        print(f"Total Charging Piles: {df_meta['pile_count'].sum()}")
    else:
        print("\n--- Zone Metadata: Optional / Not Loaded ---")

    # 4. Global statistics
    summary = compute_dataset_summary(df_vol, zone_meta=df_meta)
    print("\n--- Global Demand Summary ---")
    print(f"Average Zone Hourly Demand: {summary['avg_hourly_zone_demand']:.2f} kWh")
    print(f"Max Single-Zone Hourly Demand: {summary['max_zone_demand']:.2f} kWh")
    print(f"Average Citywide Hourly Demand: {summary['avg_hourly_citywide_demand']:.2f} kWh")
    print(f"Top 3 Highest-Demand Zones: {summary['top_10_zones']['zone_id'].iloc[:3].tolist()}")

    print("\n✅ All validation checks PASSED successfully!\n")


if __name__ == "__main__":
    validate_dataset()
