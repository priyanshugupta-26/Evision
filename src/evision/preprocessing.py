"""
Preprocessing & Data Quality Assurance Module for EVision.
Provides zone extraction, 'All Zones' aggregation, chronological dataset splitting,
and exploratory statistical summaries.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from evision.config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)

logger = logging.getLogger("EVision.Preprocessing")


def get_zone_list(df: pd.DataFrame) -> List[str]:
    """Returns sorted list of available zone identifier columns (excluding 'time')."""
    return sorted([col for col in df.columns if col != "time"])


def get_series_for_zone(df: pd.DataFrame, zone_name: str) -> pd.DataFrame:
    """
    Extracts a single univariate time series for a specified zone or 'All Zones'.
    
    If zone_name == 'All Zones' (or 'all' / 'total'), sums all zone columns at each timestamp.
    Returns a DataFrame with columns: ['time', 'demand'].
    """
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a 'time' column.")

    df_out = pd.DataFrame()
    df_out["time"] = pd.to_datetime(df["time"])

    if zone_name.lower() in ["all zones", "all", "total", "combined"]:
        zone_cols = [c for c in df.columns if c != "time"]
        df_out["demand"] = df[zone_cols].sum(axis=1)
    else:
        if zone_name not in df.columns:
            raise KeyError(f"Selected zone '{zone_name}' is not in the dataset.")
        df_out["demand"] = pd.to_numeric(df[zone_name], errors="coerce").fillna(0.0)

    # Ensure clean numeric demand and sorted index
    df_out["demand"] = df_out["demand"].astype(float)
    df_out = df_out.sort_values(by="time").reset_index(drop=True)
    return df_out


def compute_dataset_summary(
    df: pd.DataFrame,
    zone_meta: Optional[pd.DataFrame] = None,
) -> Dict[str, Union[int, float, str, pd.DataFrame]]:
    """
    Computes comprehensive exploratory data statistics across all zones.
    """
    zone_cols = [c for c in df.columns if c != "time"]
    total_obs = len(df)
    n_zones = len(zone_cols)
    min_date = df["time"].min()
    max_date = df["time"].max()

    # Per-zone metrics
    zone_means = df[zone_cols].mean()
    zone_maxs = df[zone_cols].max()
    zone_mins = df[zone_cols].min()
    zone_stds = df[zone_cols].std()
    zone_totals = df[zone_cols].sum()

    # Global aggregate across all zone observations
    all_values = df[zone_cols].values.flatten()
    mean_val = float(np.mean(all_values))
    max_val = float(np.max(all_values))
    min_val = float(np.min(all_values))
    std_val = float(np.std(all_values))

    # All-zones aggregate time-series stats (hourly total demand)
    hourly_total = df[zone_cols].sum(axis=1)
    hourly_total_mean = float(hourly_total.mean())
    hourly_total_max = float(hourly_total.max())
    hourly_total_min = float(hourly_total.min())
    hourly_total_std = float(hourly_total.std())

    # Top 10 and Bottom 10 zones by mean demand
    top_10_zones = zone_means.sort_values(ascending=False).head(10).reset_index()
    top_10_zones.columns = ["zone_id", "mean_hourly_kwh"]

    bottom_10_zones = zone_means.sort_values(ascending=True).head(10).reset_index()
    bottom_10_zones.columns = ["zone_id", "mean_hourly_kwh"]

    # Zone ranking dataframe
    zone_ranking = pd.DataFrame({
        "zone_id": zone_cols,
        "mean_demand_kwh": zone_means.values,
        "max_demand_kwh": zone_maxs.values,
        "min_demand_kwh": zone_mins.values,
        "std_demand_kwh": zone_stds.values,
        "total_volume_kwh": zone_totals.values,
    })

    if zone_meta is not None and not zone_meta.empty:
        zone_ranking["zone_id"] = zone_ranking["zone_id"].astype(str)
        zone_meta_str = zone_meta.copy()
        zone_meta_str["zone_id"] = zone_meta_str["zone_id"].astype(str)
        zone_ranking = zone_ranking.merge(zone_meta_str, on="zone_id", how="left")

    zone_ranking = zone_ranking.sort_values(by="mean_demand_kwh", ascending=False).reset_index(drop=True)

    return {
        "total_observations": total_obs,
        "number_of_zones": n_zones,
        "date_min": min_date,
        "date_max": max_date,
        "date_range_str": f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}",
        "avg_hourly_zone_demand": mean_val,
        "max_zone_demand": max_val,
        "min_zone_demand": min_val,
        "std_zone_demand": std_val,
        "avg_hourly_citywide_demand": hourly_total_mean,
        "max_citywide_demand": hourly_total_max,
        "min_citywide_demand": hourly_total_min,
        "std_citywide_demand": hourly_total_std,
        "top_10_zones": top_10_zones,
        "bottom_10_zones": bottom_10_zones,
        "zone_ranking": zone_ranking,
    }


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a time-series dataset into train, validation, and test sets
    strictly chronologically without any random shuffling.
    
    Default split: 70% Train, 15% Validation, 15% Test.
    """
    total_len = len(df)
    if total_len < 30:
        raise ValueError(f"Series length ({total_len}) is too short for 3-way chronological split.")

    # Validate split ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not (0.99 <= ratio_sum <= 1.01):
        raise ValueError(f"Split ratios must sum to 1.0 (got {ratio_sum})")

    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    logger.info(
        f"Chronological split completed: "
        f"Train={len(train_df)} ({train_df['time'].min()} to {train_df['time'].max()}), "
        f"Val={len(val_df)} ({val_df['time'].min()} to {val_df['time'].max()}), "
        f"Test={len(test_df)} ({test_df['time'].min()} to {test_df['time'].max()})"
    )

    return train_df, val_df, test_df
