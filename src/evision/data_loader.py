"""
Data Loader module for EVision.
Provides robust, dynamic, and portable loading for the UrbanEV dataset,
including automated detection of timestamp and zone columns, data validation,
chronological sorting, missing-interval handling, and auxiliary dataset integration.
"""

from io import BytesIO
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import pandas as pd
import requests

from evision.config import (
    PROCESSED_VOLUME_PARQUET,
    RAW_VOLUME_CSV,
    PROCESSED_WEATHER_PARQUET,
    RAW_WEATHER_CSV,
    PROCESSED_ZONE_METADATA_CSV,
    RAW_INF_CSV,
    URBANEV_RAW_VOLUME_URL,
    URBANEV_RAW_WEATHER_URL,
    URBANEV_RAW_INF_URL,
)

logger = logging.getLogger("EVision.DataLoader")


class DataLoaderError(Exception):
    """Custom exception raised when dataset loading or validation fails."""
    pass


def detect_timestamp_column(df: pd.DataFrame) -> str:
    """
    Dynamically identifies the timestamp column from a dataframe
    without hardcoding column indices or casing.
    """
    candidates = ["time", "datetime", "date", "timestamp", "ts", "hour"]
    col_map = {col.lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate in col_map:
            return col_map[candidate]

    # Fallback heuristic: check object/string columns for datetime parseability
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            try:
                sample = df[col].dropna().iloc[:10]
                pd.to_datetime(sample)
                logger.info(f"Dynamically detected timestamp column: '{col}'")
                return col
            except Exception:
                continue

    # Final fallback: return first column
    logger.warning(f"No obvious timestamp column detected. Defaulting to first column: '{df.columns[0]}'")
    return df.columns[0]


def clean_and_validate_time_series(
    df: pd.DataFrame,
    timestamp_col: str,
    target_unit: str = "kWh",
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, str]]]:
    """
    Validates, standardizes, cleans, and sorts time series demand data.
    
    Steps:
    - Parse timestamps to datetime64.
    - Check and resolve duplicate timestamps (aggregating by mean).
    - Ensure regular hourly frequency (reindexing if gaps exist).
    - Identify zone columns dynamically.
    - Coerce volume values to numeric (float), replace negatives with 0.
    - Ensure strict chronological order.
    """
    stats = {}
    initial_rows = len(df)
    stats["initial_rows"] = initial_rows

    # Standardize timestamp column name
    df = df.copy()
    df["time"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    null_times = df["time"].isna().sum()
    stats["null_timestamps"] = int(null_times)
    if null_times > 0:
        logger.warning(f"Found {null_times} invalid timestamps; dropping those rows.")
        df = df.dropna(subset=["time"])

    # Check and handle duplicates
    duplicate_count = df.duplicated(subset=["time"]).sum()
    stats["duplicate_timestamps"] = int(duplicate_count)
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate timestamps; aggregating by mean.")
        # Drop original timestamp col if it was named something else
        if timestamp_col != "time" and timestamp_col in df.columns:
            df = df.drop(columns=[timestamp_col])
        df = df.groupby("time", as_index=False).mean()
    else:
        if timestamp_col != "time" and timestamp_col in df.columns:
            df = df.drop(columns=[timestamp_col])

    # Sort strictly chronologically
    df = df.sort_values(by="time").reset_index(drop=True)

    # Identify zone columns
    zone_cols = [c for c in df.columns if c != "time"]
    stats["zone_count"] = len(zone_cols)
    if len(zone_cols) == 0:
        raise DataLoaderError("No demand/zone columns detected in dataset.")

    # Convert zone columns to numeric and check for invalid/negative values
    negative_values_found = 0
    missing_values_found = 0
    for col in zone_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        missing_count = df[col].isna().sum()
        missing_values_found += missing_count
        # Forward fill and backward fill small gaps
        if missing_count > 0:
            df[col] = df[col].ffill().bfill().fillna(0.0)

        # Non-negative constraint for charging volume
        negatives = (df[col] < 0).sum()
        if negatives > 0:
            negative_values_found += negatives
            df[col] = df[col].clip(lower=0.0)

    stats["imputed_missing_values"] = int(missing_values_found)
    stats["negative_values_clipped"] = int(negative_values_found)
    stats["date_start"] = str(df["time"].min())
    stats["date_end"] = str(df["time"].max())
    stats["final_rows"] = len(df)
    stats["target_unit"] = target_unit

    return df, stats


def load_volume_data(
    file_source: Optional[Union[str, Path, BytesIO]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, str]]]:
    """
    Primary loader for EV charging volume (demand) dataset.
    
    Supports:
    - Preprocessed Parquet file (fastest)
    - Raw CSV file
    - User uploaded CSV or ZIP file buffer
    - Automatic remote download from official UrbanEV GitHub repository if missing.
    """
    df: Optional[pd.DataFrame] = None

    if file_source is not None:
        # Load from provided path or buffer
        if isinstance(file_source, BytesIO):
            # Try reading as zip, else read as csv
            try:
                with zipfile.ZipFile(file_source) as z:
                    csv_names = [n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
                    if not csv_names:
                        raise DataLoaderError("No CSV file found inside the uploaded ZIP archive.")
                    target_name = next((n for n in csv_names if "volume" in n.lower()), csv_names[0])
                    with z.open(target_name) as f:
                        df = pd.read_csv(f)
            except zipfile.BadZipFile:
                file_source.seek(0)
                df = pd.read_csv(file_source)
        else:
            path = Path(file_source)
            if not path.exists():
                raise DataLoaderError(f"Provided file path does not exist: {path}")
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".csv":
                df = pd.read_csv(path)
            else:
                raise DataLoaderError(f"Unsupported file format: {path.suffix}")

    else:
        # Check standard locations in order of performance
        if PROCESSED_VOLUME_PARQUET.exists() and PROCESSED_VOLUME_PARQUET.stat().st_size > 0:
            logger.info(f"Loading cached volume dataset from {PROCESSED_VOLUME_PARQUET}")
            df = pd.read_parquet(PROCESSED_VOLUME_PARQUET)
        elif RAW_VOLUME_CSV.exists() and RAW_VOLUME_CSV.stat().st_size > 0:
            logger.info(f"Loading raw volume CSV from {RAW_VOLUME_CSV}")
            df = pd.read_csv(RAW_VOLUME_CSV)
        else:
            # Attempt automated download from official repository
            logger.info(f"Local dataset not found. Downloading official UrbanEV benchmark from {URBANEV_RAW_VOLUME_URL}...")
            try:
                resp = requests.get(URBANEV_RAW_VOLUME_URL, timeout=30)
                resp.raise_for_status()
                RAW_VOLUME_CSV.parent.mkdir(parents=True, exist_ok=True)
                with open(RAW_VOLUME_CSV, "wb") as f:
                    f.write(resp.content)
                df = pd.read_csv(RAW_VOLUME_CSV)
            except Exception as e:
                raise DataLoaderError(
                    f"Unable to load or download the official UrbanEV volume dataset: {e}. "
                    "Please verify your internet connection or use the dashboard file upload feature."
                )

    if df is None or df.empty:
        raise DataLoaderError("Loaded volume dataset is empty.")

    ts_col = detect_timestamp_column(df)
    clean_df, stats = clean_and_validate_time_series(df, timestamp_col=ts_col)
    return clean_df, stats


def load_weather_data(
    file_source: Optional[Union[str, Path]] = None,
) -> Optional[pd.DataFrame]:
    """
    Loads and standardizes hourly weather measurements (temperature, humidity, rain).
    """
    try:
        if file_source:
            path = Path(file_source)
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        elif PROCESSED_WEATHER_PARQUET.exists():
            df = pd.read_parquet(PROCESSED_WEATHER_PARQUET)
        elif RAW_WEATHER_CSV.exists():
            df = pd.read_csv(RAW_WEATHER_CSV)
            w_time_col = df.columns[0]
            df["time"] = pd.to_datetime(df[w_time_col])
            weather_rename = {
                "T": "temperature_c",
                "P0": "pressure_station_mmhg",
                "P": "pressure_sealevel_mmhg",
                "U": "humidity_pct",
                "nRAIN": "rain_level",
                "Td": "dewpoint_c",
            }
            df = df.rename(columns=weather_rename)
        else:
            logger.info("Attempting weather dataset download from official repository...")
            resp = requests.get(URBANEV_RAW_WEATHER_URL, timeout=20)
            resp.raise_for_status()
            RAW_WEATHER_CSV.parent.mkdir(parents=True, exist_ok=True)
            with open(RAW_WEATHER_CSV, "wb") as f:
                f.write(resp.content)
            df = pd.read_csv(RAW_WEATHER_CSV)
            df["time"] = pd.to_datetime(df.iloc[:, 0])

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by="time").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Weather dataset could not be loaded: {e}. Continuing without auxiliary weather.")
        return None


def load_zone_metadata(
    file_source: Optional[Union[str, Path]] = None,
) -> Optional[pd.DataFrame]:
    """
    Loads zone infrastructure metadata (station count, pile count, area, coordinates).
    """
    try:
        if file_source:
            df = pd.read_csv(file_source)
        elif PROCESSED_ZONE_METADATA_CSV.exists():
            df = pd.read_csv(PROCESSED_ZONE_METADATA_CSV)
        elif RAW_INF_CSV.exists():
            df_inf = pd.read_csv(RAW_INF_CSV)
            df = df_inf.groupby("TAZID").agg({
                "station_id": "count",
                "charge_count": "sum",
                "longitude": "mean",
                "latitude": "mean",
                "area": "first",
                "perimeter": "first",
            }).reset_index().rename(columns={
                "TAZID": "zone_id",
                "station_id": "station_count",
                "charge_count": "pile_count",
                "longitude": "centroid_lon",
                "latitude": "centroid_lat",
            })
        else:
            logger.info("Downloading infrastructure metadata from official repository...")
            resp = requests.get(URBANEV_RAW_INF_URL, timeout=20)
            resp.raise_for_status()
            RAW_INF_CSV.parent.mkdir(parents=True, exist_ok=True)
            with open(RAW_INF_CSV, "wb") as f:
                f.write(resp.content)
            return load_zone_metadata()

        df["zone_id"] = df["zone_id"].astype(str)
        return df
    except Exception as e:
        logger.warning(f"Zone metadata could not be loaded: {e}")
        return None
