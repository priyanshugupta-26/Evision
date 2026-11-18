"""
Data Acquisition Script for EVision.
Downloads the official UrbanEV benchmark dataset from GitHub and creates
optimized parquet and metadata artifacts for fast, reliable loading.
"""

import sys
import logging
from pathlib import Path
import urllib.request
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evision.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_VOLUME_CSV,
    PROCESSED_VOLUME_PARQUET,
    RAW_WEATHER_CSV,
    PROCESSED_WEATHER_PARQUET,
    RAW_INF_CSV,
    PROCESSED_ZONE_METADATA_CSV,
    URBANEV_RAW_VOLUME_URL,
    URBANEV_RAW_WEATHER_URL,
    URBANEV_RAW_INF_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EVision.Downloader")


def download_file_if_missing(url: str, destination: Path) -> Path:
    """Download a file from url to destination if it doesn't already exist."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        logger.info(f"File already exists: {destination} ({destination.stat().st_size / (1024*1024):.2f} MB)")
        return destination

    logger.info(f"Downloading from {url} to {destination}...")
    try:
        urllib.request.urlretrieve(url, destination)
        logger.info(f"Downloaded successfully: {destination} ({destination.stat().st_size / (1024*1024):.2f} MB)")
        return destination
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def process_and_cache_dataset():
    """Converts raw CSVs to optimized parquet files and generates zone metadata."""
    logger.info("Verifying and preparing dataset artifacts...")

    # 1. Volume dataset
    if not RAW_VOLUME_CSV.exists() or RAW_VOLUME_CSV.stat().st_size == 0:
        download_file_if_missing(URBANEV_RAW_VOLUME_URL, RAW_VOLUME_CSV)

    logger.info(f"Loading raw volume dataset from {RAW_VOLUME_CSV}...")
    df_vol = pd.read_csv(RAW_VOLUME_CSV)

    # Detect timestamp column
    time_col = None
    for col in df_vol.columns:
        if col.lower() in ["time", "datetime", "date", "timestamp"]:
            time_col = col
            break
    if not time_col:
        time_col = df_vol.columns[0]

    df_vol[time_col] = pd.to_datetime(df_vol[time_col])
    df_vol = df_vol.sort_values(by=time_col).reset_index(drop=True)
    df_vol = df_vol.rename(columns={time_col: "time"})

    # Save to parquet
    df_vol.to_parquet(PROCESSED_VOLUME_PARQUET, index=False, compression="snappy")
    logger.info(f"Saved optimized volume parquet: {PROCESSED_VOLUME_PARQUET} ({PROCESSED_VOLUME_PARQUET.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Weather dataset
    if not RAW_WEATHER_CSV.exists() or RAW_WEATHER_CSV.stat().st_size == 0:
        download_file_if_missing(URBANEV_RAW_WEATHER_URL, RAW_WEATHER_CSV)

    logger.info(f"Loading raw weather dataset from {RAW_WEATHER_CSV}...")
    df_weather = pd.read_csv(RAW_WEATHER_CSV)
    w_time_col = df_weather.columns[0]
    df_weather["time"] = pd.to_datetime(df_weather[w_time_col])
    df_weather = df_weather.sort_values(by="time").reset_index(drop=True)
    
    # Rename weather columns to descriptive names
    weather_rename = {
        "T": "temperature_c",
        "P0": "pressure_station_mmhg",
        "P": "pressure_sealevel_mmhg",
        "U": "humidity_pct",
        "nRAIN": "rain_level",
        "Td": "dewpoint_c"
    }
    df_weather = df_weather.rename(columns=weather_rename)
    df_weather.to_parquet(PROCESSED_WEATHER_PARQUET, index=False, compression="snappy")
    logger.info(f"Saved optimized weather parquet: {PROCESSED_WEATHER_PARQUET} ({PROCESSED_WEATHER_PARQUET.stat().st_size / 1024:.1f} KB)")

    # 3. Zone infrastructure metadata
    if not RAW_INF_CSV.exists() or RAW_INF_CSV.stat().st_size == 0:
        download_file_if_missing(URBANEV_RAW_INF_URL, RAW_INF_CSV)

    logger.info(f"Processing infrastructure metadata from {RAW_INF_CSV}...")
    df_inf = pd.read_csv(RAW_INF_CSV)
    zone_meta = df_inf.groupby("TAZID").agg({
        "station_id": "count",
        "charge_count": "sum",
        "longitude": "mean",
        "latitude": "mean",
        "area": "first",
        "perimeter": "first"
    }).reset_index().rename(columns={
        "TAZID": "zone_id",
        "station_id": "station_count",
        "charge_count": "pile_count",
        "longitude": "centroid_lon",
        "latitude": "centroid_lat"
    })
    zone_meta["zone_id"] = zone_meta["zone_id"].astype(str)
    zone_meta.to_csv(PROCESSED_ZONE_METADATA_CSV, index=False)
    logger.info(f"Saved zone metadata: {PROCESSED_ZONE_METADATA_CSV} with {len(zone_meta)} zones.")

    print("\nDataset acquisition and processing complete.")
    print(f"Total time intervals: {len(df_vol)}")
    print(f"Total zones: {len(df_vol.columns) - 1}")
    print(f"Date range: {df_vol['time'].min()} to {df_vol['time'].max()}")


if __name__ == "__main__":
    process_and_cache_dataset()
