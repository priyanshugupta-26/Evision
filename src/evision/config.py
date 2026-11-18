"""
Configuration and constants for EVision: Intelligent EV Charging Demand Forecasting System.
"""

from pathlib import Path
import os

# Base directory paths (relative & portable across OS)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file paths
RAW_VOLUME_CSV = RAW_DATA_DIR / "volume.csv"
PROCESSED_VOLUME_PARQUET = PROCESSED_DATA_DIR / "volume.parquet"
RAW_WEATHER_CSV = RAW_DATA_DIR / "weather_central.csv"
PROCESSED_WEATHER_PARQUET = PROCESSED_DATA_DIR / "weather_central.parquet"
RAW_INF_CSV = RAW_DATA_DIR / "inf.csv"
PROCESSED_ZONE_METADATA_CSV = PROCESSED_DATA_DIR / "zone_metadata.csv"

# Official UrbanEV GitHub Remote URLs
URBANEV_RAW_VOLUME_URL = (
    "https://raw.githubusercontent.com/IntelligentSystemsLab/UrbanEV/main/data/volume.csv"
)
URBANEV_RAW_WEATHER_URL = (
    "https://raw.githubusercontent.com/IntelligentSystemsLab/UrbanEV/main/data/weather_central.csv"
)
URBANEV_RAW_INF_URL = (
    "https://raw.githubusercontent.com/IntelligentSystemsLab/UrbanEV/main/data/inf.csv"
)
URBANEV_REPO_URL = "https://github.com/IntelligentSystemsLab/UrbanEV"
URBANEV_PAPER_URL = "https://www.nature.com/articles/s41597-025-04874-4"
URBANEV_DRYAD_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.np5hqc04z"

# Academic & Data Limitations Notice
DATA_LIMITATION_NOTICE = (
    "Data Attribution & Engineering Note: According to the official UrbanEV documentation, "
    "charging volume (kWh) in 'volume.csv' is estimated based on charging pile rated power "
    "and charging session duration. Actual metered power utility consumption may differ due to "
    "battery State-of-Charge (SoC) charging taper curves, ambient temperature, and conversion losses. "
    "Demand alerts indicate statistically abnormal charging demand rather than physical substation overload."
)

# Time Series & Forecasting Constants
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

AVAILABLE_HORIZONS = [6, 12, 24, 48]
DEFAULT_HORIZON = 24

CORE_MODELS = [
    "Naive (Persistence)",
    "Seasonal Naive (24h)",
    "ARIMA",
    "SARIMAX",
    "XGBoost Regressor",
]

# High Demand Warning Thresholds
HIGH_DEMAND_PERCENTILE_ALERT = 90.0
CRITICAL_DEMAND_PERCENTILE_ALERT = 95.0

# UI Theme Color Palette (Dark Navy & Electric Blue)
THEME_COLORS = {
    "bg_dark": "#070D1E",
    "bg_card": "#0F1C3F",
    "bg_card_hover": "#17285C",
    "electric_blue": "#00D2FF",
    "neon_cyan": "#00F5D4",
    "accent_blue": "#3A86FF",
    "accent_purple": "#7000FF",
    "accent_pink": "#FF007F",
    "warning_yellow": "#FFBE0B",
    "danger_red": "#FF4D6D",
    "success_green": "#06D6A0",
    "text_main": "#E2E8F0",
    "text_muted": "#8E9EB5",
    "grid_color": "rgba(255, 255, 255, 0.08)",
}
