# UrbanEV Dataset Documentation

## Overview
This directory contains the preprocessed hourly zone-level data and metadata for the **EVision** system, sourced from the official **UrbanEV** benchmark dataset.

- **Location:** Shenzhen, China
- **Collection Window:** September 1, 2022 to February 28, 2023 (6 continuous months / 181 days / 4,344 hourly intervals)
- **Spatial Scope:** 275 administrative Traffic Analysis Zones (TAZs)
- **Cleaned Infrastructure:** 1,362 public charging stations and 17,532 charging piles
- **License:** Creative Commons Zero 1.0 Universal (CC0 1.0)

## Official References & Citations
- **Nature Scientific Data Research Article:**  
  [https://www.nature.com/articles/s41597-025-04874-4](https://www.nature.com/articles/s41597-025-04874-4)
- **Official GitHub Repository:**  
  [https://github.com/IntelligentSystemsLab/UrbanEV](https://github.com/IntelligentSystemsLab/UrbanEV)
- **Dryad Data Repository (Raw Records):**  
  [https://datadryad.org/dataset/doi:10.5061/dryad.np5hqc04z](https://datadryad.org/dataset/doi:10.5061/dryad.np5hqc04z)

## Files Description

### Processed Files (`data/processed/`)
- `volume.parquet`: Highly compressed (2.4 MB), high-throughput binary storage of hourly charging volume (kWh) across all 275 zones. Automatically indexed and validated.
- `weather_central.parquet`: Cleaned hourly meteorological observations from the central weather station (Air temperature, relative humidity, precipitation category, atmospheric pressure, dew point).
- `zone_metadata.csv`: Aggregated zone-level infrastructure data containing total station count, pile count, zone area, perimeter, and geographic centroid coordinates.

### Raw Files (`data/raw/`)
- `weather_central.csv`: Raw meteorological observations.
- `inf.csv`: Granular station-level infrastructure records (station ID, longitude, latitude, charger count, TAZID).
- `volume.csv`: Available locally or downloadable via `python scripts/download_dataset.py`. Excluded from Git tracking to preserve lightweight repository cloning.

## Engineering Notice & Academic Caveat
According to the official UrbanEV repository documentation, the charging volume in `volume.csv` is computed from charging pile rated power multiplied by charging session duration. Actual metered power utility consumption may differ due to battery State-of-Charge (SoC) charging taper curves, ambient temperature, and conversion losses.
