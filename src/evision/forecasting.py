"""
Forecasting Orchestration Module for EVision.
Coordinates multi-model chronological benchmarking and future horizon forecasting.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from evision.config import AVAILABLE_HORIZONS, DEFAULT_HORIZON
from evision.evaluation import compare_models, evaluate_forecast
from evision.feature_engineering import (
    generate_time_series_features,
    prepare_train_val_test_features,
)
from evision.models import (
    ARIMAForecaster,
    NaiveForecaster,
    SARIMAXForecaster,
    SeasonalNaiveForecaster,
    XGBoostForecaster,
)
from evision.preprocessing import chronological_split

logger = logging.getLogger("EVision.Forecasting")


def run_benchmark_suite(
    df_series: pd.DataFrame,
    eval_horizon: int = 24,
) -> Dict[str, Any]:
    """
    Runs chronological evaluation across all 5 forecasting models.
    
    1. Splits dataset into 70% Train, 15% Validation, 15% Test.
    2. Trains each model strictly on training data.
    3. Evaluates predictions on the validation set for model ranking.
    4. Evaluates on the held-out test set for unbiased generalization.
    5. Returns leaderboard comparison and prediction curves.
    """
    train_df, val_df, test_df = chronological_split(df_series)

    # Feature engineering for XGBoost
    feat_train, feat_val, feat_test = prepare_train_val_test_features(train_df, val_df, test_df)

    # Predictions dictionary for plotting
    val_predictions = {}
    val_actual = val_df["demand"].values[:eval_horizon]
    val_times = val_df["time"].values[:eval_horizon]

    test_actual = test_df["demand"].values[:eval_horizon]
    test_times = test_df["time"].values[:eval_horizon]

    models_dict = {
        "Naive (Persistence)": NaiveForecaster(),
        "Seasonal Naive (24h)": SeasonalNaiveForecaster(season_length=24),
        "ARIMA": ARIMAForecaster(order=(2, 1, 1)),
        "SARIMAX": SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 1, 0, 24)),
        "XGBoost Regressor": XGBoostForecaster(n_estimators=80, max_depth=5),
    }

    val_eval_list = []
    test_eval_list = []
    fitted_models = {}

    for name, model in models_dict.items():
        logger.info(f"Training and benchmarking {name}...")

        if name == "XGBoost Regressor":
            model.fit(feat_train, feat_train["demand"], raw_history_df=train_df)
            val_preds_df = model.forecast_recursive(train_df, horizon_hours=eval_horizon)
            val_preds = val_preds_df["predicted_demand"].values

            # Test evaluation: train on train+val
            combined_train_val = pd.concat([train_df, val_df]).reset_index(drop=True)
            feat_tv = generate_time_series_features(combined_train_val, drop_na=True)
            test_xgb = XGBoostForecaster(n_estimators=80, max_depth=5)
            test_xgb.fit(feat_tv, feat_tv["demand"])
            test_preds_df = test_xgb.forecast_recursive(combined_train_val, horizon_hours=eval_horizon)
            test_preds = test_preds_df["predicted_demand"].values
        else:
            y_train = train_df["demand"].values
            model.fit(y_train)
            val_preds = model.predict(steps=eval_horizon)

            # Test evaluation model
            combined_y = np.concatenate([train_df["demand"].values, val_df["demand"].values])
            if name == "Naive (Persistence)":
                test_model = NaiveForecaster().fit(combined_y)
            elif name == "Seasonal Naive (24h)":
                test_model = SeasonalNaiveForecaster(season_length=24).fit(combined_y)
            elif name == "ARIMA":
                test_model = ARIMAForecaster(order=(2, 1, 1)).fit(combined_y)
            elif name == "SARIMAX":
                test_model = SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 1, 0, 24)).fit(combined_y)
            test_preds = test_model.predict(steps=eval_horizon)

        fitted_models[name] = model
        val_predictions[name] = val_preds

        # Evaluation metrics
        val_eval = evaluate_forecast(val_actual, val_preds, model_name=name)
        test_eval = evaluate_forecast(test_actual, test_preds, model_name=name)

        val_eval_list.append(val_eval)
        test_eval_list.append(test_eval)

    # Leaderboard DataFrames
    val_leaderboard = compare_models(val_eval_list)
    test_leaderboard = compare_models(test_eval_list)

    best_model_name = val_leaderboard.iloc[0]["Model"]

    return {
        "val_leaderboard": val_leaderboard,
        "test_leaderboard": test_leaderboard,
        "best_model_name": best_model_name,
        "val_times": val_times,
        "val_actual": val_actual,
        "val_predictions": val_predictions,
        "test_times": test_times,
        "test_actual": test_actual,
        "fitted_models": fitted_models,
        "eval_horizon": eval_horizon,
    }


def generate_future_forecast(
    df_series: pd.DataFrame,
    model_name: str,
    horizon_hours: int = DEFAULT_HORIZON,
) -> Dict[str, Any]:
    """
    Generates genuine future demand forecasts for a user-specified horizon (6, 12, 24, 48 hours).
    Fits the requested model on recent historical observations and projects future demand forward.
    """
    if horizon_hours not in AVAILABLE_HORIZONS:
        logger.warning(f"Horizon {horizon_hours} not in standard list. Proceeding with {horizon_hours} hours.")

    df_clean = df_series.copy().sort_values(by="time").reset_index(drop=True)
    last_timestamp = pd.to_datetime(df_clean["time"].iloc[-1])
    future_times = [last_timestamp + pd.Timedelta(hours=h) for h in range(1, horizon_hours + 1)]

    # Model dispatch
    if "XGBoost" in model_name:
        feat_df = generate_time_series_features(df_clean, drop_na=True)
        forecaster = XGBoostForecaster(n_estimators=100, max_depth=5)
        forecaster.fit(feat_df, feat_df["demand"], raw_history_df=df_clean)
        forecast_df = forecaster.forecast_recursive(df_clean, horizon_hours=horizon_hours)
        preds = forecast_df["predicted_demand"].values

    elif "Seasonal Naive" in model_name:
        forecaster = SeasonalNaiveForecaster(season_length=24).fit(df_clean["demand"].values)
        preds = forecaster.predict(steps=horizon_hours)

    elif "Naive" in model_name:
        forecaster = NaiveForecaster().fit(df_clean["demand"].values)
        preds = forecaster.predict(steps=horizon_hours)

    elif "SARIMAX" in model_name:
        forecaster = SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 1, 0, 24)).fit(df_clean["demand"].values)
        preds = forecaster.predict(steps=horizon_hours)

    elif "ARIMA" in model_name:
        forecaster = ARIMAForecaster(order=(2, 1, 1)).fit(df_clean["demand"].values)
        preds = forecaster.predict(steps=horizon_hours)

    else:
        # Default fallback to XGBoost
        return generate_future_forecast(df_series, "XGBoost Regressor", horizon_hours)

    preds = np.clip(np.asarray(preds, dtype=float), 0.0, None)

    # Build forecast table
    forecast_df = pd.DataFrame({
        "time": future_times,
        "predicted_demand_kwh": np.round(preds, 2),
        "hour_of_day": [t.strftime("%H:00") for t in future_times],
        "day_of_week": [t.strftime("%A") for t in future_times],
    })

    # Summary analytics
    peak_idx = int(np.argmax(preds))
    peak_time = future_times[peak_idx]
    peak_demand = float(preds[peak_idx])
    min_demand = float(np.min(preds))
    mean_demand = float(np.mean(preds))
    total_energy_kwh = float(np.sum(preds))

    return {
        "model_name": model_name,
        "horizon_hours": horizon_hours,
        "forecast_df": forecast_df,
        "last_observed_time": last_timestamp,
        "peak_time": peak_time,
        "peak_time_str": peak_time.strftime("%Y-%m-%d %H:00"),
        "peak_demand_kwh": peak_demand,
        "min_demand_kwh": min_demand,
        "mean_demand_kwh": mean_demand,
        "total_energy_kwh": total_energy_kwh,
        "historical_recent": df_clean.iloc[-72:].copy(),  # last 72 hours for continuous context plotting
    }
