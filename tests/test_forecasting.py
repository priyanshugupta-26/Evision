"""
Unit and integration tests for forecasting models and peak alerts.
"""

import numpy as np
import pandas as pd
import pytest

from evision.feature_engineering import generate_time_series_features
from evision.forecasting import generate_future_forecast, run_benchmark_suite
from evision.models import (
    ARIMAForecaster,
    NaiveForecaster,
    SARIMAXForecaster,
    SeasonalNaiveForecaster,
    XGBoostForecaster,
)
from evision.peak_alerts import calculate_zone_thresholds, evaluate_peak_alerts


@pytest.fixture
def synthetic_series():
    # 30 days of hourly data (720 hours) with realistic daily cycle
    times = pd.date_range("2022-09-01", periods=720, freq="h")
    hour = times.hour
    daily_pattern = 20.0 + 15.0 * np.sin(2 * np.pi * (hour - 6) / 24.0)
    noise = np.random.normal(0, 2.0, size=720)
    demand = np.clip(daily_pattern + noise, 0.0, None)
    return pd.DataFrame({"time": times, "demand": demand})


def test_naive_forecaster(synthetic_series):
    y = synthetic_series["demand"].values
    m = NaiveForecaster().fit(y[:-24])
    preds = m.predict(steps=24)
    assert len(preds) == 24
    assert np.all(preds == y[-25])


def test_seasonal_naive_forecaster(synthetic_series):
    y = synthetic_series["demand"].values
    m = SeasonalNaiveForecaster(season_length=24).fit(y[:-24])
    preds = m.predict(steps=24)
    assert len(preds) == 24
    assert np.allclose(preds, y[-48:-24])


def test_arima_forecaster(synthetic_series):
    y = synthetic_series["demand"].values
    m = ARIMAForecaster(order=(1, 1, 1)).fit(y[:-24])
    preds = m.predict(steps=24)
    assert len(preds) == 24
    assert (preds >= 0).all()


def test_sarimax_forecaster(synthetic_series):
    y = synthetic_series["demand"].values
    m = SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 1, 0, 24), maxiter=15).fit(y[:-24])
    preds = m.predict(steps=24)
    assert len(preds) == 24
    assert (preds >= 0).all()


def test_xgboost_recursive_forecaster(synthetic_series):
    feat_df = generate_time_series_features(synthetic_series, drop_na=True)
    m = XGBoostForecaster(n_estimators=30, max_depth=3)
    m.fit(feat_df, feat_df["demand"])

    # Test recursive projection across multiple horizons
    for h in [6, 12, 24, 48]:
        fc = m.forecast_recursive(synthetic_series, horizon_hours=h)
        assert len(fc) == h
        assert "predicted_demand" in fc.columns
        assert (fc["predicted_demand"] >= 0).all()


def test_peak_alerts_evaluation():
    hist = np.array([10.0] * 90 + [50.0] * 5 + [100.0] * 5)  # 90th percentile ~ 50, 95th ~ 100
    thresholds = calculate_zone_thresholds(hist)
    assert thresholds["p90"] > 0
    assert thresholds["p95"] >= thresholds["p90"]

    # Forecast with low, medium, and high values
    fc_df = pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=3, freq="h"),
        "predicted_demand_kwh": [10.0, thresholds["p90"] + 1.0, thresholds["p95"] + 5.0],
    })
    res = evaluate_peak_alerts(fc_df, thresholds, zone_name="TestZone")
    assert res["normal_hours_count"] == 1
    assert res["high_demand_hours_count"] == 1
    assert res["critical_hours_count"] == 1
    assert len(res["recommendations"]) > 0


def test_full_benchmark_suite(synthetic_series):
    bench = run_benchmark_suite(synthetic_series, eval_horizon=24)
    assert "val_leaderboard" in bench
    assert "test_leaderboard" in bench
    assert len(bench["val_leaderboard"]) == 5
    assert len(bench["test_leaderboard"]) == 5
    assert bench["best_model_name"] in [
        "Naive (Persistence)",
        "Seasonal Naive (24h)",
        "ARIMA",
        "SARIMAX",
        "XGBoost Regressor",
    ]
