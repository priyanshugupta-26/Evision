"""
Forecasting Models Module for EVision.
Implements the 5 core benchmark models:
1. Naive (Persistence) Forecaster
2. Seasonal Naive (24-hour diurnal cycle) Forecaster
3. ARIMA (Autoregressive Integrated Moving Average)
4. SARIMAX (Seasonal ARIMA with diurnal seasonality)
5. XGBoost Regressor (Gradient Boosted Trees with recursive multi-step forecasting)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

from evision.feature_engineering import get_feature_names

logger = logging.getLogger("EVision.Models")


class NaiveForecaster:
    """
    Naive (Persistence) Model:
    Forecasts all future periods as the last known observed actual demand value.
    y_hat_{t+h} = y_t
    """
    def __init__(self):
        self.last_value: float = 0.0
        self.is_fitted: bool = False

    def fit(self, y_train: Union[pd.Series, np.ndarray]) -> "NaiveForecaster":
        arr = np.asarray(y_train, dtype=float)
        if len(arr) == 0:
            raise ValueError("Training series cannot be empty.")
        self.last_value = float(arr[-1])
        self.is_fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        return np.full(shape=(steps,), fill_value=self.last_value, dtype=float)


class SeasonalNaiveForecaster:
    """
    Seasonal Naive Model:
    Forecasts each future hour as the value observed exactly one seasonal cycle ago (e.g. 24 hours).
    y_hat_{t+h} = y_{t + h - season_length}
    """
    def __init__(self, season_length: int = 24):
        self.season_length = season_length
        self.seasonal_cycle: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, y_train: Union[pd.Series, np.ndarray]) -> "SeasonalNaiveForecaster":
        arr = np.asarray(y_train, dtype=float)
        if len(arr) < self.season_length:
            raise ValueError(f"Series length ({len(arr)}) is less than season length ({self.season_length})")
        # Store the last complete seasonal cycle
        self.seasonal_cycle = arr[-self.season_length :]
        self.is_fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.is_fitted or self.seasonal_cycle is None:
            raise RuntimeError("Model must be fitted before predict().")
        # Tile the seasonal cycle to cover the required horizon
        repetitions = int(np.ceil(steps / self.season_length))
        tiled = np.tile(self.seasonal_cycle, repetitions)
        return tiled[:steps].astype(float)


class ARIMAForecaster:
    """
    Classical Autoregressive Integrated Moving Average (ARIMA).
    Default order: (p=2, d=1, q=1).
    Fast CPU estimation suitable for production time-series pipelines.
    """
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 1)):
        self.order = order
        self.fitted_model = None
        self.is_fitted = False

    def fit(self, y_train: Union[pd.Series, np.ndarray]) -> "ARIMAForecaster":
        arr = np.asarray(y_train, dtype=float)
        if len(arr) < 30:
            raise ValueError("ARIMA requires at least 30 observations.")

        try:
            model = ARIMA(arr, order=self.order)
            self.fitted_model = model.fit()
            self.is_fitted = True
        except Exception as e:
            logger.warning(f"ARIMA{self.order} failed: {e}. Falling back to ARIMA(1, 1, 0)")
            model = ARIMA(arr, order=(1, 1, 0))
            self.fitted_model = model.fit()
            self.is_fitted = True

        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model is None:
            raise RuntimeError("Model must be fitted before predict().")
        preds = self.fitted_model.forecast(steps=steps)
        # Demand cannot physically be negative
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)


class SARIMAXForecaster:
    """
    Seasonal ARIMA (SARIMAX) with 24-hour diurnal seasonal period.
    Order: (p=1, d=0, q=1) x (P=1, D=1, Q=0, s=24).
    Tuned for high statistical fidelity and fast convergence on standard CPUs.
    """
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 24),
        maxiter: int = 25,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.maxiter = maxiter
        self.fitted_model = None
        self.is_fitted = False

    def fit(self, y_train: Union[pd.Series, np.ndarray]) -> "SARIMAXForecaster":
        arr = np.asarray(y_train, dtype=float)
        # Use recent history if training series is very long, ensuring sub-second fit
        if len(arr) > 720:
            fit_series = arr[-720:]  # last 30 days
        else:
            fit_series = arr

        try:
            model = SARIMAX(
                fit_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted_model = model.fit(disp=False, maxiter=self.maxiter)
            self.is_fitted = True
        except Exception as e:
            logger.warning(f"SARIMAX fit failed: {e}. Falling back to seasonal differencing with AR(1).")
            model = SARIMAX(fit_series, order=(1, 0, 0), seasonal_order=(0, 1, 0, 24), enforce_stationarity=False)
            self.fitted_model = model.fit(disp=False, maxiter=self.maxiter)
            self.is_fitted = True

        return self

    def predict(self, steps: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model is None:
            raise RuntimeError("Model must be fitted before predict().")
        preds = self.fitted_model.forecast(steps=steps)
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)


class XGBoostForecaster:
    """
    XGBoost Regressor for Electric Vehicle Charging Demand Forecasting.
    Uses autoregressive lags, rolling statistics, calendar, and trigonometric cyclical features.
    Supports multi-step recursive forecasting without future data leakage.
    """
    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 5,
        learning_rate: float = 0.07,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.feature_names = get_feature_names()
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=2,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
        )
        self.is_fitted = False
        self.last_history_series: Optional[pd.DataFrame] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        raw_history_df: Optional[pd.DataFrame] = None,
    ) -> "XGBoostForecaster":
        X = X_train[self.feature_names]
        y = np.asarray(y_train, dtype=float)

        self.model.fit(X, y)
        self.is_fitted = True

        if raw_history_df is not None:
            self.last_history_series = raw_history_df.copy()

        return self

    def predict_features(self, X_eval: pd.DataFrame) -> np.ndarray:
        """Direct evaluation on pre-generated feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        X = X_eval[self.feature_names]
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, None)

    def forecast_recursive(
        self,
        recent_history_df: pd.DataFrame,
        horizon_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Generates genuine future predictions hour-by-hour recursively.
        At each step h, uses past actuals + newly generated predictions to build
        autoregressive lags and rolling features without leaking future targets.
        
        Returns DataFrame with ['time', 'predicted_demand'].
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before forecast_recursive().")

        history = recent_history_df.copy().sort_values(by="time").reset_index(drop=True)
        last_time = pd.to_datetime(history["time"].iloc[-1])
        demand_list = list(history["demand"].values)

        forecasts: List[float] = []
        forecast_times: List[pd.Timestamp] = []

        for h in range(1, horizon_hours + 1):
            next_time = last_time + pd.Timedelta(hours=h)
            forecast_times.append(next_time)

            # Build feature dictionary for step h
            feat = {}
            # Lags: 1, 2, 3, 24, 48, 168
            for lag in [1, 2, 3, 24, 48, 168]:
                if len(demand_list) >= lag:
                    feat[f"lag_{lag}"] = demand_list[-lag]
                else:
                    feat[f"lag_{lag}"] = demand_list[0]

            # Rolling stats on strictly past values
            recent_6 = demand_list[-6:]
            recent_24 = demand_list[-24:] if len(demand_list) >= 24 else demand_list
            feat["rolling_mean_6"] = float(np.mean(recent_6))
            feat["rolling_std_6"] = float(np.std(recent_6))
            feat["rolling_mean_24"] = float(np.mean(recent_24))
            feat["rolling_std_24"] = float(np.std(recent_24))

            # Calendar features
            h_val = next_time.hour
            dow = next_time.day_of_week
            m_val = next_time.month

            feat["hour"] = h_val
            feat["dayofweek"] = dow
            feat["is_weekend"] = 1 if dow in [5, 6] else 0
            feat["month"] = m_val

            # Cyclical features
            feat["hour_sin"] = np.sin(2 * np.pi * h_val / 24.0)
            feat["hour_cos"] = np.cos(2 * np.pi * h_val / 24.0)
            feat["day_sin"] = np.sin(2 * np.pi * dow / 7.0)
            feat["day_cos"] = np.cos(2 * np.pi * dow / 7.0)
            feat["month_sin"] = np.sin(2 * np.pi * (m_val - 1) / 12.0)
            feat["month_cos"] = np.cos(2 * np.pi * (m_val - 1) / 12.0)

            # Form row vector and predict
            row_df = pd.DataFrame([feat])[self.feature_names]
            step_pred = float(self.model.predict(row_df)[0])
            step_pred = max(0.0, step_pred)

            forecasts.append(step_pred)
            demand_list.append(step_pred)

        return pd.DataFrame({
            "time": forecast_times,
            "predicted_demand": forecasts,
        })
