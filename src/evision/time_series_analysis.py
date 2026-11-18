"""
Time Series Analysis Module for EVision.
Provides diagnostic tools for EV demand analysis:
- Rolling statistics (mean, standard deviation, volatility)
- Time series decomposition (Trend, Seasonality, Residuals)
- Augmented Dickey-Fuller (ADF) Stationarity testing
- First-order and seasonal differencing analysis
- Autocorrelation Function (ACF) & Partial Autocorrelation Function (PACF)
- Plain-language educational explanations for business & utility stakeholders
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger("EVision.TimeSeriesAnalysis")


def compute_rolling_statistics(
    series: pd.Series,
    window_sizes: Tuple[int, int] = (24, 168),
) -> pd.DataFrame:
    """
    Computes rolling averages and rolling standard deviations for daily (24h)
    and weekly (168h) cycles to observe structural trends and volatility shifts.
    """
    df_roll = pd.DataFrame(index=series.index)
    df_roll["demand"] = series.values

    for w in window_sizes:
        label = f"{w}h"
        df_roll[f"rolling_mean_{label}"] = series.rolling(window=w, min_periods=1).mean()
        df_roll[f"rolling_std_{label}"] = series.rolling(window=w, min_periods=1).std().fillna(0.0)

    return df_roll


def run_adf_test(series: pd.Series, maxlag: Optional[int] = None) -> Dict[str, Any]:
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.
    
    Null Hypothesis (H0): The series has a unit root (is non-stationary).
    Alternative Hypothesis (H1): The series has no unit root (is stationary).
    """
    clean_series = series.dropna()
    if len(clean_series) < 20:
        raise ValueError("Series length is too short for reliable ADF testing.")

    adf_result = adfuller(clean_series, maxlag=maxlag, autolag="AIC")

    test_stat = float(adf_result[0])
    p_value = float(adf_result[1])
    used_lag = int(adf_result[2])
    n_obs = int(adf_result[3])
    crit_values = {k: float(v) for k, v in adf_result[4].items()}

    # Stationarity verdict based on standard alpha=0.05
    is_stationary = bool(p_value < 0.05)

    if is_stationary:
        interpretation = (
            f"The p-value ({p_value:.4e}) is strictly below the significance threshold (0.05). "
            f"We reject the null hypothesis (H0): the charging demand series is STATISTICALLY STATIONARY. "
            f"Its mean and variance remain relatively stable across time."
        )
    else:
        interpretation = (
            f"The p-value ({p_value:.4f}) is above the significance threshold (0.05). "
            f"We fail to reject the null hypothesis (H0): the charging demand series is NON-STATIONARY. "
            f"It displays time-varying trend or variance drifts. Differencing or seasonal adjustment is recommended."
        )

    return {
        "test_statistic": test_stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "critical_values": crit_values,
        "is_stationary": is_stationary,
        "verdict": "Stationary" if is_stationary else "Non-Stationary",
        "interpretation": interpretation,
    }


def compute_differencing_analysis(series: pd.Series) -> Dict[str, Any]:
    """
    Computes first-order differencing (y_t - y_{t-1}) and seasonal differencing (y_t - y_{t-24}),
    evaluating stationarity improvements for classical ARIMA/SARIMA parameter tuning.
    """
    clean_series = series.dropna()
    first_diff = clean_series.diff(1).dropna()
    seasonal_diff = clean_series.diff(24).dropna()

    adf_original = run_adf_test(clean_series)
    adf_first_diff = run_adf_test(first_diff)
    adf_seasonal_diff = run_adf_test(seasonal_diff)

    return {
        "original_adf": adf_original,
        "first_diff_adf": adf_first_diff,
        "seasonal_diff_adf": adf_seasonal_diff,
        "first_diff_series": first_diff,
        "seasonal_diff_series": seasonal_diff,
    }


def decompose_time_series(
    df_demand: pd.DataFrame,
    period: int = 24,
    model: str = "additive",
) -> Dict[str, Any]:
    """
    Performs classical seasonal decomposition using statsmodels.
    
    Additive: y(t) = Trend(t) + Seasonal(t) + Residual(t)
    Multiplicative: y(t) = Trend(t) * Seasonal(t) * Residual(t)
    """
    df_ts = df_demand.copy()
    df_ts = df_ts.set_index("time").sort_index()

    # For multiplicative decomposition, avoid zero or negative values
    if model == "multiplicative":
        df_ts["demand"] = df_ts["demand"].clip(lower=0.01)

    decomp = seasonal_decompose(df_ts["demand"], period=period, model=model, extrapolate_trend="freq")

    df_result = pd.DataFrame({
        "time": df_ts.index,
        "observed": decomp.observed.values,
        "trend": decomp.trend.values,
        "seasonal": decomp.seasonal.values,
        "residual": decomp.resid.values,
    })

    return {
        "decomposition_df": df_result,
        "period": period,
        "model": model,
        "seasonal_amplitude": float(np.ptp(decomp.seasonal.dropna().values[:period])),
        "residual_std": float(np.nanstd(decomp.resid.values)),
    }


def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 72,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Calculates Autocorrelation (ACF) and Partial Autocorrelation (PACF) values
    along with 95% confidence intervals to guide ARIMA (p, d, q) selection.
    """
    clean_series = series.dropna()
    max_possible_lags = min(nlags, len(clean_series) // 2 - 1)

    acf_vals, acf_confint = acf(clean_series, nlags=max_possible_lags, alpha=alpha)
    pacf_vals, pacf_confint = pacf(clean_series, nlags=max_possible_lags, alpha=alpha, method="yule_walker")

    lags = np.arange(len(acf_vals))
    # Bartlett's approx threshold for significance: ~ 1.96 / sqrt(N)
    significance_bound = 1.96 / np.sqrt(len(clean_series))

    return {
        "lags": lags,
        "acf": acf_vals,
        "acf_confint_lower": acf_confint[:, 0],
        "acf_confint_upper": acf_confint[:, 1],
        "pacf": pacf_vals,
        "pacf_confint_lower": pacf_confint[:, 0],
        "pacf_confint_upper": pacf_confint[:, 1],
        "significance_bound": float(significance_bound),
    }


def compute_temporal_patterns(df_demand: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Computes stratified demand patterns:
    - Hourly diurnal curve (0-23)
    - Day of week pattern (Mon-Sun)
    - Weekday vs. Weekend comparative curve
    - Monthly trajectory
    """
    df = df_demand.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.day_of_week
    df["day_name"] = df["time"].dt.day_name()
    df["is_weekend"] = df["dayofweek"].isin([5, 6])
    df["month"] = df["time"].dt.month
    df["month_name"] = df["time"].dt.strftime("%b %Y")

    # 1. Hourly curve
    hourly = df.groupby("hour")["demand"].agg(["mean", "median", "std", "min", "max"]).reset_index()

    # 2. Day of week curve
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby(["dayofweek", "day_name"])["demand"].agg(["mean", "median", "std"]).reset_index()
    dow["day_name"] = pd.Categorical(dow["day_name"], categories=day_order, ordered=True)
    dow = dow.sort_values(by="dayofweek").reset_index(drop=True)

    # 3. Weekday vs Weekend hourly profile
    weekday_vs_weekend = df.groupby(["hour", "is_weekend"])["demand"].mean().unstack().reset_index()
    weekday_vs_weekend.columns = ["hour", "Weekday_Mean", "Weekend_Mean"]

    # 4. Monthly trend
    monthly = df.groupby("month_name", sort=False)["demand"].agg(["mean", "sum", "count"]).reset_index()

    return {
        "hourly": hourly,
        "day_of_week": dow,
        "weekday_vs_weekend": weekday_vs_weekend,
        "monthly": monthly,
    }
