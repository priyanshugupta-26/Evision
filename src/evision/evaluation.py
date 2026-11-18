"""
Evaluation & Benchmarking Module for EVision.
Provides mathematically sound, industry-standard metrics for time-series forecasting:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- sMAPE (Symmetric Mean Absolute Percentage Error, robust to zeros)
- Zero-handled MAPE (Mean Absolute Percentage Error on non-zero charging sessions)
- R-squared (Coefficient of Determination)
- Structured model comparison & ranking
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("EVision.Evaluation")


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Computes Symmetric Mean Absolute Percentage Error (sMAPE) in percentage [0, 200%].
    
    sMAPE = (100% / n) * sum( 2 * |y - y_hat| / (|y| + |y_hat| + eps) )
    Handles zero actual values cleanly without infinity blow-ups.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    numerator = 2.0 * np.abs(y_pred - y_true)
    return float(np.mean(numerator / denominator) * 100.0)


def calculate_filtered_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[Optional[float], int]:
    """
    Computes conventional MAPE on non-zero observations where actual demand exceeds threshold.
    Returns (mape_value, valid_count). Returns (None, 0) if no valid samples exist.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > threshold
    valid_count = int(np.sum(mask))

    if valid_count == 0:
        return None, 0

    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return mape, valid_count


def evaluate_forecast(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Calculates full benchmark metric suite for a given set of predictions.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # Clean any invalid entries
    valid_mask = np.isfinite(y_t) & np.isfinite(y_p)
    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]

    if len(y_t) == 0:
        raise ValueError(f"No valid evaluation samples for {model_name}")

    # Clip negative predictions to zero (physical domain constraint for EV power)
    y_p = np.clip(y_p, 0.0, None)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    smape = calculate_smape(y_t, y_p)
    mape, n_mape_samples = calculate_filtered_mape(y_t, y_p, threshold=0.5)

    # R2 score (variance explained)
    try:
        r2 = float(r2_score(y_t, y_p))
    except Exception:
        r2 = 0.0

    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "mape": mape,
        "n_samples": len(y_t),
        "r2": r2,
    }


def compare_models(evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregates individual model metrics into a sorted comparison leaderboard.
    Ranks models primarily by MAE (lower is better), then RMSE.
    """
    if not evaluations:
        return pd.DataFrame()

    records = []
    for ev in evaluations:
        records.append({
            "Model": ev["model_name"],
            "MAE (kWh)": round(ev["mae"], 3),
            "RMSE (kWh)": round(ev["rmse"], 3),
            "sMAPE (%)": round(ev["smape"], 2),
            "MAPE (kWh > 0.5) (%)": f"{ev['mape']:.2f}%" if ev["mape"] is not None else "N/A",
            "R² Score": round(ev["r2"], 3),
            "Samples": ev["n_samples"],
        })

    df_comp = pd.DataFrame(records)
    df_comp = df_comp.sort_values(by=["MAE (kWh)", "RMSE (kWh)"]).reset_index(drop=True)
    return df_comp
