"""
Peak Demand Warning & Operational Recommendation System for EVision.
Analyzes forecasted demand against historical statistical percentiles (90th and 95th),
detects elevated loading periods, and generates actionable operational recommendations.

Adheres strictly to utility domain standards:
Uses 'HIGH DEMAND ALERT' instead of 'GRID OVERLOAD' since dataset lacks physical feeder capacity.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from evision.config import (
    CRITICAL_DEMAND_PERCENTILE_ALERT,
    HIGH_DEMAND_PERCENTILE_ALERT,
)

logger = logging.getLogger("EVision.PeakAlerts")


def calculate_zone_thresholds(
    historical_series: Union[pd.Series, np.ndarray],
    p_high: float = HIGH_DEMAND_PERCENTILE_ALERT,
    p_critical: float = CRITICAL_DEMAND_PERCENTILE_ALERT,
) -> Dict[str, float]:
    """
    Computes statistical percentiles for the selected zone based on historical data.
    """
    arr = np.asarray(historical_series, dtype=float)
    clean_arr = arr[np.isfinite(arr)]

    if len(clean_arr) == 0:
        return {"p90": 0.0, "p95": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}

    p90 = float(np.percentile(clean_arr, p_high))
    p95 = float(np.percentile(clean_arr, p_critical))
    median_val = float(np.median(clean_arr))
    mean_val = float(np.mean(clean_arr))
    max_val = float(np.max(clean_arr))

    return {
        "p90": p90,
        "p95": p95,
        "median": median_val,
        "mean": mean_val,
        "max": max_val,
    }


def evaluate_peak_alerts(
    forecast_df: pd.DataFrame,
    thresholds: Dict[str, float],
    zone_name: str = "Selected Zone",
) -> Dict[str, Any]:
    """
    Evaluates forecasted demand against zone percentiles and produces
    alert categories and operational management strategies.
    
    Alert Levels:
    - NORMAL: Forecast <= 90th percentile
    - HIGH DEMAND ALERT: Forecast between 90th and 95th percentile
    - CRITICAL PEAK ALERT: Forecast > 95th percentile
    """
    df = forecast_df.copy()
    pred_col = "predicted_demand_kwh"
    if pred_col not in df.columns:
        raise ValueError(f"forecast_df must contain '{pred_col}' column.")

    p90 = thresholds["p90"]
    p95 = thresholds["p95"]

    alert_levels = []
    alert_badges = []

    for val in df[pred_col]:
        if val >= p95:
            alert_levels.append("CRITICAL PEAK ALERT")
            alert_badges.append("🚨 Critical Peak (>95th %ile)")
        elif val >= p90:
            alert_levels.append("HIGH DEMAND ALERT")
            alert_badges.append("⚠️ High Demand (>90th %ile)")
        else:
            alert_levels.append("NORMAL")
            alert_badges.append("✅ Normal Demand")

    df["alert_level"] = alert_levels
    df["status_badge"] = alert_badges

    high_demand_count = int((df["alert_level"] == "HIGH DEMAND ALERT").sum())
    critical_count = int((df["alert_level"] == "CRITICAL PEAK ALERT").sum())
    normal_count = int((df["alert_level"] == "NORMAL").sum())

    max_idx = int(df[pred_col].idxmax())
    max_row = df.iloc[max_idx]
    peak_demand = float(max_row[pred_col])
    peak_time_str = str(max_row["time"])
    peak_level = max_row["alert_level"]

    # Generate operations recommendations
    recommendations = generate_operational_actions(
        peak_level=peak_level,
        peak_demand=peak_demand,
        p90=p90,
        p95=p95,
        critical_count=critical_count,
        high_count=high_demand_count,
        zone_name=zone_name,
    )

    return {
        "annotated_forecast_df": df,
        "high_demand_hours_count": high_demand_count,
        "critical_hours_count": critical_count,
        "normal_hours_count": normal_count,
        "has_alerts": (high_demand_count + critical_count) > 0,
        "peak_time_str": peak_time_str,
        "peak_demand_kwh": peak_demand,
        "peak_level": peak_level,
        "threshold_p90": p90,
        "threshold_p95": p95,
        "recommendations": recommendations,
    }


def generate_operational_actions(
    peak_level: str,
    peak_demand: float,
    p90: float,
    p95: float,
    critical_count: int,
    high_count: int,
    zone_name: str,
) -> List[Dict[str, str]]:
    """
    Synthesizes actionable operations guidelines for charging hub managers and dispatchers.
    """
    actions = []

    if peak_level == "CRITICAL PEAK ALERT":
        actions.append({
            "category": "Tariff & Demand Management",
            "title": "Activate Dynamic Peak Surcharges",
            "action": (
                f"Elevate service pricing during critical peak hours in {zone_name} "
                f"to encourage price-sensitive private and ride-hail EV drivers to charge during off-peak windows."
            ),
            "priority": "HIGH",
        })
        actions.append({
            "category": "Traffic & Navigation Routing",
            "title": "Dispatch Smart App Navigation Rerouting",
            "action": (
                f"Push in-app recommendations and navigation incentives via charging apps "
                f"to direct incoming EV drivers to adjacent charging hubs with lower utilization."
            ),
            "priority": "HIGH",
        })
        actions.append({
            "category": "Infrastructure & BESS Dispatch",
            "title": "Discharge On-Site Battery Storage (BESS)",
            "action": (
                f"Trigger scheduled battery energy storage discharge to buffer the anticipated {peak_demand:.1f} kWh "
                f"demand spike, shaving localized utility transformer load."
            ),
            "priority": "CRITICAL",
        })
        actions.append({
            "category": "Maintenance Coordination",
            "title": "Freeze Hub Maintenance & Ensure 100% Pile Availability",
            "action": (
                f"Suspend non-critical charger maintenance in {zone_name} during elevated alert hours "
                f"to prevent queue bottlenecks."
            ),
            "priority": "MEDIUM",
        })

    elif peak_level == "HIGH DEMAND ALERT":
        actions.append({
            "category": "Demand Shifting",
            "title": "Notify Commercial Fleet Operators",
            "action": (
                f"Send automated demand-response notifications to registered commercial delivery/taxi fleets "
                f"suggesting scheduled staggered charging prior to the expected peak at {p90:.1f} kWh."
            ),
            "priority": "MEDIUM",
        })
        actions.append({
            "category": "Station Operations",
            "title": "Deploy On-Site Charging Attendants / Queue Management",
            "action": (
                f"Prepare digital bay management to minimize idle plug-in durations and increase turnover in {zone_name}."
            ),
            "priority": "MEDIUM",
        })
        actions.append({
            "category": "Energy Dispatch",
            "title": "Stage Local Distributed Generation / Solar Self-Consumption",
            "action": (
                f"Ensure rooftop solar / microgrid inverters are prioritized for local charging piles."
            ),
            "priority": "LOW",
        })

    else:
        actions.append({
            "category": "Standard Operations",
            "title": "Normal Operational State",
            "action": (
                f"Demand in {zone_name} is projected within normal historical baseline limits. "
                f"Standard time-of-use tariffs and standard pile operations are optimal."
            ),
            "priority": "INFO",
        })
        actions.append({
            "category": "Preventative Maintenance",
            "title": "Routine Inspection Window",
            "action": (
                f"Ideal opportunity to perform routine pile firmware updates, connector health checks, "
                f"and substation telemetry validation."
            ),
            "priority": "INFO",
        })

    return actions
