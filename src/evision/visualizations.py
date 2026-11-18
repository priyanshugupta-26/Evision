"""
Interactive Visualization Module for EVision.
Generates polished, SaaS-grade Plotly charts adhering to the Dark Navy & Electric Blue design system.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evision.config import THEME_COLORS


def get_evision_plotly_layout(title: str = "", height: int = 420) -> dict:
    """Standardizes SaaS dark theme layout parameters for all Plotly figures."""
    return dict(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=15, color=THEME_COLORS["text_main"]),
            x=0.01,
            y=0.97,
        ),
        paper_bgcolor=THEME_COLORS["bg_card"],
        plot_bgcolor=THEME_COLORS["bg_card"],
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color=THEME_COLORS["text_muted"], size=12),
        margin=dict(l=45, r=25, t=50, b=40),
        height=height,
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor=THEME_COLORS["grid_color"],
            linecolor="rgba(255, 255, 255, 0.15)",
            tickfont=dict(color=THEME_COLORS["text_muted"], size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=THEME_COLORS["grid_color"],
            linecolor="rgba(255, 255, 255, 0.15)",
            tickfont=dict(color=THEME_COLORS["text_muted"], size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=THEME_COLORS["text_main"]),
            bgcolor="rgba(15, 28, 63, 0.6)",
        ),
    )


def plot_historical_demand(
    df_series: pd.DataFrame,
    zone_label: str = "Selected Zone",
    show_slider: bool = True,
) -> go.Figure:
    """Plots interactive time-series curve with range slider and volume gradients."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_series["time"],
        y=df_series["demand"],
        mode="lines",
        name=f"{zone_label} Demand",
        line=dict(color=THEME_COLORS["electric_blue"], width=1.8),
        fill="tozeroy",
        fillcolor="rgba(0, 210, 255, 0.08)",
        hovertemplate="%{x|%Y-%m-%d %H:00}<br>Demand: <b>%{y:.1f} kWh</b><extra></extra>",
    ))

    layout = get_evision_plotly_layout(f"Historical EV Charging Demand — {zone_label}")
    if show_slider:
        layout["xaxis"]["rangeslider"] = dict(visible=True, bgcolor=THEME_COLORS["bg_dark"])
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Charging Volume (kWh)")
    return fig


def plot_rolling_diagnostics(df_roll: pd.DataFrame, zone_label: str = "Zone") -> go.Figure:
    """Displays raw demand against 24h and 168h rolling statistics."""
    fig = go.Figure()

    # Raw demand
    fig.add_trace(go.Scatter(
        x=df_roll.index,
        y=df_roll["demand"],
        mode="lines",
        name="Observed Demand",
        line=dict(color="rgba(255, 255, 255, 0.25)", width=1),
        hoverinfo="skip",
    ))

    # Rolling 24h Mean
    if "rolling_mean_24h" in df_roll.columns:
        fig.add_trace(go.Scatter(
            x=df_roll.index,
            y=df_roll["rolling_mean_24h"],
            mode="lines",
            name="24h Rolling Mean (Diurnal Trend)",
            line=dict(color=THEME_COLORS["neon_cyan"], width=2.2),
        ))

    # Rolling 168h Mean (Weekly)
    if "rolling_mean_168h" in df_roll.columns:
        fig.add_trace(go.Scatter(
            x=df_roll.index,
            y=df_roll["rolling_mean_168h"],
            mode="lines",
            name="168h (7-Day) Rolling Mean",
            line=dict(color=THEME_COLORS["warning_yellow"], width=2.5),
        ))

    layout = get_evision_plotly_layout(f"Trend & Smoothing Diagnostics — {zone_label}")
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Demand (kWh)")
    return fig


def plot_decomposition(decomp_dict: Dict) -> go.Figure:
    """Plots stacked 4-panel time-series decomposition (Observed, Trend, Seasonal, Residual)."""
    df = decomp_dict["decomposition_df"]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Observed Series (y_t)", "Trend Component (T_t)", "Seasonal Component (S_t, 24h cycle)", "Residual Component (e_t)"],
    )

    fig.add_trace(go.Scatter(x=df["time"], y=df["observed"], line=dict(color=THEME_COLORS["text_main"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["trend"], line=dict(color=THEME_COLORS["electric_blue"], width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["seasonal"], line=dict(color=THEME_COLORS["neon_cyan"], width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["residual"], mode="markers", marker=dict(size=2.5, color=THEME_COLORS["danger_red"], opacity=0.6)), row=4, col=1)

    fig.update_layout(
        paper_bgcolor=THEME_COLORS["bg_card"],
        plot_bgcolor=THEME_COLORS["bg_card"],
        font=dict(color=THEME_COLORS["text_muted"], size=11),
        height=650,
        showlegend=False,
        margin=dict(l=45, r=25, t=40, b=30),
    )
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor=THEME_COLORS["grid_color"], row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=THEME_COLORS["grid_color"], row=i, col=1)
    return fig


def plot_acf_pacf(acf_pacf_dict: Dict) -> go.Figure:
    """Displays ACF and PACF bar plots with Bartlett's 95% confidence intervals."""
    lags = acf_pacf_dict["lags"]
    acf_vals = acf_pacf_dict["acf"]
    pacf_vals = acf_pacf_dict["pacf"]
    bound = acf_pacf_dict["significance_bound"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"])

    # ACF
    fig.add_trace(go.Bar(x=lags, y=acf_vals, marker_color=THEME_COLORS["electric_blue"], name="ACF"), row=1, col=1)
    fig.add_hline(y=bound, line_dash="dash", line_color=THEME_COLORS["warning_yellow"], line_width=1.2, row=1, col=1)
    fig.add_hline(y=-bound, line_dash="dash", line_color=THEME_COLORS["warning_yellow"], line_width=1.2, row=1, col=1)

    # PACF
    fig.add_trace(go.Bar(x=lags, y=pacf_vals, marker_color=THEME_COLORS["neon_cyan"], name="PACF"), row=1, col=2)
    fig.add_hline(y=bound, line_dash="dash", line_color=THEME_COLORS["warning_yellow"], line_width=1.2, row=1, col=2)
    fig.add_hline(y=-bound, line_dash="dash", line_color=THEME_COLORS["warning_yellow"], line_width=1.2, row=1, col=2)

    fig.update_layout(
        paper_bgcolor=THEME_COLORS["bg_card"],
        plot_bgcolor=THEME_COLORS["bg_card"],
        font=dict(color=THEME_COLORS["text_muted"], size=11),
        height=340,
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    for col in (1, 2):
        fig.update_xaxes(title_text="Lag Order (Hours)", showgrid=True, gridcolor=THEME_COLORS["grid_color"], row=1, col=col)
        fig.update_yaxes(title_text="Correlation", showgrid=True, gridcolor=THEME_COLORS["grid_color"], row=1, col=col)
    return fig


def plot_temporal_patterns(patterns: Dict[str, pd.DataFrame]) -> Dict[str, go.Figure]:
    """Generates diurnal, weekday vs weekend, day of week, and monthly pattern charts."""
    figs = {}

    # 1. Diurnal Hourly Profile
    hourly = patterns["hourly"]
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=hourly["hour"],
        y=hourly["mean"],
        mode="lines+markers",
        name="Mean Demand",
        line=dict(color=THEME_COLORS["electric_blue"], width=2.5),
        marker=dict(size=6, color=THEME_COLORS["neon_cyan"]),
    ))
    fig_h.add_trace(go.Scatter(
        x=hourly["hour"],
        y=hourly["median"],
        mode="lines",
        name="Median Demand",
        line=dict(color=THEME_COLORS["warning_yellow"], dash="dot", width=1.8),
    ))
    fig_h.update_layout(**get_evision_plotly_layout("Diurnal Hourly Profile (00:00 - 23:00)"))
    fig_h.update_xaxes(title_text="Hour of Day", tickmode="linear", tick0=0, dtick=2)
    fig_h.update_yaxes(title_text="Demand (kWh)")
    figs["hourly"] = fig_h

    # 2. Weekday vs Weekend Profile
    wvw = patterns["weekday_vs_weekend"]
    fig_wvw = go.Figure()
    fig_wvw.add_trace(go.Scatter(
        x=wvw["hour"],
        y=wvw["Weekday_Mean"],
        mode="lines+markers",
        name="Weekday Mean",
        line=dict(color=THEME_COLORS["electric_blue"], width=2.5),
    ))
    fig_wvw.add_trace(go.Scatter(
        x=wvw["hour"],
        y=wvw["Weekend_Mean"],
        mode="lines+markers",
        name="Weekend Mean",
        line=dict(color=THEME_COLORS["accent_pink"], width=2.5),
    ))
    fig_wvw.update_layout(**get_evision_plotly_layout("Weekday vs. Weekend Diurnal Comparison"))
    fig_wvw.update_xaxes(title_text="Hour of Day", tickmode="linear", tick0=0, dtick=2)
    fig_wvw.update_yaxes(title_text="Demand (kWh)")
    figs["weekday_vs_weekend"] = fig_wvw

    # 3. Day of Week Profile
    dow = patterns["day_of_week"]
    fig_dow = go.Figure()
    fig_dow.add_trace(go.Bar(
        x=dow["day_name"],
        y=dow["mean"],
        marker_color=[THEME_COLORS["accent_blue"] if i < 5 else THEME_COLORS["accent_pink"] for i in range(7)],
        name="Daily Mean Demand",
    ))
    fig_dow.update_layout(**get_evision_plotly_layout("Weekly Demand Profile (Day of Week)"))
    fig_dow.update_yaxes(title_text="Mean Demand (kWh)")
    figs["day_of_week"] = fig_dow

    return figs


def plot_future_forecast_with_alerts(
    forecast_data: Dict[str, Any],
    thresholds: Dict[str, float],
    zone_label: str = "Selected Zone",
) -> go.Figure:
    """
    Renders continuous forecast trajectory from recent actual history into future forecast,
    annotating high-demand thresholds and the expected peak hour.
    """
    hist_df = forecast_data["historical_recent"]
    fc_df = forecast_data["forecast_df"]
    peak_time = forecast_data["peak_time"]
    peak_demand = forecast_data["peak_demand_kwh"]

    p90 = thresholds.get("p90", 0.0)
    p95 = thresholds.get("p95", 0.0)

    fig = go.Figure()

    # Recent actuals
    fig.add_trace(go.Scatter(
        x=hist_df["time"],
        y=hist_df["demand"],
        mode="lines",
        name="Recent Actual Demand",
        line=dict(color="rgba(255, 255, 255, 0.6)", width=1.5),
        hovertemplate="%{x|%Y-%m-%d %H:00}<br>Actual: <b>%{y:.1f} kWh</b><extra></extra>",
    ))

    # Forecast trajectory (connect continuous line from last actual to first forecast)
    connect_times = [hist_df["time"].iloc[-1]] + list(fc_df["time"])
    connect_demand = [hist_df["demand"].iloc[-1]] + list(fc_df["predicted_demand_kwh"])

    fig.add_trace(go.Scatter(
        x=connect_times,
        y=connect_demand,
        mode="lines+markers",
        name=f"Predicted Demand ({forecast_data['model_name']})",
        line=dict(color=THEME_COLORS["electric_blue"], width=2.8),
        marker=dict(size=4, color=THEME_COLORS["neon_cyan"]),
        hovertemplate="%{x|%Y-%m-%d %H:00}<br>Forecast: <b>%{y:.1f} kWh</b><extra></extra>",
    ))

    # 90th percentile threshold (High Demand Alert)
    if p90 > 0:
        fig.add_hline(
            y=p90,
            line_dash="dash",
            line_color=THEME_COLORS["warning_yellow"],
            line_width=1.6,
            annotation_text=f"High Demand Alert (>90th %ile: {p90:.1f} kWh)",
            annotation_position="top left",
            annotation_font=dict(color=THEME_COLORS["warning_yellow"], size=10),
        )

    # 95th percentile threshold (Critical Alert)
    if p95 > 0:
        fig.add_hline(
            y=p95,
            line_dash="dot",
            line_color=THEME_COLORS["danger_red"],
            line_width=1.8,
            annotation_text=f"Critical Alert (>95th %ile: {p95:.1f} kWh)",
            annotation_position="top left",
            annotation_font=dict(color=THEME_COLORS["danger_red"], size=10),
        )

    # Peak demand callout marker
    fig.add_trace(go.Scatter(
        x=[peak_time],
        y=[peak_demand],
        mode="markers+text",
        name="Expected Peak",
        marker=dict(size=11, color=THEME_COLORS["danger_red"], symbol="star"),
        text=[f"Peak: {peak_demand:.1f} kWh"],
        textposition="top center",
        textfont=dict(color=THEME_COLORS["text_main"], size=11),
        hoverinfo="skip",
    ))

    layout = get_evision_plotly_layout(f"Forecast Horizon ({forecast_data['horizon_hours']} Hours Ahead) — {zone_label}", height=450)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Charging Volume (kWh)")
    return fig


def plot_model_comparison_curves(
    actual_times: np.ndarray,
    actual_values: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    split_name: str = "Validation",
) -> go.Figure:
    """Plots actual values versus predictions from all benchmark models side-by-side."""
    fig = go.Figure()

    # Actual ground truth
    fig.add_trace(go.Scatter(
        x=actual_times,
        y=actual_values,
        mode="lines",
        name="Ground Truth (Actual)",
        line=dict(color="#FFFFFF", width=2.5),
        hovertemplate="%{x|%Y-%m-%d %H:00}<br>Actual: <b>%{y:.1f} kWh</b><extra></extra>",
    ))

    colors = [
        THEME_COLORS["electric_blue"],
        THEME_COLORS["neon_cyan"],
        THEME_COLORS["warning_yellow"],
        THEME_COLORS["accent_purple"],
        THEME_COLORS["accent_pink"],
    ]

    for (name, preds), color in zip(predictions_dict.items(), colors):
        fig.add_trace(go.Scatter(
            x=actual_times,
            y=preds,
            mode="lines",
            name=name,
            line=dict(color=color, width=1.8, dash="solid" if "XGBoost" in name else "dot"),
            hovertemplate=f"%{{x|%Y-%m-%d %H:00}}<br>{name}: <b>%{{y:.1f}} kWh</b><extra></extra>",
        ))

    layout = get_evision_plotly_layout(f"Forecasting Models Comparison — {split_name} Horizon Window")
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Demand (kWh)")
    return fig


def plot_model_metrics_bar(leaderboard_df: pd.DataFrame) -> go.Figure:
    """Displays comparative MAE and RMSE performance across models."""
    df = leaderboard_df.copy()
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Model"],
        y=df["MAE (kWh)"],
        name="MAE (kWh)",
        marker_color=THEME_COLORS["electric_blue"],
        text=df["MAE (kWh)"],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        x=df["Model"],
        y=df["RMSE (kWh)"],
        name="RMSE (kWh)",
        marker_color=THEME_COLORS["accent_purple"],
        text=df["RMSE (kWh)"],
        textposition="auto",
    ))

    layout = get_evision_plotly_layout("Model Accuracy Leaderboard (Lower Error is Better)", height=380)
    layout["barmode"] = "group"
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Error Metric (kWh)")
    return fig


def plot_top_zones_bar(top_zones_df: pd.DataFrame, title: str = "Top Charging Zones") -> go.Figure:
    """Horizontal bar chart for top charging zones."""
    df = top_zones_df.sort_values(by=top_zones_df.columns[1], ascending=True)
    val_col = df.columns[1]

    fig = go.Figure(go.Bar(
        x=df[val_col],
        y=[f"Zone {z}" for z in df["zone_id"]],
        orientation="h",
        marker=dict(
            color=df[val_col],
            colorscale=[[0, THEME_COLORS["accent_blue"]], [1, THEME_COLORS["neon_cyan"]]],
        ),
        text=np.round(df[val_col], 1),
        textposition="auto",
    ))

    layout = get_evision_plotly_layout(title, height=360)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Average Hourly Demand (kWh)")
    return fig


def plot_weather_correlation(
    df_series: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> Optional[go.Figure]:
    """Generates correlation heatmap between charging demand and meteorological variables."""
    if df_weather is None or df_weather.empty:
        return None

    df_comb = pd.merge(df_series, df_weather, on="time", how="inner")
    numeric_cols = ["demand", "temperature_c", "humidity_pct", "rain_level", "pressure_station_mmhg", "dewpoint_c"]
    available_cols = [c for c in numeric_cols if c in df_comb.columns]

    if len(available_cols) < 2:
        return None

    corr_matrix = df_comb[available_cols].corr()
    clean_labels = [c.replace("_", " ").title() for c in available_cols]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=clean_labels,
        y=clean_labels,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont=dict(color=THEME_COLORS["text_main"], size=11),
    ))

    layout = get_evision_plotly_layout("Correlation Matrix: EV Demand vs. Weather Variables", height=400)
    fig.update_layout(**layout)
    return fig


def plot_zone_map(zone_meta_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plots interactive charging pile density map of Shenzhen administrative zones."""
    if zone_meta_df is None or "centroid_lat" not in zone_meta_df.columns:
        return None

    df = zone_meta_df.dropna(subset=["centroid_lat", "centroid_lon"]).copy()
    if df.empty:
        return None

    fig = px.scatter_mapbox(
        df,
        lat="centroid_lat",
        lon="centroid_lon",
        size="pile_count",
        color="station_count",
        hover_name="zone_id",
        hover_data={"station_count": True, "pile_count": True, "centroid_lat": False, "centroid_lon": False},
        color_continuous_scale=[[0, "#3A86FF"], [0.5, "#00D2FF"], [1, "#00F5D4"]],
        zoom=9.5,
        center=dict(lat=22.5431, lon=114.0579),
        mapbox_style="carto-darkmatter",
    )

    fig.update_layout(
        paper_bgcolor=THEME_COLORS["bg_card"],
        margin=dict(l=0, r=0, t=35, b=0),
        height=450,
        title=dict(
            text="<b>Shenzhen Charging Hub Infrastructure Distribution</b>",
            font=dict(color=THEME_COLORS["text_main"], size=14),
            x=0.01,
        ),
    )
    return fig
