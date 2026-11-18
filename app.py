"""
EVision — Intelligent EV Charging Demand Forecasting & Peak Load Prediction System
Main Streamlit Application Entrypoint.
"""

from io import BytesIO
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="EVision — EV Charging Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evision.config import (
    AVAILABLE_HORIZONS,
    CORE_MODELS,
    CRITICAL_DEMAND_PERCENTILE_ALERT,
    DATA_LIMITATION_NOTICE,
    DEFAULT_HORIZON,
    HIGH_DEMAND_PERCENTILE_ALERT,
    THEME_COLORS,
    URBANEV_DRYAD_URL,
    URBANEV_PAPER_URL,
    URBANEV_REPO_URL,
)
from evision.data_loader import (
    DataLoaderError,
    load_volume_data,
    load_weather_data,
    load_zone_metadata,
)
from evision.evaluation import compare_models
from evision.forecasting import generate_future_forecast, run_benchmark_suite
from evision.peak_alerts import calculate_zone_thresholds, evaluate_peak_alerts
from evision.preprocessing import (
    compute_dataset_summary,
    get_series_for_zone,
    get_zone_list,
)
from evision.time_series_analysis import (
    compute_acf_pacf,
    compute_differencing_analysis,
    compute_rolling_statistics,
    compute_temporal_patterns,
    decompose_time_series,
    run_adf_test,
)
from evision.visualizations import (
    plot_acf_pacf,
    plot_decomposition,
    plot_future_forecast_with_alerts,
    plot_historical_demand,
    plot_model_comparison_curves,
    plot_model_metrics_bar,
    plot_rolling_diagnostics,
    plot_temporal_patterns,
    plot_top_zones_bar,
    plot_weather_correlation,
    plot_zone_map,
)

# Custom SaaS UI Styling (Dark Navy & Electric Blue)
st.markdown("""
<style>
    /* Global theme overrides */
    .stApp {
        background-color: #070D1E;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header & Branding */
    .evision-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }
    .evision-logo {
        background: linear-gradient(135deg, #00D2FF 0%, #3A86FF 100%);
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 20px;
        color: #070D1E;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.4);
    }
    .evision-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF 0%, #00D2FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .evision-subtitle {
        font-size: 13px;
        color: #8E9EB5;
        margin-top: -4px;
        margin-bottom: 20px;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(145deg, #0F1C3F 0%, #0B1530 100%);
        border: 1px solid rgba(0, 210, 255, 0.15);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 210, 255, 0.4);
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #8E9EB5;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .kpi-sub {
        font-size: 11px;
        color: #00D2FF;
        margin-top: 4px;
    }
    
    /* Alert Badges */
    .badge-critical {
        background: rgba(255, 77, 109, 0.15);
        color: #FF4D6D;
        border: 1px solid rgba(255, 77, 109, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-high {
        background: rgba(255, 190, 11, 0.15);
        color: #FFBE0B;
        border: 1px solid rgba(255, 190, 11, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-normal {
        background: rgba(6, 214, 160, 0.15);
        color: #06D6A0;
        border: 1px solid rgba(6, 214, 160, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Disclaimer / Callout banner */
    .disclaimer-box {
        background: rgba(15, 28, 63, 0.7);
        border-left: 3px solid #00D2FF;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 12px;
        color: #A0AEC0;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #0A132C;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# Data Caching Functions
@st.cache_data(show_spinner=False)
def get_cached_volume_data():
    return load_volume_data()


@st.cache_data(show_spinner=False)
def get_cached_weather_data():
    return load_weather_data()


@st.cache_data(show_spinner=False)
def get_cached_zone_metadata():
    return load_zone_metadata()


@st.cache_data(show_spinner=False)
def get_cached_benchmark(zone_name: str, eval_horizon: int):
    df_vol, _ = get_cached_volume_data()
    series_df = get_series_for_zone(df_vol, zone_name)
    return run_benchmark_suite(series_df, eval_horizon=eval_horizon)


@st.cache_data(show_spinner=False)
def get_cached_forecast(zone_name: str, model_name: str, horizon: int):
    df_vol, _ = get_cached_volume_data()
    series_df = get_series_for_zone(df_vol, zone_name)
    return generate_future_forecast(series_df, model_name=model_name, horizon_hours=horizon)


# Sidebar Navigation & Global Controls
with st.sidebar:
    st.markdown("""
    <div class="evision-brand">
        <div class="evision-logo">⚡</div>
        <h2 class="evision-title">EVision</h2>
    </div>
    <div class="evision-subtitle">Intelligent EV Charging Analytics</div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation Menu",
        [
            "Overview Dashboard",
            "Data Exploration",
            "Time Series Analysis",
            "Demand Forecasting",
            "Model Comparison",
            "Peak Demand Alerts",
            "About the Project",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎛️ Analysis Controls")

    # Load data dynamically
    try:
        df_volume, data_stats = get_cached_volume_data()
        available_zones = get_zone_list(df_volume)
        zone_options = ["All Zones (Aggregated)"] + available_zones
    except Exception as e:
        st.error(f"Error loading UrbanEV dataset: {e}")
        st.stop()

    selected_zone_option = st.selectbox(
        "Select Charging Zone",
        zone_options,
        index=0,
        help="Select a specific zone or 'All Zones (Aggregated)' for city-scale load.",
    )
    zone_query = "All Zones" if "All Zones" in selected_zone_option else selected_zone_option

    selected_horizon = st.select_slider(
        "Forecasting Horizon",
        options=AVAILABLE_HORIZONS,
        value=DEFAULT_HORIZON,
        format_func=lambda h: f"{h} Hours",
        help="Select lookahead horizon (6 to 48 hours ahead).",
    )

    selected_model = st.selectbox(
        "Forecasting Algorithm",
        CORE_MODELS,
        index=4,  # default to XGBoost Regressor
    )

    st.markdown("---")
    st.caption("UrbanEV Benchmark • Shenzhen, China")
    st.caption("181 Days • 4,344 Hourly Intervals")

    # Optional Custom Data Upload expander
    with st.expander("📤 Upload Custom CSV/ZIP"):
        uploaded_file = st.file_uploader("Upload volume.csv or .zip", type=["csv", "zip"])
        if uploaded_file is not None:
            try:
                buffer = BytesIO(uploaded_file.read())
                df_custom, custom_stats = load_volume_data(buffer)
                st.success(f"Loaded custom dataset with {len(df_custom)} rows & {custom_stats['zone_count']} zones!")
                df_volume = df_custom
            except Exception as ex:
                st.error(f"Failed to parse uploaded dataset: {ex}")


# Header & Limitation Banner
st.markdown(f"""
<div class="disclaimer-box">
    <b>⚡ System Operational Notice:</b> {DATA_LIMITATION_NOTICE}
</div>
""", unsafe_allow_html=True)


# Extract current zone series & summary
zone_series_df = get_series_for_zone(df_volume, zone_query)
zone_meta_df = get_cached_zone_metadata()
summary_stats = compute_dataset_summary(df_volume, zone_meta=zone_meta_df)
thresholds = calculate_zone_thresholds(zone_series_df["demand"])


# 1. OVERVIEW DASHBOARD
if nav_selection == "Overview Dashboard":
    st.markdown(f"### 📊 Operational Overview — {selected_zone_option}")
    st.markdown(f"**Date Coverage:** `{data_stats['date_start']}` to `{data_stats['date_end']}` ({data_stats['final_rows']} continuous hourly records)")

    # KPI Top Metrics Cards
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    with kpi1:
        st.markdown(rf"""
        <div class="kpi-card">
            <div class="kpi-label">Active Zones</div>
            <div class="kpi-value">{summary_stats['number_of_zones']}</div>
            <div class="kpi-sub">Shenzhen Metro</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(rf"""
        <div class="kpi-card">
            <div class="kpi-label">Hourly Observations</div>
            <div class="kpi-value">{summary_stats['total_observations']:,}</div>
            <div class="kpi-sub">Sep 2022 – Feb 2023</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        avg_display = zone_series_df["demand"].mean()
        st.markdown(rf"""
        <div class="kpi-card">
            <div class="kpi-label">Average Demand</div>
            <div class="kpi-value">{avg_display:.1f} <span style="font-size:14px;color:#8E9EB5;">kWh</span></div>
            <div class="kpi-sub">Hourly mean load</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        max_display = zone_series_df["demand"].max()
        st.markdown(rf"""
        <div class="kpi-card">
            <div class="kpi-label">Max Observed</div>
            <div class="kpi-value">{max_display:.1f} <span style="font-size:14px;color:#8E9EB5;">kWh</span></div>
            <div class="kpi-sub">Historical peak record</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi5:
        # Generate quick forecast for next hour
        with st.spinner("Generating 24h operational projection..."):
            quick_fc = get_cached_forecast(zone_query, selected_model, selected_horizon)
            next_hour_pred = quick_fc["forecast_df"]["predicted_demand_kwh"].iloc[0]
            st.markdown(rf"""
            <div class="kpi-card">
                <div class="kpi-label">Next Hour Forecast</div>
                <div class="kpi-value">{next_hour_pred:.1f} <span style="font-size:14px;color:#8E9EB5;">kWh</span></div>
                <div class="kpi-sub">T+1 projection ({selected_model.split()[0]})</div>
            </div>
            """, unsafe_allow_html=True)

    with kpi6:
        st.markdown(rf"""
        <div class="kpi-card">
            <div class="kpi-label">Expected Peak</div>
            <div class="kpi-value">{quick_fc['peak_demand_kwh']:.1f} <span style="font-size:14px;color:#8E9EB5;">kWh</span></div>
            <div class="kpi-sub">At {quick_fc['peak_time'].strftime('%H:00')} ({quick_fc['peak_time'].strftime('%b %d')})</div>
        </div>
        """, unsafe_allow_html=True)

    # Forecast vs Recent Chart
    st.markdown("#### ⚡ 24-48 Hour Lookahead Projection & High Demand Thresholds")
    fig_fc = plot_future_forecast_with_alerts(quick_fc, thresholds, zone_label=selected_zone_option)
    st.plotly_chart(fig_fc, use_container_width=True)

    # Historical overview & Zone breakdown side by side
    col_left, col_right = st.columns([1.6, 1.0])

    with col_left:
        st.markdown("#### 📈 Historical Charging Demand Profile")
        fig_hist = plot_historical_demand(zone_series_df, zone_label=selected_zone_option, show_slider=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.markdown("#### 🏆 Top 10 Charging Zones by Volume")
        fig_top = plot_top_zones_bar(summary_stats["top_10_zones"], title="Top 10 Highest Average Demand Zones")
        st.plotly_chart(fig_top, use_container_width=True)


# 2. DATA EXPLORATION
elif nav_selection == "Data Exploration":
    st.markdown(f"### 🔍 UrbanEV Exploratory Data Analysis")
    st.markdown("Inspect granular time-series profiles, administrative zone distributions, and infrastructure features.")

    tab_temp, tab_zones, tab_weather, tab_geo = st.tabs([
        "Temporal Patterns",
        "Zone Distributions & Comparisons",
        "Weather & Environmental Correlation",
        "Infrastructure & Geospatial Distribution",
    ])

    with tab_temp:
        patterns = compute_temporal_patterns(zone_series_df)
        p_figs = plot_temporal_patterns(patterns)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(p_figs["hourly"], use_container_width=True)
            st.info("💡 **Diurnal Insight:** EV charging demand follows a distinct bimodal peak pattern: a midday peak corresponding to lunch/commercial driver top-ups, and a late-evening surge as private commuters plug in overnight.")
        with c2:
            st.plotly_chart(p_figs["weekday_vs_weekend"], use_container_width=True)
            st.info("💡 **Weekday vs Weekend:** Weekdays exhibit sharper commuting morning and evening peaks, whereas weekends show a smoother, midday leisure charging distribution.")

        st.plotly_chart(p_figs["day_of_week"], use_container_width=True)

    with tab_zones:
        st.markdown("#### Zone-Wise Charging Demand Comparison")
        ranking_df = summary_stats["zone_ranking"]
        st.dataframe(
            ranking_df[["zone_id", "mean_demand_kwh", "max_demand_kwh", "std_demand_kwh", "station_count", "pile_count"]],
            use_container_width=True,
            height=300,
        )

        st.markdown("#### Multi-Zone Head-to-Head Comparison")
        compare_zones = st.multiselect(
            "Select 2 to 5 zones to compare over time:",
            available_zones,
            default=available_zones[:3],
        )
        if compare_zones:
            df_slice = df_volume[["time"] + compare_zones].iloc[-336:]  # last 2 weeks
            fig_multi = plot_historical_demand(
                get_series_for_zone(df_slice, compare_zones[0]),
                zone_label=f"Zone {compare_zones[0]}",
                show_slider=False,
            )
            for z in compare_zones[1:]:
                s = get_series_for_zone(df_slice, z)
                fig_multi.add_scatter(x=s["time"], y=s["demand"], name=f"Zone {z}", mode="lines")
            fig_multi.update_layout(title="<b>Multi-Zone Demand Trajectory (Past 14 Days)</b>")
            st.plotly_chart(fig_multi, use_container_width=True)

    with tab_weather:
        st.markdown("#### Correlation with Central Meteorological Observations")
        df_weather = get_cached_weather_data()
        if df_weather is not None:
            fig_weather = plot_weather_correlation(zone_series_df, df_weather)
            if fig_weather:
                st.plotly_chart(fig_weather, use_container_width=True)
                st.markdown("""
                **Meteorological Variables Dictionary:**
                - `temperature_c`: 2-meter air temperature (°C)
                - `humidity_pct`: Relative atmospheric humidity (%)
                - `rain_level`: Precipitation category (0=Normal, 1=Light, 2=Moderate, 3=Heavy)
                - `pressure_station_mmhg`: Atmospheric pressure (mmHg)
                """)
            else:
                st.info("Weather data alignment requires corresponding timestamps.")
        else:
            st.warning("Weather dataset not available.")

    with tab_geo:
        st.markdown("#### Charging Station & Pile Density Across Shenzhen")
        if zone_meta_df is not None:
            fig_map = plot_zone_map(zone_meta_df)
            if fig_map:
                st.plotly_chart(fig_map, use_container_width=True)
                st.caption("Circle size represents charging pile count; color scale represents station count per zone.")
        else:
            st.info("Geographic coordinates from inf.csv are loading.")


# 3. TIME SERIES ANALYSIS
elif nav_selection == "Time Series Analysis":
    st.markdown(f"### 📈 Academic Time Series Diagnostics — {selected_zone_option}")
    st.markdown("Rigorous econometric and statistical diagnostics: Trend, Seasonality, Stationarity (ADF), and Autoregressive Structure.")

    col_adf, col_diff = st.columns([1.2, 1.8])

    with col_adf:
        st.markdown("#### 🔬 Augmented Dickey-Fuller (ADF) Test")
        adf_res = run_adf_test(zone_series_df["demand"])

        status_class = "badge-normal" if adf_res["is_stationary"] else "badge-critical"
        st.markdown(rf"""
        <div style="background:#0F1C3F;padding:15px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);">
            <div style="margin-bottom:8px;">Verdict: <span class="{status_class}">{adf_res['verdict']}</span></div>
            <div><b>ADF Test Statistic:</b> <code>{adf_res['test_statistic']:.4f}</code></div>
            <div><b>p-value:</b> <code>{adf_res['p_value']:.4e}</code></div>
            <div><b>Lags Used:</b> {adf_res['used_lag']} hours</div>
            <div><b>Observations:</b> {adf_res['n_obs']:,}</div>
            <div style="margin-top:8px;font-size:11px;color:#8E9EB5;">Critical Values: 1%: {adf_res['critical_values']['1%']:.2f} | 5%: {adf_res['critical_values']['5%']:.2f} | 10%: {adf_res['critical_values']['10%']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"**Interpretation:** {adf_res['interpretation']}")

    with col_diff:
        st.markdown("#### 🔄 Differencing & Transformation Diagnostics")
        diff_res = compute_differencing_analysis(zone_series_df["demand"])
        d_table = pd.DataFrame([
            {
                "Series Version": "Original Level (y_t)",
                "ADF Statistic": f"{diff_res['original_adf']['test_statistic']:.3f}",
                "p-value": f"{diff_res['original_adf']['p_value']:.4f}",
                "Stationary?": diff_res['original_adf']['verdict'],
            },
            {
                "Series Version": "First Difference (Δ y_t)",
                "ADF Statistic": f"{diff_res['first_diff_adf']['test_statistic']:.3f}",
                "p-value": f"{diff_res['first_diff_adf']['p_value']:.4e}",
                "Stationary?": diff_res['first_diff_adf']['verdict'],
            },
            {
                "Series Version": "Seasonal Difference (Δ₂₄ y_t)",
                "ADF Statistic": f"{diff_res['seasonal_diff_adf']['test_statistic']:.3f}",
                "p-value": f"{diff_res['seasonal_diff_adf']['p_value']:.4e}",
                "Stationary?": diff_res['seasonal_diff_adf']['verdict'],
            },
        ])
        st.dataframe(d_table, use_container_width=True, hide_index=True)
        st.caption("First-order differencing strips non-stationary trend drifts; seasonal differencing (period=24) eliminates cyclical daily drift.")

    st.markdown("---")
    st.markdown("#### 📊 Rolling Statistics (Mean & Standard Deviation)")
    df_roll = compute_rolling_statistics(zone_series_df.set_index("time")["demand"])
    fig_roll = plot_rolling_diagnostics(df_roll, zone_label=selected_zone_option)
    st.plotly_chart(fig_roll, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🧩 Classical Time Series Decomposition (24-Hour Cycle)")
    decomp_res = decompose_time_series(zone_series_df, period=24, model="additive")
    fig_decomp = plot_decomposition(decomp_res)
    st.plotly_chart(fig_decomp, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📐 Autocorrelation (ACF) & Partial Autocorrelation (PACF)")
    acf_pacf_res = compute_acf_pacf(zone_series_df["demand"], nlags=72)
    fig_acf = plot_acf_pacf(acf_pacf_res)
    st.plotly_chart(fig_acf, use_container_width=True)
    st.info("💡 **Model Guidance:** Strong spikes at lag 24, 48, and 72 in the ACF confirm strong diurnal seasonality. The PACF sharp drop after lag 1-2 supports AR(1) or AR(2) autoregressive specifications combined with seasonal differencing.")


# 4. DEMAND FORECASTING
elif nav_selection == "Demand Forecasting":
    st.markdown(f"### 🔮 Predictive Demand Forecasting Engine")
    st.markdown(f"Generate forward-looking predictions for **{selected_zone_option}** across user-defined horizons.")

    c_btn, c_info = st.columns([1, 3])
    with c_btn:
        forecast_clicked = st.button("🚀 Forecast Demand", type="primary", use_container_width=True)

    with st.spinner(f"Computing {selected_horizon}-hour ahead forecast using {selected_model}..."):
        fc_data = get_cached_forecast(zone_query, selected_model, selected_horizon)
        alert_eval = evaluate_peak_alerts(fc_data["forecast_df"], thresholds, zone_name=selected_zone_option)

    # Summary Cards
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(rf"""
        <div class="kpi-card">
            <div class="kpi-label">Forecasting Horizon</div>
            <div class="kpi-value">{selected_horizon} <span style="font-size:14px;color:#8E9EB5;">Hours</span></div>
            <div class="kpi-sub">Step size: 1-hour increments</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(rf"""
        <div class="kpi-card">
            <div class="kpi-label">Selected Model</div>
            <div class="kpi-value" style="font-size:18px;">{selected_model}</div>
            <div class="kpi-sub">Deterministic inference</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown(rf"""
        <div class="kpi-card">
            <div class="kpi-label">Peak Demand Window</div>
            <div class="kpi-value">{fc_data['peak_demand_kwh']:.1f} <span style="font-size:14px;color:#8E9EB5;">kWh</span></div>
            <div class="kpi-sub">At {fc_data['peak_time_str']}</div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown(rf"""
        <div class="kpi-card">
            <div class="kpi-label">Total Expected Energy</div>
            <div class="kpi-value">{fc_data['total_energy_kwh']:.0f} <span style="font-size:14px;color:#8E9EB5;">kWh</span></div>
            <div class="kpi-sub">Sum over {selected_horizon}h horizon</div>
        </div>
        """, unsafe_allow_html=True)

    # Forecast Graph
    st.plotly_chart(plot_future_forecast_with_alerts(fc_data, thresholds, zone_label=selected_zone_option), use_container_width=True)

    # Detailed Forecast Table & Export
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.markdown("#### 📋 Hourly Forecast Schedule")
        display_df = alert_eval["annotated_forecast_df"][["time", "hour_of_day", "day_of_week", "predicted_demand_kwh", "status_badge"]]
        st.dataframe(display_df, use_container_width=True, height=280)

    with col_t2:
        st.markdown("#### 📥 Export Predictions")
        csv_buffer = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Forecast as CSV",
            data=csv_buffer,
            file_name=f"evision_forecast_{zone_query}_{selected_horizon}h_{selected_model.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(rf"""
        <div style="background:#0F1C3F;padding:12px;border-radius:8px;border:1px solid rgba(255,255,255,0.08);margin-top:10px;font-size:12px;">
            <b>Zone Thresholds:</b><br>
            • Normal: &lt; {thresholds['p90']:.1f} kWh<br>
            • High Demand (&gt;90th %ile): {thresholds['p90']:.1f} kWh<br>
            • Critical Peak (&gt;95th %ile): {thresholds['p95']:.1f} kWh
        </div>
        """, unsafe_allow_html=True)


# 5. MODEL COMPARISON
elif nav_selection == "Model Comparison":
    st.markdown(f"### 🏆 Forecasting Models Benchmarking & Comparison")
    st.markdown("Rigorous chronological evaluation using a **70% Train, 15% Validation, 15% Test** split. Models are ranked based on Validation Mean Absolute Error (MAE).")

    with st.spinner(f"Evaluating all 5 forecasting models on {selected_zone_option}..."):
        bench_data = get_cached_benchmark(zone_query, selected_horizon)

    best_name = bench_data["best_model_name"]
    st.success(f"🥇 **Best Performing Model:** **{best_name}** achieved the lowest validation MAE on {selected_zone_option}!")

    # Comparative Metric Leaderboards
    tab_val, tab_test = st.tabs(["Validation Set Leaderboard (Model Selection)", "Held-Out Test Set Leaderboard (Generalization)"])
    with tab_val:
        st.dataframe(bench_data["val_leaderboard"], use_container_width=True, hide_index=True)
    with tab_test:
        st.dataframe(bench_data["test_leaderboard"], use_container_width=True, hide_index=True)

    # Bar chart of errors
    st.plotly_chart(plot_model_metrics_bar(bench_data["val_leaderboard"]), use_container_width=True)

    # Actual vs Predictions Plot
    st.markdown("#### 🔬 Ground Truth vs. Model Predictions Overlay")
    fig_comp_curves = plot_model_comparison_curves(
        actual_times=bench_data["val_times"],
        actual_values=bench_data["val_actual"],
        predictions_dict=bench_data["val_predictions"],
        split_name="Validation Set",
    )
    st.plotly_chart(fig_comp_curves, use_container_width=True)


# 6. PEAK DEMAND ALERTS
elif nav_selection == "Peak Demand Alerts":
    st.markdown(f"### 🚨 High-Demand Alert & Operations Center")
    st.markdown("Proactive identification of elevated charging loads to optimize grid integration, storage dispatch, and station operations.")

    with st.spinner("Analyzing peak load alerts..."):
        fc_data = get_cached_forecast(zone_query, selected_model, selected_horizon)
        alert_eval = evaluate_peak_alerts(fc_data["forecast_df"], thresholds, zone_name=selected_zone_option)

    # Alert status header
    if alert_eval["critical_hours_count"] > 0:
        st.markdown(rf"""
        <div style="background:rgba(255,77,109,0.15);border:1px solid #FF4D6D;border-radius:8px;padding:16px;margin-bottom:20px;">
            <h3 style="color:#FF4D6D;margin:0 0 8px 0;">🚨 CRITICAL PEAK ALERT DETECTED</h3>
            <div>{alert_eval['critical_hours_count']} upcoming hour(s) exceed the 95th percentile historical threshold ({thresholds['p95']:.1f} kWh). Peak expected at <b>{alert_eval['peak_time_str']}</b> with demand of <b>{alert_eval['peak_demand_kwh']:.1f} kWh</b>.</div>
        </div>
        """, unsafe_allow_html=True)
    elif alert_eval["high_demand_hours_count"] > 0:
        st.markdown(rf"""
        <div style="background:rgba(255,190,11,0.15);border:1px solid #FFBE0B;border-radius:8px;padding:16px;margin-bottom:20px;">
            <h3 style="color:#FFBE0B;margin:0 0 8px 0;">⚠️ HIGH DEMAND ALERT</h3>
            <div>{alert_eval['high_demand_hours_count']} hour(s) exceed the 90th percentile threshold ({thresholds['p90']:.1f} kWh). Pre-emptive load shifting recommended.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(rf"""
        <div style="background:rgba(6,214,160,0.15);border:1px solid #06D6A0;border-radius:8px;padding:16px;margin-bottom:20px;">
            <h3 style="color:#06D6A0;margin:0 0 8px 0;">✅ NORMAL DEMAND PROJECTED</h3>
            <div>All forecasted hours fall comfortably within baseline charging tolerances (&lt; {thresholds['p90']:.1f} kWh).</div>
        </div>
        """, unsafe_allow_html=True)

    # Operational Recommendations
    st.markdown("#### 🛠️ Recommended Operational Interventions")
    for rec in alert_eval["recommendations"]:
        badge_style = "badge-critical" if rec["priority"] in ["CRITICAL", "HIGH"] else "badge-normal"
        st.markdown(rf"""
        <div style="background:#0F1C3F;border-left:4px solid #00D2FF;border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <b style="font-size:14px;color:#FFFFFF;">{rec['title']}</b>
                <span class="{badge_style}">{rec['priority']} PRIORITY</span>
            </div>
            <div style="font-size:11px;color:#8E9EB5;margin-bottom:6px;">Category: {rec['category']}</div>
            <div style="font-size:13px;color:#CBD5E0;">{rec['action']}</div>
        </div>
        """, unsafe_allow_html=True)


# 7. ABOUT THE PROJECT
elif nav_selection == "About the Project":
    st.markdown("### 📖 About EVision & Dataset Documentation")

    st.markdown(rf"""
    **EVision** is an intelligent, production-grade electric vehicle charging demand forecasting and peak load prediction platform designed for utility operators, charging network operators (CPOs), and municipal energy planners.

    #### 📚 Research Dataset Attribution
    This application utilizes the official **UrbanEV** benchmark dataset:
    - **Title:** *UrbanEV: A Benchmark Dataset for City-Scale Electric Vehicle Charging Demand Prediction*
    - **Authors & Lab:** Intelligent Systems Lab (ISLab)
    - **Published in:** *Nature Scientific Data* (2025)
    - **DOI / Paper:** [{URBANEV_PAPER_URL}]({URBANEV_PAPER_URL})
    - **Official GitHub:** [{URBANEV_REPO_URL}]({URBANEV_REPO_URL})
    - **Dryad Repository:** [{URBANEV_DRYAD_URL}]({URBANEV_DRYAD_URL})
    - **License:** Creative Commons Zero (CC0 1.0 Universal)

    #### 🗺️ Dataset Scope
    - **Geographic Location:** Shenzhen, China
    - **Observation Window:** September 1, 2022 to February 28, 2023 (6 continuous months)
    - **Granularity:** Hourly time-series intervals (4,344 timestamps)
    - **Coverage:** 275 administrative Traffic Analysis Zones (TAZs)
    - **Cleaned Infrastructure:** 1,362 public charging stations and 17,532 charging piles

    #### 🤖 Implemented Forecasting Models
    1. **Naive (Persistence) Forecaster:** Establishes the baseline by assuming $\hat{{y}}_{{t+h}} = y_t$.
    2. **Seasonal Naive (24h) Forecaster:** Repeats the observed diurnal cycle from exactly 24 hours prior ($\hat{{y}}_{{t+h}} = y_{{t+h-24}}$).
    3. **ARIMA (Autoregressive Integrated Moving Average):** Classic statistical model with $(p, d, q)$ tuning for level-trend correction.
    4. **SARIMAX (Seasonal ARIMA):** Captures 24-hour diurnal seasonal autocorrelation with state-space estimation.
    5. **XGBoost Regressor:** Gradient-boosted decision trees trained on lag features (1, 2, 3, 24, 48, 168), rolling window statistics (shifted to prevent leakage), calendar markers, and continuous cyclical sine/cosine encodings.

    #### ⚠️ Data Engineering & Electrical Grid Notice
    > **Crucial Academic Caveat:** In the official UrbanEV dataset, the primary target file `volume.csv` reports charging volume in kWh estimated from charging pile rated power multiplied by charging duration. Actual metered power utility consumption may differ due to battery State-of-Charge (SoC) charging taper curves, ambient temperature, and conversion losses.
    > 
    > Furthermore, because the public dataset does not contain physical substation feeder capacities or transformer ratings, EVision uses the terminology **HIGH DEMAND ALERT** (benchmarked against historical 90th and 95th percentiles) rather than **GRID OVERLOAD**.
    """)
