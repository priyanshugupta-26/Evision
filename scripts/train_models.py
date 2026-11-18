"""
Model Training & Benchmarking Script for EVision.
Evaluates all 5 forecasting models on sample zones and All Zones combined,
generating leaderboard metrics and ensuring reproducible modeling.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evision.data_loader import load_volume_data
from evision.preprocessing import get_series_for_zone
from evision.forecasting import run_benchmark_suite, generate_future_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EVision.TrainModels")


def train_and_benchmark(zones_to_test=["102", "All Zones"]):
    logger.info("Loading dataset for offline model benchmarking...")
    df_vol, _ = load_volume_data()

    for zone in zones_to_test:
        print(f"\n==========================================")
        print(f"  BENCHMARK SUITE: {zone.upper()}")
        print(f"==========================================")
        series_df = get_series_for_zone(df_vol, zone)
        print(f"Series length: {len(series_df)} hours")
        print(f"Date range: {series_df['time'].min()} to {series_df['time'].max()}")

        bench_results = run_benchmark_suite(series_df, eval_horizon=24)

        print("\n--- Validation Leaderboard (Model Selection) ---")
        print(bench_results["val_leaderboard"].to_string(index=False))

        print("\n--- Test Set Leaderboard (Generalization) ---")
        print(bench_results["test_leaderboard"].to_string(index=False))

        print(f"\n🥇 Winning Model for {zone}: {bench_results['best_model_name']}")

        # Test future projection
        fc = generate_future_forecast(series_df, bench_results['best_model_name'], horizon_hours=24)
        print(f"24h Projected Peak: {fc['peak_demand_kwh']:.2f} kWh at {fc['peak_time_str']}")
        print(f"24h Total Projected Energy: {fc['total_energy_kwh']:.1f} kWh")

    print("\n✅ Model benchmarking script completed successfully!")


if __name__ == "__main__":
    train_and_benchmark()
