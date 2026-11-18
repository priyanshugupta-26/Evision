"""
Unit tests for evaluation module.
"""

import numpy as np
import pytest

from evision.evaluation import (
    calculate_filtered_mape,
    calculate_smape,
    compare_models,
    evaluate_forecast,
)


def test_calculate_smape():
    y_true = np.array([100.0, 50.0, 0.0, 10.0])
    y_pred = np.array([100.0, 60.0, 0.0, 15.0])

    smape = calculate_smape(y_true, y_pred)
    assert 0.0 <= smape <= 200.0

    # Perfect forecast should yield 0%
    assert calculate_smape(np.array([10.0, 20.0]), np.array([10.0, 20.0])) == pytest.approx(0.0, abs=1e-3)


def test_calculate_filtered_mape():
    # Contains zeros
    y_true = np.array([0.0, 0.2, 10.0, 20.0])
    y_pred = np.array([5.0, 0.5, 12.0, 18.0])

    mape, count = calculate_filtered_mape(y_true, y_pred, threshold=0.5)
    assert count == 2  # only 10.0 and 20.0
    assert mape is not None
    # 10 vs 12 is 20%, 20 vs 18 is 10% -> mean = 15%
    assert mape == pytest.approx(15.0, abs=1e-2)


def test_evaluate_forecast():
    y_t = np.array([10.0, 20.0, 30.0, 40.0])
    y_p = np.array([12.0, 18.0, 33.0, 38.0])

    res = evaluate_forecast(y_t, y_p, model_name="TestModel")
    assert res["model_name"] == "TestModel"
    assert res["mae"] == pytest.approx(2.25, abs=1e-2)
    assert res["rmse"] > 0
    assert res["r2"] > 0.8
    assert res["n_samples"] == 4


def test_compare_models():
    eval1 = {"model_name": "ModelA", "mae": 5.0, "rmse": 7.0, "smape": 15.0, "mape": 14.0, "n_samples": 50, "r2": 0.85}
    eval2 = {"model_name": "ModelB", "mae": 3.0, "rmse": 4.5, "smape": 10.0, "mape": 9.5, "n_samples": 50, "r2": 0.92}

    leaderboard = compare_models([eval1, eval2])
    assert len(leaderboard) == 2
    # ModelB has lower MAE so it should be ranked #1
    assert leaderboard.iloc[0]["Model"] == "ModelB"
    assert leaderboard.iloc[1]["Model"] == "ModelA"
