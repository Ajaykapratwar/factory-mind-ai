"""
tests/test_metrics.py — Step 1.3 verification
Unit tests for evaluation/metrics.py against hand-computed values on a 5-row synthetic array.
Also tests check_acceptance() with a deliberately-bad forecast.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from evaluation.metrics import (
    compute_mae, compute_mse, compute_rmse,
    compute_mape, compute_smape, compute_wape,
    compute_r2, compute_rmse_pct, compute_mae_pct,
    compute_bias_pct, compute_pinball_loss,
    compute_regression_metrics, check_acceptance,
)

# ---------------------------------------------------------------------------
# 5-row synthetic arrays (hand-computed ground truth)
# ---------------------------------------------------------------------------
Y_TRUE = np.array([100.0, 200.0, 150.0, 80.0, 120.0])
Y_PRED = np.array([ 90.0, 210.0, 140.0, 95.0, 130.0])

# Hand-computed values:
# errors:         [-10,  10, -10,  15,  10]   (pred - true)
# abs errors:     [ 10,  10,  10,  15,  10]
# sq errors:      [100, 100, 100, 225, 100]
# MAE  = (10+10+10+15+10)/5 = 55/5 = 11.0
# MSE  = (100+100+100+225+100)/5 = 625/5 = 125.0
# RMSE = sqrt(125) = 11.180339...
# MAPE = mean(|err|/|true|)*100 = mean([0.10,0.05,0.0667,0.1875,0.0833])*100
#       = (0.10+0.05+0.0667+0.1875+0.0833)/5*100 = 0.4975/5*100 = 9.95
# WAPE = sum(abs err)/sum(true)*100 = 55/650*100 = 8.4615...
# R² : SS_res = 625, mean_true = 130, SS_tot = ((100-130)²+(200-130)²+(150-130)²+(80-130)²+(120-130)²)
#           = 900+4900+400+2500+100 = 8800
#       R² = 1 - 625/8800 = 0.92898...
# bias% = sum(pred-true)/sum(true)*100 = (−10+10−10+15+10)/650*100 = 15/650*100 = 2.3077...


class TestMAE:
    def test_value(self):
        result = compute_mae(Y_TRUE, Y_PRED)
        assert abs(result - 11.0) < 1e-6, f"MAE expected 11.0, got {result}"


class TestMSE:
    def test_value(self):
        result = compute_mse(Y_TRUE, Y_PRED)
        assert abs(result - 125.0) < 1e-6, f"MSE expected 125.0, got {result}"


class TestRMSE:
    def test_value(self):
        result = compute_rmse(Y_TRUE, Y_PRED)
        assert abs(result - np.sqrt(125.0)) < 1e-4, f"RMSE expected {np.sqrt(125.0):.4f}, got {result}"


class TestMAPE:
    def test_value(self):
        result = compute_mape(Y_TRUE, Y_PRED)
        expected = np.mean(np.abs((Y_TRUE - Y_PRED) / Y_TRUE)) * 100
        assert abs(result - expected) < 1e-4, f"MAPE expected {expected:.4f}, got {result}"

    def test_zero_denominator(self):
        y_t = np.array([0.0, 100.0])
        y_p = np.array([5.0, 110.0])
        result = compute_mape(y_t, y_p)
        # Should only use row where y_true != 0
        assert abs(result - 10.0) < 1e-4, f"MAPE with zero: expected 10.0, got {result}"


class TestSMAPE:
    def test_value(self):
        result = compute_smape(Y_TRUE, Y_PRED)
        denom = np.abs(Y_TRUE) + np.abs(Y_PRED)
        expected = np.mean(2 * np.abs(Y_PRED - Y_TRUE) / denom) * 100
        assert abs(result - expected) < 1e-4, f"SMAPE expected {expected:.4f}, got {result}"

    def test_bounded(self):
        # SMAPE should be between 0 and 200
        assert 0 <= compute_smape(Y_TRUE, Y_PRED) <= 200


class TestWAPE:
    def test_value(self):
        result = compute_wape(Y_TRUE, Y_PRED)
        expected = np.sum(np.abs(Y_TRUE - Y_PRED)) / np.sum(np.abs(Y_TRUE)) * 100
        assert abs(result - expected) < 1e-4, f"WAPE expected {expected:.4f}, got {result}"

    def test_zero_actuals(self):
        y_t = np.array([0.0, 0.0])
        y_p = np.array([1.0, 1.0])
        result = compute_wape(y_t, y_p)
        assert np.isnan(result), "WAPE with zero actuals should be NaN"


class TestR2:
    def test_value(self):
        result = compute_r2(Y_TRUE, Y_PRED)
        ss_res = np.sum((Y_TRUE - Y_PRED) ** 2)
        ss_tot = np.sum((Y_TRUE - np.mean(Y_TRUE)) ** 2)
        expected = 1 - ss_res / ss_tot
        assert abs(result - expected) < 1e-6, f"R² expected {expected:.6f}, got {result}"

    def test_can_be_negative(self):
        """Bad model → R² can be negative; do NOT clip."""
        y_t = np.array([1.0, 2.0, 3.0])
        y_p = np.array([100.0, 200.0, 300.0])
        assert compute_r2(y_t, y_p) < 0


class TestRMSEPct:
    def test_value(self):
        result = compute_rmse_pct(Y_TRUE, Y_PRED)
        expected = compute_rmse(Y_TRUE, Y_PRED) / np.mean(np.abs(Y_TRUE)) * 100
        assert abs(result - expected) < 1e-4


class TestMAEPct:
    def test_value(self):
        result = compute_mae_pct(Y_TRUE, Y_PRED)
        expected = compute_mae(Y_TRUE, Y_PRED) / np.mean(np.abs(Y_TRUE)) * 100
        assert abs(result - expected) < 1e-4


class TestBias:
    def test_positive_bias(self):
        """Over-forecast → positive bias."""
        y_t = np.array([100.0])
        y_p = np.array([120.0])
        result = compute_bias_pct(y_t, y_p)
        assert result > 0, "Over-forecast should give positive bias"

    def test_negative_bias(self):
        """Under-forecast → negative bias."""
        y_t = np.array([100.0])
        y_p = np.array([80.0])
        result = compute_bias_pct(y_t, y_p)
        assert result < 0, "Under-forecast should give negative bias"

    def test_value(self):
        result = compute_bias_pct(Y_TRUE, Y_PRED)
        expected = np.sum(Y_PRED - Y_TRUE) / np.sum(np.abs(Y_TRUE)) * 100
        assert abs(result - expected) < 1e-4


class TestPinball:
    def test_p50_equals_half_mae(self):
        """Pinball at q=0.5 equals MAE/2."""
        result = compute_pinball_loss(Y_TRUE, Y_PRED, q=0.5)
        mae_half = compute_mae(Y_TRUE, Y_PRED) / 2
        assert abs(result - mae_half) < 1e-6, f"Pinball q=0.5 should equal MAE/2: {mae_half}"


class TestComputeRegressionMetrics:
    def test_returns_all_keys(self):
        m = compute_regression_metrics(Y_TRUE, Y_PRED, label="test")
        expected_keys = ["label", "n", "mae", "mse", "rmse", "mape", "smape",
                         "wape", "r2", "rmse_pct", "mae_pct", "bias_pct", "pinball_50"]
        for k in expected_keys:
            assert k in m, f"Key '{k}' missing from compute_regression_metrics output"


class TestCheckAcceptance:
    def test_good_forecast_passes(self):
        """A near-perfect forecast should pass all thresholds."""
        good_metrics = {
            "r2": 0.95,
            "mape": 5.0,
            "smape": 5.0,
            "wape": 5.0,
            "bias_pct": 1.0,
            "rmse_pct": 8.0,
            "mae_pct": 6.0,
        }
        result = check_acceptance(good_metrics)
        assert result["all_pass"] is True, f"Good forecast failed: {result}"

    def test_bad_forecast_fails(self):
        """Deliberately-bad forecast must be flagged as failing."""
        bad_metrics = {
            "r2": -0.5,       # below 0.55
            "mape": 80.0,     # above 25.0
            "smape": 70.0,    # above 25.0
            "wape": 60.0,     # above 20.0
            "bias_pct": 30.0, # |bias| above 8.0
            "rmse_pct": 90.0, # above 25.0
            "mae_pct": 75.0,  # above 20.0
        }
        result = check_acceptance(bad_metrics)
        assert result["all_pass"] is False, "Bad forecast should fail acceptance check"
        # Each metric individually should fail
        assert not result["r2"]["pass"]
        assert not result["mape"]["pass"]


if __name__ == "__main__":
    # Allow running directly without pytest
    import unittest
    # Just run pytest programmatically
    import pytest as pt
    pt.main([__file__, "-v"])
