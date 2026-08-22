"""
evaluation/metrics.py — Step 1.3
Full pharma forecast metric suite per spec §5.

All functions accept (y_true, y_pred, label="") and return flat dicts.
Division by zero is guarded throughout.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

EPSILON = 1e-8  # guard against zero-demand division


# 5.1 Point-forecast accuracy metrics

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Excludes rows where y_true ≈ 0 (< EPSILON) to avoid instability.
    """
    mask = np.abs(y_true) > EPSILON
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE — bounded [0, 200%], stable on low-volume SKUs.
    """
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > EPSILON
    if not mask.any():
        return 0.0
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Volume-Weighted APE — robust to zero-demand rows.
    WAPE % = sum(|actual − pred|) / sum(|actual|) × 100
    """
    total_actual = np.sum(np.abs(y_true))
    if total_actual < EPSILON:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Can be negative. Never clipped."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < EPSILON:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_rmse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Scale-normalized RMSE: RMSE / mean(y_true) × 100."""
    mean_actual = np.mean(np.abs(y_true))
    if mean_actual < EPSILON:
        return np.nan
    return float(compute_rmse(y_true, y_pred) / mean_actual * 100)


def compute_mae_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Scale-normalized MAE: MAE / mean(y_true) × 100."""
    mean_actual = np.mean(np.abs(y_true))
    if mean_actual < EPSILON:
        return np.nan
    return float(compute_mae(y_true, y_pred) / mean_actual * 100)


# 5.2 Bias metrics

def compute_bias_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Bias % = sum(pred − actual) / sum(actual) × 100.
    Positive = over-forecast (excess/expired stock risk).
    Negative = under-forecast (stockout risk).
    """
    total_actual = np.sum(np.abs(y_true))
    if total_actual < EPSILON:
        return np.nan
    return float(np.sum(y_pred - y_true) / total_actual * 100)


def compute_rolling_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 30,
) -> pd.Series:
    """
    30-day rolling bias % = rolling mean of (pred − actual) / actual × 100.
    Returns a Series; values are NaN where y_true ≈ 0.
    """
    ratio = np.where(np.abs(y_true) > EPSILON,
                     (y_pred - y_true) / (y_true + EPSILON) * 100,
                     np.nan)
    s = pd.Series(ratio)
    return s.rolling(window, min_periods=1).mean()


# 5.3 Probabilistic / Quantile metrics

def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.5) -> float:
    """
    Pinball (quantile) loss for quantile q.
    Phase 1: called with q=0.5 (= MAE / 2).
    Phase 2: called with q ∈ {0.5, 0.75, 0.90} for full quantile forecasting.

    Args:
        y_true: actual values
        y_pred: predicted values (point forecast used as q-quantile in Phase 1)
        q:      quantile level, default 0.5
    """
    e = y_true - y_pred
    loss = np.where(e >= 0, q * e, (q - 1) * e)
    return float(np.mean(loss))


# 5.4 Classification-style metrics (stockout_flag binary prediction)

def compute_classification_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    """
    Precision, Recall, F1 for binary stockout flag prediction.
    Only needed if the Operational dashboard uses a predicted flag (not just raw historical).
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    return {
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall":    float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1":        float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
    }


# 5.5 Acceptance thresholds

THRESHOLDS = {
    "r2_min":          0.55,   # pharma demand is noisier than HVAC
    "mape_max":        25.0,   # Phase 1 baseline target; Phase 2 tightens to 20%
    "smape_max":       25.0,
    "wape_max":        20.0,   # accuracy = 1 − WAPE >= 80%
    "bias_max_abs":     8.0,   # Phase 1; Phase 2 tightens to 5%
    "rmse_pct_max":    25.0,
    "mae_pct_max":     20.0,
}


def check_acceptance(metrics: dict) -> dict:
    """
    Returns pass/fail per metric + overall `all_pass`.
    Matching the reference project's pattern.
    """
    results = {}

    r2   = metrics.get("r2", np.nan)
    mape = metrics.get("mape", np.nan)
    smape = metrics.get("smape", np.nan)
    wape  = metrics.get("wape", np.nan)
    bias  = metrics.get("bias_pct", np.nan)
    rmse_pct = metrics.get("rmse_pct", np.nan)
    mae_pct  = metrics.get("mae_pct", np.nan)

    def _check(value, threshold, direction="below", name=""):
        if np.isnan(value):
            return {"value": value, "threshold": threshold, "pass": False, "reason": "NaN"}
        passed = (value >= threshold) if direction == "above" else (value <= threshold)
        return {"value": round(value, 4), "threshold": threshold, "pass": bool(passed)}

    results["r2"]        = _check(r2,             THRESHOLDS["r2_min"],        "above")
    results["mape"]      = _check(mape,            THRESHOLDS["mape_max"],      "below")
    results["smape"]     = _check(smape,           THRESHOLDS["smape_max"],     "below")
    results["wape"]      = _check(wape,            THRESHOLDS["wape_max"],      "below")
    results["bias_abs"]  = _check(abs(bias) if not np.isnan(bias) else np.nan,
                                  THRESHOLDS["bias_max_abs"], "below")
    results["rmse_pct"]  = _check(rmse_pct,        THRESHOLDS["rmse_pct_max"],  "below")
    results["mae_pct"]   = _check(mae_pct,          THRESHOLDS["mae_pct_max"],   "below")

    results["all_pass"] = all(v["pass"] for v in results.values() if isinstance(v, dict))
    return results


# Master compute function

def compute_regression_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    label: str = "",
) -> dict:
    """
    Master metric function — same signature as reference project's compute_regression_metrics.
    Returns a flat dict of all Phase 0–1 metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    metrics = {
        "label":     label,
        "n":         len(y_true),
        "mae":       compute_mae(y_true, y_pred),
        "mse":       compute_mse(y_true, y_pred),
        "rmse":      compute_rmse(y_true, y_pred),
        "mape":      compute_mape(y_true, y_pred),
        "smape":     compute_smape(y_true, y_pred),
        "wape":      compute_wape(y_true, y_pred),
        "r2":        compute_r2(y_true, y_pred),
        "rmse_pct":  compute_rmse_pct(y_true, y_pred),
        "mae_pct":   compute_mae_pct(y_true, y_pred),
        "bias_pct":  compute_bias_pct(y_true, y_pred),
        "pinball_50": compute_pinball_loss(y_true, y_pred, q=0.5),
    }
    return metrics


def compute_per_sku_metrics(
    test_df: pd.DataFrame,
    actual_col: str = "units_sold",
    pred_col: str = "lgb_pred",
    label: str = "lgb",
) -> pd.DataFrame:
    """
    Compute metrics per SKU (product_id).
    Returns a DataFrame with one row per product_id.
    """
    rows = []
    for pid, grp in test_df.groupby("product_id"):
        y_true = grp[actual_col].values
        y_pred = grp[pred_col].values
        m = compute_regression_metrics(y_true, y_pred, label=f"{label}_{pid}")
        m["product_id"] = pid
        rows.append(m)
    return pd.DataFrame(rows)
