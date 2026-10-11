"""
forecast_svc/diagnostics.py — Phase 2, §1
Residual diagnostic analysis for the Phase 1 LightGBM baseline.

Confirms which of the 4 listed causes of R² ≈ 0.60 are present:
  1. Single-model variance ceiling
  2. Thin feature set
  3. No per-segment modelling
  4. Heteroscedasticity

Outputs a dict of findings; does NOT modify any model.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _load_predictions(outputs_dir: Path) -> Optional[pd.DataFrame]:
    p = outputs_dir / "test_predictions.parquet"
    if not p.exists():
        logger.warning("test_predictions.parquet not found at %s", p)
        return None
    df = pd.read_parquet(p, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_mft(processed_dir: Path) -> pd.DataFrame:
    p = processed_dir / "master_feature_table.parquet"
    df = pd.read_parquet(p, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Cause 1: Single-model variance ceiling ─────────────────────────────────

def _check_cause1_variance_ceiling(preds: pd.DataFrame) -> Dict:
    """
    Checks autocorrelation in residuals (lag-1, lag-7).
    Strong autocorrelation → model misses temporal patterns → variance ceiling.
    """
    preds = preds.dropna(subset=["lgb_pred", "units_sold"])
    preds = preds.sort_values(["product_id", "date"]).copy()
    preds["residual"] = preds["units_sold"] - preds["lgb_pred"]

    # Per-product residual autocorrelation at lag 1 and lag 7
    lag1_corrs, lag7_corrs = [], []
    for pid, grp in preds.groupby("product_id"):
        r = grp["residual"].values
        if len(r) > 10:
            lag1 = np.corrcoef(r[1:], r[:-1])[0, 1]
            if not np.isnan(lag1):
                lag1_corrs.append(lag1)
            if len(r) > 14:
                lag7 = np.corrcoef(r[7:], r[:-7])[0, 1]
                if not np.isnan(lag7):
                    lag7_corrs.append(lag7)

    mean_lag1 = float(np.mean(lag1_corrs)) if lag1_corrs else np.nan
    mean_lag7 = float(np.mean(lag7_corrs)) if lag7_corrs else np.nan
    # Strong if |corr| > 0.25
    evidence = (abs(mean_lag1) > 0.25) or (abs(mean_lag7) > 0.25)

    return {
        "mean_lag1_autocorr": round(mean_lag1, 4) if not np.isnan(mean_lag1) else None,
        "mean_lag7_autocorr": round(mean_lag7, 4) if not np.isnan(mean_lag7) else None,
        "evidence_found": bool(evidence),
        "interpretation": (
            "Residual autocorrelation detected — model misses temporal structure "
            "(short-run autocorrelation or seasonal cycles)."
            if evidence else
            "No strong residual autocorrelation — single-model variance ceiling "
            "less likely the primary driver."
        ),
    }


# ── Cause 2: Thin feature set ───────────────────────────────────────────────

def _check_cause2_thin_features(preds: pd.DataFrame, mft: pd.DataFrame) -> Dict:
    """
    Checks whether residuals correlate with features that were NOT used in Phase 1
    (e.g., day-of-month, month-start/end effects, rolling std).
    Proxy: R² of residuals regressed on expanded feature candidates vs just time.
    """
    preds = preds.dropna(subset=["lgb_pred", "units_sold"]).copy()
    preds["residual"] = preds["units_sold"] - preds["lgb_pred"]
    preds = preds.merge(
        mft[["date", "product_id", "disease_season_index",
             "market_demand_index", "promotion_flag", "criticality",
             "category"]].drop_duplicates(["date", "product_id"]),
        on=["date", "product_id"], how="left"
    )

    preds["day_of_month"] = preds["date"].dt.day
    preds["is_month_start"] = preds["date"].dt.is_month_start.astype(int)
    preds["is_month_end"] = preds["date"].dt.is_month_end.astype(int)
    preds["month"] = preds["date"].dt.month

    # Correlate each unused feature with |residual| (error magnitude)
    abs_resid = preds["residual"].abs()
    correlations = {}
    for col in ["day_of_month", "is_month_start", "is_month_end", "month",
                "disease_season_index", "market_demand_index"]:
        if col in preds.columns and preds[col].notna().sum() > 100:
            c = np.corrcoef(preds[col].fillna(0), abs_resid)[0, 1]
            correlations[col] = round(float(c), 4) if not np.isnan(c) else 0.0

    # Evidence: any correlation > 0.10
    max_corr = max(abs(v) for v in correlations.values()) if correlations else 0.0
    evidence = max_corr > 0.10

    return {
        "abs_residual_correlations": correlations,
        "max_abs_corr": round(max_corr, 4),
        "evidence_found": bool(evidence),
        "interpretation": (
            "Unused calendar/demand features correlate with errors — thin feature set confirmed."
            if evidence else
            "Calendar/demand features show low correlation with errors — thin features less likely primary driver."
        ),
    }


# ── Cause 3: No per-segment modelling ──────────────────────────────────────

def _check_cause3_per_segment(preds: pd.DataFrame, mft: pd.DataFrame) -> Dict:
    """
    Checks whether R² varies substantially across product criticality/category segments.
    High variance in per-segment R² → single global model is misfit.
    """
    from evaluation.metrics import compute_r2

    merged = preds.merge(
        mft[["product_id", "criticality", "category"]].drop_duplicates("product_id"),
        on="product_id", how="left"
    ).dropna(subset=["lgb_pred", "units_sold"])

    seg_r2 = {}
    for seg_col in ["criticality", "category"]:
        if seg_col not in merged.columns:
            continue
        r2_by_seg = {}
        for seg_val, grp in merged.groupby(seg_col):
            if len(grp) > 50:
                r2 = compute_r2(grp["units_sold"].values, grp["lgb_pred"].values)
                r2_by_seg[str(seg_val)] = round(float(r2), 4)
        seg_r2[seg_col] = r2_by_seg

    # Evidence: R² range > 0.15 across any segment
    ranges = {}
    evidence = False
    for seg_col, vals in seg_r2.items():
        if len(vals) >= 2:
            rng = max(vals.values()) - min(vals.values())
            ranges[seg_col] = round(rng, 4)
            if rng > 0.15:
                evidence = True

    return {
        "r2_by_segment": seg_r2,
        "r2_ranges": ranges,
        "evidence_found": bool(evidence),
        "interpretation": (
            "R² varies substantially across segments — per-segment modelling is needed."
            if evidence else
            "R² is relatively consistent across segments — per-segment modelling less critical."
        ),
    }


# ── Cause 4: Heteroscedasticity ────────────────────────────────────────────

def _check_cause4_heteroscedasticity(preds: pd.DataFrame, mft: pd.DataFrame) -> Dict:
    """
    Checks whether error variance (squared residual) increases with predicted/actual magnitude.
    Breusch-Pagan-style proxy: rank correlation of |residual| vs |predicted|.
    Also checks per-SKU R² distribution for heavy tail of under-performing SKUs.
    """
    from evaluation.metrics import compute_r2

    preds = preds.dropna(subset=["lgb_pred", "units_sold"]).copy()
    preds["residual"] = preds["units_sold"] - preds["lgb_pred"]
    preds["abs_residual"] = preds["residual"].abs()

    # Spearman rank correlation of |residual| vs |predicted|
    from scipy.stats import spearmanr
    rho, pval = spearmanr(preds["lgb_pred"].abs(), preds["abs_residual"])

    # Per-SKU R² distribution
    sku_r2 = {}
    for pid, grp in preds.groupby("product_id"):
        if len(grp) > 5:
            r2 = compute_r2(grp["units_sold"].values, grp["lgb_pred"].values)
            sku_r2[pid] = float(r2)

    r2_vals = list(sku_r2.values())
    n_poor = sum(1 for v in r2_vals if v < 0.40)   # SKUs dragging the aggregate down
    n_good = sum(1 for v in r2_vals if v >= 0.80)

    evidence = (abs(rho) > 0.25) or (n_poor > len(r2_vals) * 0.20)

    return {
        "spearman_rho_resid_vs_pred": round(float(rho), 4),
        "spearman_pval": round(float(pval), 4),
        "n_sku_r2_below_0.40": n_poor,
        "n_sku_r2_above_0.80": n_good,
        "total_skus": len(r2_vals),
        "r2_p25": round(float(np.percentile(r2_vals, 25)), 4) if r2_vals else None,
        "r2_p50": round(float(np.percentile(r2_vals, 50)), 4) if r2_vals else None,
        "r2_p75": round(float(np.percentile(r2_vals, 75)), 4) if r2_vals else None,
        "evidence_found": bool(evidence),
        "interpretation": (
            "Heteroscedasticity confirmed: error variance scales with predicted magnitude / "
            "many poor-fit SKUs drag aggregate R² down."
            if evidence else
            "No strong heteroscedasticity signal."
        ),
    }


# ── Master diagnostic function ──────────────────────────────────────────────

def run_diagnostic(
    processed_dir: Path,
    outputs_dir: Path,
) -> Dict:
    """
    §1 diagnostic: runs all 4 cause-checks and returns a structured findings dict.
    Call this BEFORE any model changes.
    """
    logger.info("=== §1 Residual Diagnostic ===")

    preds = _load_predictions(outputs_dir)
    if preds is None:
        return {"error": "test_predictions.parquet not found — run Phase 1 pipeline first"}

    mft = _load_mft(processed_dir)

    from evaluation.metrics import compute_r2, compute_regression_metrics
    preds_clean = preds.dropna(subset=["lgb_pred", "units_sold"])
    baseline_agg = compute_regression_metrics(
        preds_clean["units_sold"].values,
        preds_clean["lgb_pred"].values,
        label="Phase1_LightGBM"
    )

    logger.info("Phase 1 baseline: R²=%.4f  MAPE=%.2f%%", baseline_agg["r2"], baseline_agg["mape"])

    cause1 = _check_cause1_variance_ceiling(preds)
    cause2 = _check_cause2_thin_features(preds, mft)
    cause3 = _check_cause3_per_segment(preds, mft)
    cause4 = _check_cause4_heteroscedasticity(preds, mft)

    findings = {
        "phase1_baseline_metrics": {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in baseline_agg.items()},
        "cause1_variance_ceiling": cause1,
        "cause2_thin_features": cause2,
        "cause3_no_per_segment": cause3,
        "cause4_heteroscedasticity": cause4,
        "summary": {
            "cause1_confirmed": cause1["evidence_found"],
            "cause2_confirmed": cause2["evidence_found"],
            "cause3_confirmed": cause3["evidence_found"],
            "cause4_confirmed": cause4["evidence_found"],
        },
    }

    # Print summary
    print("\n" + "=" * 68)
    print("  §1 RESIDUAL DIAGNOSTIC REPORT")
    print("=" * 68)
    print(f"\n  Phase 1 Baseline: R²={baseline_agg['r2']:.4f}  MAPE={baseline_agg['mape']:.2f}%")
    print(f"  Target:           R²≥0.80")
    print(f"  Gap to close:     {0.80 - baseline_agg['r2']:.4f}")
    print("\n  Cause findings:")
    for i, (key, label) in enumerate([
        ("cause1_confirmed", "Cause 1: Single-model variance ceiling (residual autocorrelation)"),
        ("cause2_confirmed", "Cause 2: Thin feature set (calendar/demand feature correlation)"),
        ("cause3_confirmed", "Cause 3: No per-segment modelling (R² variance across segments)"),
        ("cause4_confirmed", "Cause 4: Heteroscedasticity (error scales with volume)"),
    ], 1):
        status = "[CONFIRMED]" if findings["summary"][key] else "[not found]"
        print(f"    {status}  {label}")

    print("\n  Detail per cause:")
    print(f"    C1 lag-1 autocorr: {cause1['mean_lag1_autocorr']}")
    print(f"    C1 lag-7 autocorr: {cause1['mean_lag7_autocorr']}")
    print(f"    C2 max |corr| with abs-residual: {cause2['max_abs_corr']}")
    if cause3["r2_ranges"]:
        for seg, rng in cause3["r2_ranges"].items():
            print(f"    C3 R²-range across {seg}: {rng:.4f}")
    print(f"    C4 Spearman ρ(|pred|, |resid|): {cause4['spearman_rho_resid_vs_pred']}")
    print(f"    C4 SKUs with R²<0.40: {cause4['n_sku_r2_below_0.40']}/{cause4['total_skus']}")
    print("=" * 68)

    return findings
