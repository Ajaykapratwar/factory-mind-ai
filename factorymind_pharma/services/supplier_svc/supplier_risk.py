"""
services/supplier_svc/supplier_risk.py — Phase 2, §3.3
Supplier risk scoring.

Method: Simple, auditable weighted composite score combining:
  - delivery_reliability_pct  (higher = lower risk)
  - rejection_rate_pct        (higher = higher risk)

Score formula (spec §3.3 — deterministic, not a learned model):
  risk_score = w_reject * rejection_rate_pct + w_delay * (100 - delivery_reliability_pct)

Normalised to [0, 100] where 0 = safest, 100 = highest risk.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Weights — chosen so that rejection rate and delivery unreliability
# contribute roughly equally to the composite; auditable and interpretable.
W_REJECTION = 0.60   # rejection_rate has larger consequence (product quality)
W_DELIVERY  = 0.40   # delivery unreliability (operational impact)


def compute_supplier_risk(
    raw_materials_df: pd.DataFrame,
    suppliers_df: pd.DataFrame,
    models_dir: Path,
) -> pd.DataFrame:
    """
    §3.3: Compute risk scores for all suppliers.
    Returns a DataFrame with one row per supplier_id with risk_score [0-100].
    """
    logger.info("=== §3.3 Supplier Risk Scoring ===")

    raw_df  = raw_materials_df.copy()
    supp_df = suppliers_df.copy()

    # Detect supplier ID column
    supp_id_col = next(
        (c for c in ["supplier_id", "supplier_name", "id"] if c in supp_df.columns),
        supp_df.columns[0]
    )

    # Detect reliability / rejection columns
    rel_col = next(
        (c for c in ["delivery_reliability_pct", "delivery_reliability", "on_time_delivery_pct"]
         if c in supp_df.columns),
        None
    )
    rej_col = next(
        (c for c in ["rejection_rate_pct", "rejection_rate", "defect_rate_pct"]
         if c in supp_df.columns),
        None
    )

    # Also check raw_materials_df for aggregated values
    if rel_col is None and "delivery_reliability_pct" in raw_df.columns:
        rel_agg = raw_df.groupby("supplier_id" if "supplier_id" in raw_df.columns else raw_df.columns[0])["delivery_reliability_pct"].mean().reset_index()
        supp_df = supp_df.merge(rel_agg, on=supp_id_col, how="left")
        rel_col = "delivery_reliability_pct"
    if rej_col is None and "rejection_rate_pct" in raw_df.columns:
        rej_agg = raw_df.groupby("supplier_id" if "supplier_id" in raw_df.columns else raw_df.columns[0])["rejection_rate_pct"].mean().reset_index()
        supp_df = supp_df.merge(rej_agg, on=supp_id_col, how="left")
        rej_col = "rejection_rate_pct"

    logger.info("Supplier risk: using reliability='%s', rejection='%s'", rel_col, rej_col)

    # Compute raw risk components
    if rel_col and rel_col in supp_df.columns:
        supp_df["delivery_unreliability_pct"] = 100.0 - supp_df[rel_col].fillna(supp_df[rel_col].median())
    else:
        supp_df["delivery_unreliability_pct"] = 50.0   # neutral if missing

    if rej_col and rej_col in supp_df.columns:
        supp_df["rejection_rate_pct_clean"] = supp_df[rej_col].fillna(supp_df[rej_col].median())
    else:
        supp_df["rejection_rate_pct_clean"] = 5.0   # neutral if missing

    # Weighted composite risk score (raw, not yet normalised)
    supp_df["risk_raw"] = (
        W_REJECTION * supp_df["rejection_rate_pct_clean"] +
        W_DELIVERY  * supp_df["delivery_unreliability_pct"]
    )

    # Normalise to [0, 100]
    r_min = supp_df["risk_raw"].min()
    r_max = supp_df["risk_raw"].max()
    if r_max > r_min:
        supp_df["risk_score"] = (supp_df["risk_raw"] - r_min) / (r_max - r_min) * 100
    else:
        supp_df["risk_score"] = 50.0  # all equal — neutral

    supp_df["risk_score"] = supp_df["risk_score"].round(2)

    # Risk tier (for dashboard colouring)
    supp_df["risk_tier"] = pd.cut(
        supp_df["risk_score"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"],
    )

    # Verify: no nulls in risk_score
    n_null = supp_df["risk_score"].isna().sum()
    assert n_null == 0, f"risk_score has {n_null} nulls after computation"
    logger.info("Supplier risk scores computed for %d suppliers — no nulls", len(supp_df))

    # Print spot-check: top 5 highest risk
    print(f"\n  §3.3 Supplier Risk:")
    print(f"    Total suppliers scored: {len(supp_df)}")
    print(f"    Null risk_scores: {n_null}  ✓")
    print(f"    Weights: W_rejection={W_REJECTION}  W_delivery_unreliability={W_DELIVERY}")
    print(f"\n    Top 5 highest-risk suppliers:")
    top5 = supp_df.nlargest(5, "risk_score")[[
        supp_id_col,
        "rejection_rate_pct_clean" if "rejection_rate_pct_clean" in supp_df.columns else supp_id_col,
        "delivery_unreliability_pct",
        "risk_score", "risk_tier"
    ]]
    print(top5.to_string(index=False))
    print(f"\n    5 lowest-risk suppliers:")
    bot5 = supp_df.nsmallest(5, "risk_score")[[
        supp_id_col,
        "rejection_rate_pct_clean" if "rejection_rate_pct_clean" in supp_df.columns else supp_id_col,
        "delivery_unreliability_pct",
        "risk_score", "risk_tier"
    ]]
    print(bot5.to_string(index=False))

    # Save
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "supplier_risk_scorer.pkl", "wb") as f:
        pickle.dump({
            "method": "weighted_composite",
            "w_rejection": W_REJECTION,
            "w_delivery": W_DELIVERY,
            "supplier_id_col": supp_id_col,
            "rel_col": rel_col,
            "rej_col": rej_col,
        }, f)
    logger.info("Saved models/supplier_risk_scorer.pkl")

    return supp_df
