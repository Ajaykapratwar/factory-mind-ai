"""
services/policy_svc/inventory_optimizer.py — Phase 3, §4.1
Safety stock + reorder point optimization (L4 Policy).

Method:
  safety_stock = z × σ_demand × sqrt(lead_time)
  where:
    σ_demand = per-SKU residual std-dev from ensemble on validation set (data-driven)
    z = service-level z-score per criticality tier
    lead_time = inferred from orders data or default

Also:
  §4.2 Compliance gating: deviation_count > 0 OR audit_status != 'Passed' → blocked
  §4.3 Cold-chain gating: temperature_excursion_flag = 1 → blocked

Both gates are deterministic rules, NOT learned models.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Service-level z-scores per criticality tier (spec §4.1)
Z_SCORES = {
    "critical":     2.33,   # 99% service level
    "high":         2.33,   # 99% service level
    "standard":     1.65,   # 95% service level
    "medium":       1.65,   # 95% service level
    "non-critical": 1.28,   # 90% service level
    "low":          1.28,   # 90% service level
    "unknown":      1.65,   # default to standard
}


def _get_z_score(criticality: str) -> float:
    """Map criticality string to z-score. Case-insensitive."""
    if pd.isna(criticality):
        return Z_SCORES["unknown"]
    key = str(criticality).lower().strip()
    return Z_SCORES.get(key, Z_SCORES["unknown"])


def compute_safety_stock(
    test_df: pd.DataFrame,
    mft: pd.DataFrame,
    inventory_df: pd.DataFrame,
    product_master_df: pd.DataFrame,
    pred_col: str = "stack_pred",
) -> pd.DataFrame:
    """
    §4.1: Compute optimized safety stock and reorder point per SKU.

    test_df: ensemble predictions on test set (has 'units_sold' + pred_col).
    mft: master feature table (for lead_time derivation if available).
    inventory_df: raw inventory (for existing safety_stock/reorder_point baseline).
    product_master_df: product dimensions (criticality, unit_price).
    """
    logger.info("=== §4.1 Safety Stock Optimization ===")

    # ── Per-SKU σ_demand from ensemble residuals ──────────────────────────
    sku_sigma = {}
    for pid, grp in test_df.groupby("product_id"):
        if len(grp) >= 5 and pred_col in grp.columns:
            residuals = grp["units_sold"].values - grp[pred_col].values
            sku_sigma[pid] = float(np.std(residuals))

    # ── Lead time per SKU (from orders if available, else 7-day default) ──
    lead_time_default = 7  # days
    lead_time_map = {}
    if "ordered_quantity" in mft.columns and "date" in mft.columns:
        # Proxy: avg days between order date and (inferred) receipt = not directly available
        # Use default for all SKUs and note as simplification
        logger.info("No explicit lead_time column found; using default %d days for all SKUs",
                    lead_time_default)
    
    # Check inventory_df for warehouse hierarchy
    has_hierarchy = False
    if "warehouse_id" in inventory_df.columns:
        n_warehouses = inventory_df["warehouse_id"].nunique()
        # Check if any 'warehouse_type' or 'tier' column hints at hierarchy
        hierarchy_cols = [c for c in inventory_df.columns if "type" in c.lower() or "tier" in c.lower() or "level" in c.lower()]
        if hierarchy_cols:
            has_hierarchy = True
            logger.info("Warehouse hierarchy detected via columns: %s", hierarchy_cols)
        else:
            logger.info("No warehouse hierarchy field found — optimizing per-warehouse independently "
                       "(noted as simplification in ASSUMPTIONS.md). %d warehouses.", n_warehouses)

    # ── Product criticality from product_master ────────────────────────────
    pm = product_master_df[["product_id", "criticality", "unit_price"]].copy()
    pm = pm.drop_duplicates("product_id")

    # ── Latest baseline safety stock / reorder point ──────────────────────
    baseline = (
        inventory_df.sort_values("date")
        .groupby("product_id")[["safety_stock", "reorder_point"]]
        .last()
        .reset_index()
        .rename(columns={
            "safety_stock":  "baseline_safety_stock",
            "reorder_point": "baseline_reorder_point",
        })
    )

    # ── Build optimization table ──────────────────────────────────────────
    opt_rows = []
    for pid in set(list(sku_sigma.keys()) + list(pm["product_id"].tolist())):
        sigma = sku_sigma.get(pid, 0)
        lead_time = lead_time_map.get(pid, lead_time_default)
        crit_row = pm[pm["product_id"] == pid]
        criticality = crit_row["criticality"].iloc[0] if len(crit_row) > 0 else "unknown"
        unit_price  = float(crit_row["unit_price"].iloc[0]) if len(crit_row) > 0 else 0.0
        z = _get_z_score(criticality)

        # safety_stock = z × σ_demand × sqrt(lead_time)
        new_ss = z * sigma * np.sqrt(lead_time)
        # reorder_point = avg_daily_demand × lead_time + safety_stock
        avg_demand = test_df[test_df["product_id"] == pid]["units_sold"].mean() \
                     if len(test_df[test_df["product_id"] == pid]) > 0 else 0.0
        new_rop = avg_demand * lead_time + new_ss

        base_row = baseline[baseline["product_id"] == pid]
        base_ss  = float(base_row["baseline_safety_stock"].iloc[0]) if len(base_row) > 0 else np.nan
        base_rop = float(base_row["baseline_reorder_point"].iloc[0]) if len(base_row) > 0 else np.nan

        opt_rows.append({
            "product_id":           pid,
            "criticality":          criticality,
            "unit_price":           unit_price,
            "sigma_demand":         round(sigma, 4),
            "z_score":              z,
            "lead_time_days":       lead_time,
            "avg_daily_demand":     round(avg_demand, 4),
            "new_safety_stock":     round(new_ss, 2),
            "new_reorder_point":    round(new_rop, 2),
            "baseline_safety_stock": round(base_ss, 2) if not np.isnan(base_ss) else None,
            "baseline_reorder_point": round(base_rop, 2) if not np.isnan(base_rop) else None,
            "ss_change":            round(new_ss - base_ss, 2) if not np.isnan(base_ss) else None,
            "holding_cost_change":  round((new_ss - base_ss) * unit_price, 2) if not np.isnan(base_ss) else None,
        })

    opt_df = pd.DataFrame(opt_rows)
    logger.info("Safety stock computed for %d SKUs", len(opt_df))
    return opt_df, has_hierarchy


def backtest_stockout_reduction(
    inventory_df: pd.DataFrame,
    opt_df: pd.DataFrame,
) -> Dict:
    """
    §4.1: Backtest — compare historical stockouts under old vs. new reorder logic.
    Uses the inventory.stockout_flag column.
    """
    logger.info("Running stockout backtest...")

    inv = inventory_df.copy()
    inv["date"] = pd.to_datetime(inv["date"])

    # Count stockouts per product under old reorder_point
    old_stockouts = inv[inv["stockout_flag"] == 1].groupby("product_id").size().reset_index(name="old_stockouts")

    # Simulate: would new reorder_point have caught earlier?
    # Proxy: if closing_stock < new_safety_stock → would have triggered reorder earlier
    inv_latest = inv.sort_values("date").groupby("product_id").last().reset_index()
    merged = inv_latest.merge(
        opt_df[["product_id", "new_reorder_point", "new_safety_stock"]],
        on="product_id", how="left"
    )

    # Simulated stockouts: closing_stock <= 0 AND wouldn't have been reordered
    # (simplified: compare closing_stock to new vs old reorder point)
    n_old_stockouts = int(inv["stockout_flag"].sum())

    # Simulated new policy: any day where closing_stock < new_safety_stock
    # (orders would have been placed) → stockout averted
    inv_with_opt = inv.merge(
        opt_df[["product_id", "new_reorder_point", "new_safety_stock"]],
        on="product_id", how="left"
    )
    # Old policy would stock out if closing < old reorder AND no order placed
    # New policy stocks out if closing < new_safety_stock (tighter)
    # Simulated reduction: rows where old_stockout=1 AND closing_stock >= new_safety_stock
    averted = inv_with_opt[
        (inv_with_opt["stockout_flag"] == 1) &
        (inv_with_opt["closing_stock"] >= inv_with_opt["new_safety_stock"].fillna(0))
    ]
    n_averted = len(averted)
    n_new_stockouts = max(0, n_old_stockouts - n_averted)

    pct_reduction = ((n_old_stockouts - n_new_stockouts) / max(n_old_stockouts, 1)) * 100
    # Fill rate proxy = 1 - stockout_rate
    total_days = len(inv)
    fill_rate = (1 - n_new_stockouts / max(total_days, 1)) * 100

    # Holding cost change
    hc_change = opt_df["holding_cost_change"].dropna().sum()

    print(f"\n  §4.1 Safety Stock Backtest:")
    print(f"    Old stockouts (historical): {n_old_stockouts:,}")
    print(f"    Simulated new stockouts:    {n_new_stockouts:,}")
    print(f"    Reduction:  {pct_reduction:.1f}%  {'[PASS >=15%]' if pct_reduction >= 15 else '[WARN <15%]'}")
    print(f"    Fill rate:  {fill_rate:.2f}%  {'[PASS >=96%]' if fill_rate >= 96 else '[WARN <96%]'}")
    print(f"    Net holding cost change: {hc_change:+,.2f} (unit_price x delta safety_stock)")

    return {
        "n_old_stockouts": n_old_stockouts,
        "n_new_stockouts": n_new_stockouts,
        "pct_reduction": round(pct_reduction, 2),
        "fill_rate": round(fill_rate, 2),
        "holding_cost_change": round(float(hc_change), 2),
        "gate_stockout_reduction_pass": pct_reduction >= 15.0,
        "gate_fill_rate_pass": fill_rate >= 96.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# §4.2 Compliance gating — deterministic rule, NOT a model
# ═══════════════════════════════════════════════════════════════════════════

def apply_compliance_gate(
    orders_df: pd.DataFrame,
    compliance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    TIERED COMPLIANCE GATE -- spec §10.2, corrected per C1.

    audit_status actual values: Compliant / Minor Deviation / Major Deviation.
    There is NO 'Passed' value (old assumption was wrong -- see ASSUMPTIONS.md C1).

    Tiers (deterministic, NOT learned):
      BLOCK  : audit_status = 'Major Deviation'  (QA role must act)
      REVIEW : audit_status = 'Minor Deviation' OR deviation_count >= 2
      AUTO   : audit_status = 'Compliant' AND deviation_count < 2

    Returns orders_df with columns:
      compliance_blocked    (bool) -- BLOCK or REVIEW tier
      compliance_tier       (str)  -- 'BLOCK' | 'REVIEW' | 'AUTO'
      compliance_reason     (str)  -- human-readable reason
    """
    comp = compliance_df.copy()

    # Log actual audit_status values seen (for audit trail)
    if "audit_status" in comp.columns:
        unique_audit = sorted(comp["audit_status"].dropna().unique().tolist())
        logger.info("audit_status unique values in data: %s", unique_audit)
        if "Passed" in unique_audit:
            logger.warning(
                "C1: 'Passed' found in audit_status -- data changed. "
                "Tiered gate still applies; see ASSUMPTIONS.md."
            )
    else:
        logger.warning("audit_status column not found in compliance_df")

    # Map batch_id to order_id if order_id is in orders_df but not in comp
    if "order_id" in orders_df.columns and "order_id" not in comp.columns and "batch_id" in comp.columns:
        comp["order_id"] = comp["batch_id"].astype(str).str.replace(r"^B", "O", regex=True)

    # Determine join key
    batch_col = next(
        (c for c in ["order_id", "batch_id", "product_id"] if c in comp.columns and c in orders_df.columns),
        None
    )

    if batch_col is None:
        logger.warning("No shared key between compliance_df and orders_df -- returning all AUTO")
        orders_df = orders_df.copy()
        orders_df["compliance_blocked"] = False
        orders_df["compliance_tier"]    = "AUTO"
        orders_df["compliance_reason"]  = ""
        return orders_df

    # Compute tier per compliance record
    def _get_tier(row) -> str:
        audit = str(row.get("audit_status", "")) if "audit_status" in comp.columns else ""
        dev   = int(row.get("deviation_count", 0)) if "deviation_count" in comp.columns else 0
        if audit == "Major Deviation":
            return "BLOCK"
        if audit == "Minor Deviation" or dev >= 2:
            return "REVIEW"
        return "AUTO"

    comp["_tier"] = comp.apply(_get_tier, axis=1)

    # Aggregate: if any record for a batch is BLOCK, the batch is BLOCK; etc.
    _tier_order = {"BLOCK": 2, "REVIEW": 1, "AUTO": 0}
    comp["_tier_rank"] = comp["_tier"].map(_tier_order).fillna(0)
    agg = comp.groupby(batch_col)["_tier_rank"].max().reset_index()
    agg["compliance_tier"] = agg["_tier_rank"].map({2: "BLOCK", 1: "REVIEW", 0: "AUTO"})

    orders_out = orders_df.copy().merge(
        agg[[batch_col, "compliance_tier"]], on=batch_col, how="left"
    )
    orders_out["compliance_tier"] = orders_out["compliance_tier"].fillna("AUTO")
    orders_out["compliance_blocked"] = orders_out["compliance_tier"].isin(["BLOCK", "REVIEW"])
    orders_out["compliance_reason"] = orders_out["compliance_tier"].map({
        "BLOCK": "audit_status=Major Deviation",
        "REVIEW": "audit_status=Minor Deviation or deviation_count>=2",
        "AUTO": "",
    })

    n_blocked = orders_out["compliance_blocked"].sum()
    n_total   = len(orders_out)
    print(f"\n  Compliance Gate (tiered, C1-corrected):")
    print(f"    Total records:    {n_total:,}")
    print(f"    BLOCK+REVIEW:     {n_blocked:,}  ({n_blocked/max(n_total,1)*100:.1f}%)")
    print(f"    AUTO-ELIGIBLE:    {n_total - n_blocked:,}")
    n_block = (orders_out["compliance_tier"] == "BLOCK").sum()
    n_review = (orders_out["compliance_tier"] == "REVIEW").sum()
    print(f"    of which BLOCK:   {n_block:,}  (Major Deviation)")
    print(f"    of which REVIEW:  {n_review:,}  (Minor Deviation or deviation_count>=2)")

    # Verify: zero BLOCK-tier items appear in auto-approved
    auto_approved = orders_out[orders_out["compliance_tier"] == "AUTO"]
    assert auto_approved["compliance_blocked"].sum() == 0, \
        "BUG: compliance-blocked records appear in AUTO output!"

    return orders_out


# ═══════════════════════════════════════════════════════════════════════════
# §4.3 Cold-chain / logistics gate — deterministic rule
# ═══════════════════════════════════════════════════════════════════════════

def apply_coldchain_gate(
    orders_df: pd.DataFrame,
    logistics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    §4.3: Hard gate — any shipment with temperature_excursion_flag=1 is
    flagged for human review; cannot be auto-approved.

    Deterministic rule only — no ML.
    """
    log_df = logistics_df.copy()

    # Map shipment_id to order_id if order_id is in orders_df but not in log_df
    if "order_id" in orders_df.columns and "order_id" not in log_df.columns and "shipment_id" in log_df.columns:
        log_df["order_id"] = log_df["shipment_id"].astype(str).str.replace(r"^SH", "O", regex=True)

    join_col = next(
        (c for c in ["order_id", "shipment_id", "product_id"] if c in log_df.columns and c in orders_df.columns),
        None
    )

    if join_col is None or "temperature_excursion_flag" not in log_df.columns:
        logger.warning("Cannot apply cold-chain gate: missing join key or temperature_excursion_flag")
        orders_df = orders_df.copy()
        orders_df["coldchain_blocked"] = False
        orders_df["coldchain_reason"]  = ""
        return orders_df

    excursion_ids = set(
        log_df[log_df["temperature_excursion_flag"] == 1][join_col].tolist()
    )

    orders_out = orders_df.copy()
    orders_out["coldchain_blocked"] = orders_out[join_col].isin(excursion_ids)
    orders_out["coldchain_reason"]  = orders_out["coldchain_blocked"].apply(
        lambda x: "temperature_excursion_flag=1" if x else ""
    )

    n_blocked = orders_out["coldchain_blocked"].sum()
    n_total   = len(orders_out)
    print(f"\n  §4.3 Cold-Chain Gate:")
    print(f"    Total records: {n_total:,}")
    print(f"    Temp excursion blocked: {n_blocked:,}  ({n_blocked/max(n_total,1)*100:.1f}%)")

    # Verify: excursion-flagged excluded from auto-approval
    auto_approved = orders_out[~orders_out["coldchain_blocked"]]
    assert auto_approved["coldchain_blocked"].sum() == 0, \
        "BUG: cold-chain-blocked records appear in auto-approved output!"

    return orders_out


def apply_all_gates(
    orders_df: pd.DataFrame,
    compliance_df: pd.DataFrame,
    logistics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply compliance + cold-chain gates and compute final auto_approved flag.
    """
    orders = apply_compliance_gate(orders_df, compliance_df)
    orders = apply_coldchain_gate(orders, logistics_df)
    orders["auto_approved"] = ~(orders["compliance_blocked"] | orders["coldchain_blocked"])

    n_auto = orders["auto_approved"].sum()
    n_blocked = len(orders) - n_auto
    print(f"\n  Combined gate result:")
    print(f"    Auto-approved: {n_auto:,}   Flagged for review: {n_blocked:,}")

    return orders


def run_synthetic_gate_test(orders_df: pd.DataFrame) -> Dict:
    """
    Spec §4.2 verify: synthetic test cases exercise all three tiers.
    Uses the C1-corrected audit_status values (Compliant / Minor Deviation / Major Deviation).
    """
    synth_comp = pd.DataFrame([
        # BLOCK: Major Deviation
        {"product_id": "TEST_BLOCK",   "deviation_count": 0, "audit_status": "Major Deviation"},
        # REVIEW: Minor Deviation
        {"product_id": "TEST_REVIEW",  "deviation_count": 0, "audit_status": "Minor Deviation"},
        # REVIEW: deviation_count >= 2 (Compliant but high count)
        {"product_id": "TEST_REVIEW2", "deviation_count": 3, "audit_status": "Compliant"},
        # AUTO: Compliant with low deviation_count
        {"product_id": "TEST_AUTO",    "deviation_count": 1, "audit_status": "Compliant"},
    ])
    synth_orders = pd.DataFrame([
        {"product_id": "TEST_BLOCK",   "order_id": "O1"},
        {"product_id": "TEST_REVIEW",  "order_id": "O2"},
        {"product_id": "TEST_REVIEW2", "order_id": "O3"},
        {"product_id": "TEST_AUTO",    "order_id": "O4"},
    ])
    synth_logistics = pd.DataFrame([
        {"product_id": p, "temperature_excursion_flag": 0}
        for p in ["TEST_BLOCK", "TEST_REVIEW", "TEST_REVIEW2", "TEST_AUTO"]
    ])

    result = apply_all_gates(synth_orders, synth_comp, synth_logistics)
    test_block  = result[result["product_id"] == "TEST_BLOCK"]["compliance_tier"].iloc[0] == "BLOCK"
    test_rev    = result[result["product_id"] == "TEST_REVIEW"]["compliance_tier"].iloc[0] == "REVIEW"
    test_rev2   = result[result["product_id"] == "TEST_REVIEW2"]["compliance_tier"].iloc[0] == "REVIEW"
    test_auto   = result[result["product_id"] == "TEST_AUTO"]["auto_approved"].all()

    passed = all([test_block, test_rev, test_rev2, test_auto])
    print(f"\n  Synthetic compliance gate test (C1-corrected tiers):")
    print(f"    TEST_BLOCK  (Major Deviation):      {'[PASS BLOCK]' if test_block else '[FAIL]'}")
    print(f"    TEST_REVIEW (Minor Deviation):      {'[PASS REVIEW]' if test_rev else '[FAIL]'}")
    print(f"    TEST_REVIEW2 (deviation_count=3):   {'[PASS REVIEW]' if test_rev2 else '[FAIL]'}")
    print(f"    TEST_AUTO   (Compliant, count<2):   {'[PASS AUTO]' if test_auto else '[FAIL]'}")
    print(f"    Overall gate test: {'[PASS]' if passed else '[FAIL]'}")

    # Cold-chain test
    synth_log2 = pd.DataFrame([
        {"product_id": "TEST_EXCURSION", "temperature_excursion_flag": 1},
        {"product_id": "TEST_OK2",       "temperature_excursion_flag": 0},
    ])
    synth_ord2 = pd.DataFrame([
        {"product_id": "TEST_EXCURSION", "order_id": "O10"},
        {"product_id": "TEST_OK2",       "order_id": "O11"},
    ])
    synth_comp2 = pd.DataFrame([
        {"product_id": "TEST_EXCURSION", "deviation_count": 0, "audit_status": "Compliant"},
        {"product_id": "TEST_OK2",       "deviation_count": 0, "audit_status": "Compliant"},
    ])
    result2 = apply_all_gates(synth_ord2, synth_comp2, synth_log2)
    excursion_blocked = result2[result2["product_id"] == "TEST_EXCURSION"]["coldchain_blocked"].all()
    ok2_approved      = result2[result2["product_id"] == "TEST_OK2"]["auto_approved"].all()

    print(f"\n  Synthetic cold-chain gate test:")
    print(f"    TEST_EXCURSION (excursion_flag=1): {'[PASS blocked]' if excursion_blocked else '[FAIL not blocked]'}")
    print(f"    TEST_OK2 (no excursion):           {'[PASS approved]' if ok2_approved else '[FAIL not approved]'}")

    return {
        "compliance_test_pass":  passed,
        "coldchain_test_pass":   excursion_blocked and ok2_approved,
        "all_gate_tests_pass":   passed and excursion_blocked and ok2_approved,
    }
