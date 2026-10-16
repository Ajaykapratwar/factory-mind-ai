"""
dashboard/data_loader.py — Step 1.4
Thin data-access layer for the dashboards.
Reads master_feature_table.parquet and forecast outputs directly (no FastAPI needed in Phase 1).
"""

import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_MFT_CACHE: Optional[pd.DataFrame] = None
_PRED_CACHE: Optional[pd.DataFrame] = None
_INV_CACHE: Optional[pd.DataFrame] = None

# Resolve paths relative to this file
_BASE = Path(__file__).parent.parent
_CONFIG_PATH = _BASE / "configs" / "config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_processed_dir() -> Path:
    cfg = _load_config()
    p = Path(cfg["data"]["processed_dir"])
    if not p.is_absolute():
        p = _BASE / p
    return p


def _get_outputs_dir() -> Path:
    cfg = _load_config()
    p = Path(cfg["data"]["outputs_dir"])
    if not p.is_absolute():
        p = _BASE / p
    return p


def _get_raw_dir() -> Path:
    cfg = _load_config()
    p = Path(cfg["data"]["raw_dir"])
    if not p.is_absolute():
        p = _BASE / p
    return p


def _load_mft() -> pd.DataFrame:
    global _MFT_CACHE
    if _MFT_CACHE is None:
        path = _get_processed_dir() / "master_feature_table.parquet"
        _MFT_CACHE = pd.read_parquet(path, engine="pyarrow")
        _MFT_CACHE["date"] = pd.to_datetime(_MFT_CACHE["date"])
    return _MFT_CACHE


def _load_predictions() -> Optional[pd.DataFrame]:
    """Prefer full-grid predictions (zero-leakage, calendar grid) over the old sales-rows-only file."""
    global _PRED_CACHE
    if _PRED_CACHE is None:
        full_grid_path = _get_outputs_dir() / "test_predictions_full_grid.parquet"
        legacy_path    = _get_outputs_dir() / "test_predictions.parquet"
        path = full_grid_path if full_grid_path.exists() else legacy_path
        if not path.exists():
            return None
        _PRED_CACHE = pd.read_parquet(path, engine="pyarrow")
        _PRED_CACHE["date"] = pd.to_datetime(_PRED_CACHE["date"])
    return _PRED_CACHE


def _load_inventory_raw() -> pd.DataFrame:
    global _INV_CACHE
    if _INV_CACHE is None:
        cfg = _load_config()
        raw_dir = _get_raw_dir()
        path = raw_dir / cfg["data"]["files"]["inventory"]
        _INV_CACHE = pd.read_csv(path, low_memory=False)
        _INV_CACHE["date"] = pd.to_datetime(_INV_CACHE["date"])
    return _INV_CACHE


# ---------------------------------------------------------------------------
# Public API — Step 1.4
# ---------------------------------------------------------------------------

def get_kpi_summary() -> dict:
    """
    Returns top-level KPI dict for the Executive Summary dashboard.
    Keys: total_revenue, total_units_sold, forecast_accuracy_wape, avg_days_of_supply, stockout_count
    """
    mft = _load_mft()
    preds = _load_predictions()

    total_revenue = float(mft["revenue"].sum())
    total_units   = float(mft["units_sold"].sum())

    # Product-week WAPE from full-grid evaluation (the correct reporting grain).
    # NOTE: "1 − WAPE" at product-day on the full calendar grid is NOT a meaningful accuracy
    # metric and must NOT be shown on dashboards. Use product-week WAPE instead.
    forecast_wape_pct   = None   # product-week WAPE (hurdle model, full grid)
    forecast_wape_grain = None   # grain label for display
    p90_coverage_pct    = None   # empirical P90 quantile coverage
    pred_col = None
    if preds is not None:
        if "pred_hurdle" in preds.columns:
            pred_col = "pred_hurdle"
        elif "pred_lgb" in preds.columns:
            pred_col = "pred_lgb"
        elif "lgb_pred" in preds.columns:
            pred_col = "lgb_pred"
    if preds is not None and pred_col is not None:
        # Compute weekly WAPE by aggregating to product-week blocks
        _p = preds.copy()
        _p["date"] = pd.to_datetime(_p["date"])
        test_start = pd.Timestamp("2022-10-20")
        _p = _p[_p["date"] >= test_start].copy()
        if len(_p) > 0:
            _p["block_7d"] = (_p["date"] - test_start).dt.days // 7
            max_block = (73 // 7) - 1  # 10 complete 7-day blocks
            _p = _p[_p["block_7d"] <= max_block]
            pw = _p.groupby(["product_id", "block_7d"]).agg(
                units_sold=("units_sold", "sum"), pred=(pred_col, "sum")
            ).reset_index()
            wape_num = np.sum(np.abs(pw["units_sold"].values - pw["pred"].values))
            wape_den = np.sum(np.abs(pw["units_sold"].values))
            if wape_den > 0:
                forecast_wape_pct   = round(wape_num / wape_den * 100, 1)
                forecast_wape_grain = "product-week (10 complete 7-day blocks)"
        # P90 coverage
        if "pred_q90" in preds.columns:
            _q = preds[preds["date"] >= test_start] if "date" in preds.columns else preds
            p90_coverage_pct = round(float(np.mean(_q["units_sold"].values <= _q["pred_q90"].values)) * 100, 1)

    # Days of supply = closing_stock / avg_daily_units_sold
    if "total_closing_stock" in mft.columns:
        avg_daily_sold = mft.groupby("product_id")["units_sold"].mean()
        closing = mft.groupby("product_id")["total_closing_stock"].last()
        dos = (closing / avg_daily_sold.clip(lower=1)).mean()
        avg_dos = round(float(dos), 1)
    else:
        avg_dos = None

    # Stockout count from raw inventory
    inv = _load_inventory_raw()
    stockout_count = int(inv["stockout_flag"].sum()) if "stockout_flag" in inv.columns else 0

    return {
        "total_revenue":         round(total_revenue, 2),
        "total_units_sold":      int(total_units),
        "forecast_wape_pct":     forecast_wape_pct,
        "forecast_wape_grain":   forecast_wape_grain,
        "p90_coverage_pct":      p90_coverage_pct,
        "avg_days_of_supply":    avg_dos,
        "stockout_count":        stockout_count,
    }


def get_demand_vs_actual(product_id: Optional[str] = None) -> pd.DataFrame:
    """
    Returns a DataFrame of actual + naive + hurdle/lgb forecasts (+ quantile bands) for charting.
    If product_id is None → aggregate across all SKUs by date.
    Prefers full-grid predictions when available.
    """
    mft  = _load_mft()
    pred = _load_predictions()

    if product_id:
        mft  = mft[mft["product_id"] == product_id]
        pred = pred[pred["product_id"] == product_id] if pred is not None else None

    # Aggregate actuals by date
    actual_agg = (
        mft.groupby("date", as_index=False)["units_sold"]
        .sum()
        .rename(columns={"units_sold": "actual"})
    )

    if pred is not None:
        agg_cols = {}
        if "naive_pred" in pred.columns:
            agg_cols["naive_pred"] = ("naive_pred", "sum")
        # Prefer hurdle → lgb → legacy
        for col in ["pred_hurdle", "pred_lgb", "lgb_pred"]:
            if col in pred.columns:
                agg_cols[col] = (col, "sum")
                break
        for qcol in ["pred_q10", "pred_q90"]:
            if qcol in pred.columns:
                agg_cols[qcol] = (qcol, "sum")
        if agg_cols:
            pred_agg = pred.groupby("date", as_index=False).agg(**agg_cols)
        else:
            pred_agg = pred[["date"]].drop_duplicates()
        result = actual_agg.merge(pred_agg, on="date", how="left")
    else:
        result = actual_agg

    return result.sort_values("date").reset_index(drop=True)



def get_inventory_status(warehouse_id: Optional[str] = None) -> pd.DataFrame:
    """
    Returns warehouse-level inventory table.
    If warehouse_id is None → all warehouses.
    """
    inv = _load_inventory_raw()

    if warehouse_id:
        inv = inv[inv["warehouse_id"] == warehouse_id]

    # Latest snapshot per product+warehouse
    latest = inv.sort_values("date").groupby(["product_id", "warehouse_id"], as_index=False).last()

    return latest[[
        "product_id", "warehouse_id", "date",
        "opening_stock", "received_quantity", "production_received",
        "units_sold", "closing_stock", "safety_stock", "reorder_point", "stockout_flag"
    ]].reset_index(drop=True)


def get_per_sku_mape_table() -> pd.DataFrame:
    """Returns per-SKU MAPE for both models (for top/bottom-5 table in Executive dashboard)."""
    pred = _load_predictions()
    if pred is None:
        return pd.DataFrame()

    rows = []
    for pid, grp in pred.groupby("product_id"):
        y_true = grp["units_sold"].values
        from evaluation.metrics import compute_mape
        mape_lgb   = compute_mape(y_true, grp["lgb_pred"].fillna(0).values)
        mape_naive = compute_mape(y_true, grp["naive_pred"].fillna(0).values)
        rows.append({
            "product_id": pid,
            "mape_lgb":   round(mape_lgb, 2) if not np.isnan(mape_lgb) else None,
            "mape_naive": round(mape_naive, 2) if not np.isnan(mape_naive) else None,
        })
    return pd.DataFrame(rows).sort_values("mape_lgb")
