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
    global _PRED_CACHE
    if _PRED_CACHE is None:
        path = _get_outputs_dir() / "test_predictions.parquet"
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

    # Forecast accuracy (WAPE on test set if predictions available)
    forecast_accuracy = None
    if preds is not None and "lgb_pred" in preds.columns:
        y_true = preds["units_sold"].values
        y_pred = preds["lgb_pred"].fillna(0).values
        total_actual = np.sum(np.abs(y_true))
        if total_actual > 0:
            wape = np.sum(np.abs(y_true - y_pred)) / total_actual * 100
            forecast_accuracy = round(100.0 - wape, 2)  # accuracy = 1 - WAPE

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
        "forecast_accuracy_pct": forecast_accuracy,
        "avg_days_of_supply":    avg_dos,
        "stockout_count":        stockout_count,
    }


def get_demand_vs_actual(product_id: Optional[str] = None) -> pd.DataFrame:
    """
    Returns a DataFrame of actual + naive + lgb forecasts for charting.
    If product_id is None → aggregate across all SKUs by date.
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
        pred_agg = (
            pred.groupby("date", as_index=False)
            .agg(naive_pred=("naive_pred", "sum"), lgb_pred=("lgb_pred", "sum"))
        )
        result = actual_agg.merge(pred_agg, on="date", how="left")
    else:
        result = actual_agg
        result["naive_pred"] = None
        result["lgb_pred"]   = None

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
