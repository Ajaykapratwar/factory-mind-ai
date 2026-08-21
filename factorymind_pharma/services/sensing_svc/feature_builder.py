"""
sensing_svc/feature_builder.py — Step 1.1
Builds the master feature table (L1 Sensing layer).

Join strategy:
  - Anchor: sales_demand (date × product × region × channel) — densest table.
  - Left-join demand_factors on (date, product_id).
  - Left-join orders on (date, product_id) — aggregate ordered_quantity by day+product.
  - Left-join inventory on (date, product_id) — aggregate across warehouses for sales grain.
  - Attach product_master dimensions.
  - Impute nulls per spec §3.2: forward-fill per product_id; flag imputed rows.
  - Write master_feature_table.parquet.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def _forward_fill_per_product(df: pd.DataFrame, cols: list, group_col: str = "product_id") -> pd.DataFrame:
    """
    Forward-fill selected columns within each product group (sorted by date).
    Adds a boolean `<col>_imputed` flag for each imputed column.
    """
    df = df.sort_values([group_col, "date"]).reset_index(drop=True)

    for col in cols:
        if col not in df.columns:
            continue
        was_null = df[col].isna()
        df[col] = df.groupby(group_col)[col].transform(lambda x: x.ffill())
        df[f"{col}_imputed"] = was_null & df[col].notna()

    return df


def build_master_feature_table(dfs: Dict[str, pd.DataFrame], processed_dir: str) -> pd.DataFrame:
    """
    Builds and writes master_feature_table.parquet.
    Returns the DataFrame for immediate use.

    Grain of output = sales_demand grain (date × product_id × region × channel).
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Anchor table: sales_demand
    # -----------------------------------------------------------------------
    sd = dfs["sales_demand"].copy()
    sd["date"] = pd.to_datetime(sd["date"])
    anchor_rows = len(sd)
    logger.info("Anchor table (sales_demand): %d rows", anchor_rows)

    # -----------------------------------------------------------------------
    # 2. demand_factors — aggregate to (date, product_id) then merge
    #    demand_factors has multiple rows per (date, product_id) matching
    #    the region/channel grain of sales_demand. Aggregate to a single
    #    representative row per (date, product_id) before joining.
    # -----------------------------------------------------------------------
    df_defs = dfs["demand_factors"].copy()
    df_defs["date"] = pd.to_datetime(df_defs["date"])
    # Aggregate numeric cols by mean; categorical by first
    num_cols_df = ["temperature_c", "humidity_pct", "competitor_price",
                   "market_demand_index", "disease_season_index", "regional_demand_index"]
    cat_cols_df = ["season"]
    flag_cols_df = ["promotion_flag"]

    agg_dict_df = {c: "mean" for c in num_cols_df if c in df_defs.columns}
    agg_dict_df.update({c: "first" for c in cat_cols_df if c in df_defs.columns})
    agg_dict_df.update({c: "max" for c in flag_cols_df if c in df_defs.columns})

    df_defs_agg = df_defs.groupby(["date", "product_id"], as_index=False).agg(agg_dict_df)

    mft = sd.merge(
        df_defs_agg,
        on=["date", "product_id"],
        how="left",
    )

    # -----------------------------------------------------------------------
    # 3. orders — aggregate daily ordered_quantity per (date, product_id)
    # -----------------------------------------------------------------------
    ord_df = dfs["orders"].copy()
    ord_df["date"] = pd.to_datetime(ord_df["date"])
    ord_agg = (
        ord_df.groupby(["date", "product_id"], as_index=False)
        .agg(
            total_ordered_quantity=("ordered_quantity", "sum"),
            num_orders=("order_id", "count"),
        )
    )
    mft = mft.merge(ord_agg, on=["date", "product_id"], how="left")

    # -----------------------------------------------------------------------
    # 4. inventory — aggregate across warehouses to (date, product_id) grain
    #    Executive Summary needs cross-warehouse; sales stays at region grain.
    #    We aggregate: sum opening/closing/received, max stockout_flag, mean safety/reorder.
    # -----------------------------------------------------------------------
    inv_df = dfs["inventory"].copy()
    inv_df["date"] = pd.to_datetime(inv_df["date"])
    inv_agg = (
        inv_df.groupby(["date", "product_id"], as_index=False)
        .agg(
            total_closing_stock=("closing_stock", "sum"),
            total_opening_stock=("opening_stock", "sum"),
            total_received_quantity=("received_quantity", "sum"),
            stockout_flag=("stockout_flag", "max"),       # 1 if any warehouse stocked out
            mean_safety_stock=("safety_stock", "mean"),
            mean_reorder_point=("reorder_point", "mean"),
        )
    )
    mft = mft.merge(inv_agg, on=["date", "product_id"], how="left")

    # -----------------------------------------------------------------------
    # 5. product_master dimensions
    # -----------------------------------------------------------------------
    pm = dfs["product_master"][
        ["product_id", "category", "dosage_form", "criticality", "unit_price",
         "strength_mg", "pack_size", "production_cost", "shelf_life_days"]
    ].copy()
    mft = mft.merge(pm, on="product_id", how="left")

    # -----------------------------------------------------------------------
    # 6. Null imputation — forward-fill per product_id (spec §3.2)
    # -----------------------------------------------------------------------
    impute_cols = [
        "total_ordered_quantity", "total_closing_stock", "stockout_flag",
        "market_demand_index", "disease_season_index", "promotion_flag",
        "competitor_price", "regional_demand_index",
    ]
    mft = _forward_fill_per_product(mft, impute_cols)

    # After ffill, any remaining NaN (e.g., at the very start of a product's history)
    # get filled with 0 for numeric cols
    numeric_impute_fill = {
        "total_ordered_quantity": 0.0,
        "total_closing_stock": 0.0,
        "stockout_flag": 0,
        "market_demand_index": mft["market_demand_index"].median() if "market_demand_index" in mft.columns else 0,
        "disease_season_index": 1.0,
        "promotion_flag": 0,
        "competitor_price": mft["competitor_price"].median() if "competitor_price" in mft.columns else 0,
        "regional_demand_index": 1.0,
    }
    mft.fillna(numeric_impute_fill, inplace=True)

    # -----------------------------------------------------------------------
    # 7. Verify: row count must equal anchor
    # -----------------------------------------------------------------------
    assert len(mft) == anchor_rows, (
        f"CRITICAL: master feature table has {len(mft)} rows but anchor (sales_demand) "
        f"had {anchor_rows} rows. Left join violated!"
    )

    # -----------------------------------------------------------------------
    # 8. Spot-check: verify zero nulls in key columns
    # -----------------------------------------------------------------------
    key_null_cols = ["units_sold", "revenue", "total_closing_stock"]
    for col in key_null_cols:
        if col in mft.columns:
            n_null = mft[col].isna().sum()
            if n_null > 0:
                raise ValueError(f"Column '{col}' still has {n_null} nulls after imputation!")

    # -----------------------------------------------------------------------
    # 9. Write to parquet
    # -----------------------------------------------------------------------
    out_path = processed_dir / "master_feature_table.parquet"
    mft.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Wrote master_feature_table.parquet  rows=%d  cols=%d", len(mft), len(mft.columns))

    print(f"\n✓  master_feature_table.parquet written")
    print(f"   Rows:    {len(mft):,}  (matches sales_demand anchor)")
    print(f"   Columns: {len(mft.columns)}")
    print(f"   Path:    {out_path}")

    _null_report(mft, key_null_cols)

    return mft


def _null_report(df: pd.DataFrame, cols: list) -> None:
    print("\n   Null check on critical columns:")
    for c in cols:
        if c in df.columns:
            n = df[c].isna().sum()
            status = "✓" if n == 0 else "✗"
            print(f"   {status}  {c}: {n} nulls")
