"""
sensing_svc/loader.py — Step 0.3
Loads and validates all 15 source CSV files.

Rules enforced:
  - Each file must have its expected columns (loud failure if missing).
  - product_id in files 02–06 must be a subset of product_master.product_id.
  - batch_id in file 06 must not repeat for the same date+product.
  - Row counts reported on completion.
  - Files 07–15 are stub-validated (schema check only).
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected column schemas for all 15 files
# ---------------------------------------------------------------------------
SCHEMAS: Dict[str, list] = {
    "product_master":      ["product_id", "product_name", "category", "dosage_form",
                            "strength_mg", "pack_size", "unit_price", "production_cost",
                            "shelf_life_days", "storage_temperature_c", "criticality"],
    "sales_demand":        ["date", "product_id", "region", "channel",
                            "units_sold", "selling_price", "discount_pct", "revenue"],
    "demand_factors":      ["date", "product_id", "season", "temperature_c",
                            "humidity_pct", "promotion_flag", "competitor_price",
                            "market_demand_index", "disease_season_index", "regional_demand_index"],
    "orders":              ["order_id", "date", "product_id", "customer_type", "region",
                            "ordered_quantity", "delivery_days_promised", "priority", "order_status"],
    "inventory":           ["date", "product_id", "warehouse_id", "opening_stock",
                            "received_quantity", "production_received", "units_sold",
                            "closing_stock", "safety_stock", "reorder_point", "stockout_flag"],
    "production":          ["date", "batch_id", "product_id", "production_line",
                            "planned_quantity", "actual_quantity", "production_time_hours",
                            "downtime_hours", "yield_percentage", "rejected_quantity"],
    # Phase 2+ stubs — schema validated only
    "batch_manufacturing": ["batch_id", "product_id", "batch_size", "start_time",
                            "operator_count", "raw_material_quantity", "process_temperature_c",
                            "process_pressure_bar", "mixing_time_min", "granulation_time_min",
                            "drying_time_min", "end_time"],
    "quality_control":     ["batch_id", "product_id", "assay_pct", "dissolution_pct",
                            "hardness_n", "moisture_pct", "impurity_level_pct",
                            "weight_variation_pct", "microbial_count", "quality_score",
                            "release_status"],
    "machine_sensor_data": ["timestamp", "machine_id", "production_line", "temperature_c",
                            "vibration_mm_s", "pressure_bar", "rpm", "power_consumption_kw",
                            "operating_hours", "downtime_minutes", "maintenance_flag"],
    "maintenance":         ["maintenance_id", "machine_id", "date", "maintenance_type",
                            "failure_type", "downtime_hours", "maintenance_cost",
                            "days_since_previous_maintenance", "next_maintenance_due_days"],
    "raw_materials":       ["raw_material_id", "raw_material_name", "supplier_id", "unit_cost",
                            "lead_time_days", "minimum_order_quantity", "available_quantity",
                            "quality_grade", "expiry_date"],
    "suppliers":           ["supplier_id", "supplier_name", "material_id", "delivery_time_days",
                            "delivery_reliability_pct", "quality_score", "unit_cost",
                            "rejection_rate_pct"],
    "logistics":           ["shipment_id", "product_id", "origin", "destination",
                            "shipping_date", "delivery_date", "transport_mode", "shipping_cost",
                            "delay_days", "temperature_excursion_flag"],
    "compliance":          ["batch_id", "inspection_date", "compliance_score",
                            "documentation_score", "deviation_count", "audit_status"],
    "financials":          ["date", "product_id", "production_cost", "raw_material_cost",
                            "labor_cost", "energy_cost", "storage_cost", "logistics_cost",
                            "revenue", "profit"],
}

# File number → logical key mapping
FILE_MAP = {
    "01": "product_master",
    "02": "sales_demand",
    "03": "demand_factors",
    "04": "orders",
    "05": "inventory",
    "06": "production",
    "07": "batch_manufacturing",
    "08": "quality_control",
    "09": "machine_sensor_data",
    "10": "maintenance",
    "11": "raw_materials",
    "12": "suppliers",
    "13": "logistics",
    "14": "compliance",
    "15": "financials",
}


def _resolve_path(raw_dir: str, filename: str) -> Path:
    p = Path(raw_dir) / filename
    if not p.exists():
        raise FileNotFoundError(f"Expected data file not found: {p}")
    return p


def load_all_files(raw_dir: str, config: dict) -> Dict[str, pd.DataFrame]:
    """
    Load all 15 source CSV files, validate schemas, and return a dict of DataFrames.
    Fails loud on missing columns or missing files.
    """
    raw_dir = Path(raw_dir)
    dfs: Dict[str, pd.DataFrame] = {}
    summary_rows = []

    file_cfg = config["data"]["files"]

    for num, key in FILE_MAP.items():
        filename = file_cfg.get(key)
        path = _resolve_path(raw_dir, filename)

        df = pd.read_csv(path, low_memory=False)

        # Schema validation — loud failure on missing columns
        expected = SCHEMAS[key]
        missing_cols = [c for c in expected if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"[{filename}] Missing expected columns: {missing_cols}. "
                f"Found: {list(df.columns)}"
            )

        # Parse date columns as datetime (UTC-naive)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        dfs[key] = df
        summary_rows.append({"file": filename, "key": key, "rows": len(df), "cols": len(df.columns)})
        logger.info("Loaded %-35s  rows=%d", filename, len(df))

    # Print summary table
    print("\n" + "=" * 60)
    print(f"  SOURCE FILE LOAD SUMMARY  ({len(dfs)} files)")
    print("=" * 60)
    print(f"{'File':<35} {'Rows':>8} {'Cols':>6}")
    print("-" * 60)
    for r in summary_rows:
        print(f"{r['file']:<35} {r['rows']:>8,} {r['cols']:>6}")
    print("=" * 60 + "\n")

    return dfs


def validate_foreign_keys(dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Validates:
    1. product_id in files 02–06 is a subset of product_master.product_id.
    2. batch_id in file 06 has no duplicates for the same date+product.
    Raises ValueError on violation.
    """
    valid_pids = set(dfs["product_master"]["product_id"].unique())

    fk_files = ["sales_demand", "demand_factors", "orders", "inventory", "production"]
    for key in fk_files:
        df = dfs[key]
        if "product_id" not in df.columns:
            continue
        orphans = set(df["product_id"].unique()) - valid_pids
        if orphans:
            raise ValueError(
                f"[{key}] Found {len(orphans)} orphan product_id(s) not in product_master: "
                f"{sorted(orphans)[:10]}{'...' if len(orphans) > 10 else ''}"
            )
        logger.info("[%s] FK check passed — all product_ids valid.", key)

    # Batch uniqueness check: one batch per date+product
    prod = dfs["production"]
    dup_mask = prod.duplicated(subset=["date", "product_id", "batch_id"], keep=False)
    if dup_mask.any():
        n_dup = dup_mask.sum()
        raise ValueError(
            f"[production] {n_dup} rows have duplicate batch_id for the same date+product."
        )
    logger.info("[production] batch_id uniqueness check passed.")

    print("✓  Foreign key validation passed — zero orphan product_ids in files 02–06.")
    print("✓  Production batch_id uniqueness validated.")
