"""
run_pipeline.py - Phase 0 + Phase 1 end-to-end pipeline runner.
Execute: python run_pipeline.py

Steps:
  Phase 0:
    0.1  Repo scaffold (already done by file creation)
    0.2  Config validation
    0.3  Load & validate all 15 source files + FK checks

  Phase 1:
    1.1  Build master feature table
    1.2  Train baseline forecast models (Naive + LightGBM)
    1.3  Evaluate and report metrics
    1.4  Verify data access layer
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add project root to sys.path so all imports resolve
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def _load_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", cfg_path)
    return cfg


def _resolve_paths(cfg: dict) -> dict:
    """Make all data paths absolute."""
    raw = PROJECT_ROOT / cfg["data"]["raw_dir"]
    proc = PROJECT_ROOT / cfg["data"]["processed_dir"]
    out  = PROJECT_ROOT / cfg["data"]["outputs_dir"]
    proc.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    return {"raw": raw, "processed": proc, "outputs": out}


# ===========================================================================
# Phase 0 - Scaffold & Contracts
# ===========================================================================

def phase0_config_verify(cfg: dict) -> None:
    """Step 0.2 - verify config parses and contains required keys."""
    required = ["data", "split", "seed", "model", "features"]
    for k in required:
        assert k in cfg, f"Config missing required key: '{k}'"
    s = cfg["split"]
    assert abs(s["train_ratio"] + s["val_ratio"] + s["test_ratio"] - 1.0) < 1e-9, \
        "Split ratios must sum to 1.0"
    print("[OK]  Config verified - all keys present, split ratios sum to 1.0")


def phase0_load_validate(cfg: dict, paths: dict):
    """Step 0.3 - load all 15 files, validate schemas and FK integrity."""
    from services.sensing_svc.loader import load_all_files, validate_foreign_keys
    dfs = load_all_files(str(paths["raw"]), cfg)
    validate_foreign_keys(dfs)
    return dfs


# ===========================================================================
# Phase 1 - Foundation: Data Layer + Core Dashboards
# ===========================================================================

def phase1_build_feature_table(dfs: dict, paths: dict) -> pd.DataFrame:
    """Step 1.1 - build master feature table."""
    from services.sensing_svc.feature_builder import build_master_feature_table
    return build_master_feature_table(dfs, str(paths["processed"]))


def phase1_train_forecast(mft: pd.DataFrame, cfg: dict, paths: dict) -> dict:
    """Step 1.2 - train naive + LightGBM baseline forecast."""
    from services.forecast_svc.trainer import run_forecast_pipeline
    return run_forecast_pipeline(mft, cfg, str(paths["outputs"]))


def phase1_evaluate(forecast_result: dict, cfg: dict, paths: dict) -> pd.DataFrame:
    """Step 1.3 - compute and report metrics, check acceptance thresholds."""
    from evaluation.metrics import (
        compute_regression_metrics,
        compute_per_sku_metrics,
        check_acceptance,
    )

    test_df = forecast_result["test_df"]
    # Drop rows with null predictions
    test_valid = test_df.dropna(subset=["lgb_pred", "naive_pred"])

    # -- Aggregate metrics: LightGBM --------------------------
    lgb_agg = compute_regression_metrics(
        test_valid["units_sold"].values,
        test_valid["lgb_pred"].values,
        label="LightGBM_aggregate"
    )

    # -- Aggregate metrics: Naive ------------------------------
    naive_agg = compute_regression_metrics(
        test_valid["units_sold"].values,
        test_valid["naive_pred"].values,
        label="Naive_aggregate"
    )

    # -- Per-SKU metrics ---------------------------------------
    sku_metrics_lgb   = compute_per_sku_metrics(test_valid, actual_col="units_sold", pred_col="lgb_pred",   label="lgb")
    sku_metrics_naive = compute_per_sku_metrics(test_valid, actual_col="units_sold", pred_col="naive_pred", label="naive")

    # -- Requirement: LGBMbeats naive on >= 60% of SKUs by MAPE --
    merged = sku_metrics_lgb[["product_id", "mape"]].rename(columns={"mape": "mape_lgb"}).merge(
        sku_metrics_naive[["product_id", "mape"]].rename(columns={"mape": "mape_naive"}),
        on="product_id"
    )
    merged = merged.dropna(subset=["mape_lgb", "mape_naive"])
    lgb_wins = (merged["mape_lgb"] < merged["mape_naive"]).sum()
    pct_wins = lgb_wins / len(merged) * 100 if len(merged) > 0 else 0

    # -- Acceptance check --------------------------------------
    acceptance = check_acceptance(lgb_agg)

    # -- Print summary -----------------------------------------
    print("\n" + "=" * 68)
    print("  EVALUATION REPORT - Phase 1 Baseline")
    print("=" * 68)
    _print_metrics("LightGBM (Aggregate)", lgb_agg)
    _print_metrics("Naive Seasonal (Aggregate)", naive_agg)

    print(f"\n  LightGBM beats Naive on MAPE: {lgb_wins}/{len(merged)} SKUs "
          f"({pct_wins:.1f}%)  - {'[OK] PASS' if pct_wins >= 60 else '[X] FAIL'} (threshold: 60%)")

    print("\n  Acceptance Check (Phase 1 thresholds):")
    for metric, result in acceptance.items():
        if metric == "all_pass":
            continue
        status = "[OK]" if result["pass"] else "[X]"
        print(f"    {status}  {metric:<12} {result['value']:.3f}  "
              f"({'?' if metric == 'r2' else '?'} {result['threshold']})")
    print(f"\n  all_pass = {acceptance['all_pass']}")
    print("=" * 68)

    # -- Save per-SKU report to CSV ----------------------------
    eval_df = sku_metrics_lgb.copy()
    eval_df = eval_df.merge(
        sku_metrics_naive[["product_id", "mape", "rmse", "mae"]].rename(
            columns={"mape": "naive_mape", "rmse": "naive_rmse", "mae": "naive_mae"}
        ),
        on="product_id", how="left"
    )
    eval_df["lgb_beats_naive"] = eval_df["mape"] < eval_df["naive_mape"]

    # Aggregate row
    agg_row = pd.DataFrame([{
        "product_id": "AGGREGATE",
        "mape": lgb_agg["mape"], "smape": lgb_agg["smape"],
        "wape": lgb_agg["wape"], "r2": lgb_agg["r2"],
        "mae": lgb_agg["mae"], "rmse": lgb_agg["rmse"],
        "bias_pct": lgb_agg["bias_pct"],
        "naive_mape": naive_agg["mape"],
        "lgb_beats_naive": pct_wins >= 60,
    }])
    eval_df = pd.concat([agg_row, eval_df], ignore_index=True)

    out_path = paths["outputs"] / "evaluation_report.csv"
    eval_df.to_csv(out_path, index=False)
    print(f"\n  [OK]  evaluation_report.csv written ? {out_path}")

    return eval_df


def _print_metrics(label: str, m: dict) -> None:
    print(f"\n  {label}:")
    for k in ["mae", "rmse", "mape", "smape", "wape", "r2", "rmse_pct", "mae_pct", "bias_pct"]:
        v = m.get(k)
        if v is not None and not np.isnan(v):
            print(f"    {k:<12} {v:.4f}")


def phase1_verify_data_layer() -> None:
    """Step 1.4 - verify each data access function returns non-empty correctly-typed output."""
    from dashboard.data_loader import get_kpi_summary, get_demand_vs_actual, get_inventory_status
    print("\n  Verifying data access layer...")

    kpi = get_kpi_summary()
    assert isinstance(kpi, dict) and len(kpi) > 0, "get_kpi_summary() returned empty/non-dict"
    print(f"    [OK]  get_kpi_summary()         keys={list(kpi.keys())}")

    demand = get_demand_vs_actual()
    assert isinstance(demand, pd.DataFrame) and len(demand) > 0, "get_demand_vs_actual() empty"
    print(f"    [OK]  get_demand_vs_actual()     rows={len(demand):,}")

    inv = get_inventory_status()
    assert isinstance(inv, pd.DataFrame) and len(inv) > 0, "get_inventory_status() empty"
    print(f"    [OK]  get_inventory_status()     rows={len(inv):,}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    print("\n" + "=" * 68)
    print("  FactoryMind Pharma - Phase 0 + Phase 1 Pipeline")
    print("=" * 68)

    # -- Phase 0 ---------------------------------------------
    print("\n[ Phase 0 - Scaffold & Contracts ]")

    cfg = _load_config()
    paths = _resolve_paths(cfg)
    phase0_config_verify(cfg)

    print("\n  Step 0.3 - Loading all 15 source files...")
    dfs = phase0_load_validate(cfg, paths)

    # -- Phase 1 ---------------------------------------------
    print("\n[ Phase 1 - Foundation ]")

    print("\n  Step 1.1 - Building master feature table...")
    mft = phase1_build_feature_table(dfs, paths)

    print("\n  Step 1.2 - Training baseline forecast models...")
    forecast_result = phase1_train_forecast(mft, cfg, paths)

    print("\n  Step 1.3 - Evaluating metrics...")
    eval_df = phase1_evaluate(forecast_result, cfg, paths)

    print("\n  Step 1.4 - Verifying data access layer...")
    phase1_verify_data_layer()

    # -- Final summary ----------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*68}")
    print(f"  [OK]  Pipeline completed in {elapsed:.1f}s")
    print(f"  [OK]  master_feature_table.parquet ? {paths['processed']}")
    print(f"  [OK]  evaluation_report.csv        ? {paths['outputs']}")
    print(f"\n  To start dashboards:")
    print(f"    cd factorymind_pharma")
    print(f"    python dashboard/app.py")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
