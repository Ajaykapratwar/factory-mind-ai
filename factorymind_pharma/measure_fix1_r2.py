"""
measure_fix1_r2.py
==================
Stand-alone script to:
  1. Load master_feature_table.parquet
  2. Build leakage-free features via the updated _build_expanded_features()
  3. Train §2.1 LightGBM expanded model
  4. Report row-level and product-week R² on the test split
  5. Report feature column list (verify banned columns absent)

Run from factorymind_pharma/:
  python measure_fix1_r2.py
"""

import sys, os, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from services.forecast_svc.ensemble_trainer import (
    _build_expanded_features,
    _label_encode_cats,
    _make_splits,
    _get_feature_cols,
    EXPANDED_FEATURE_COLS,
    BANNED_COLS_VERIFY,
    CAT_COLS,
)
from evaluation.metrics import (
    compute_r2,
    compute_mape,
    compute_wape,
    compute_bias_pct,
    compute_rmse,
)
import lightgbm as lgb

# ── Load config ───────────────────────────────────────────────────────────────
with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

PROCESSED_DIR = ROOT / cfg["data"]["processed_dir"]
SEED = 42

print("=" * 78)
print("  Fix 1 Verification: Leakage-free feature retraining & multi-grain evaluation")
print("=" * 78)

# ── 1. Verify banned columns absent from EXPANDED_FEATURE_COLS ───────────────
BANNED = {"total_ordered_quantity", "num_orders", "total_received_quantity"}
present = BANNED & set(EXPANDED_FEATURE_COLS)
if present:
    print(f"\n[FAIL] Banned columns still in EXPANDED_FEATURE_COLS: {present}")
    sys.exit(1)
print(f"\n[PASS] No banned raw columns in EXPANDED_FEATURE_COLS")
print(f"       Lag-1 variants present: {sorted(BANNED_COLS_VERIFY)}")

# ── 2. Load data ──────────────────────────────────────────────────────────────
t0 = time.time()
mft = pd.read_parquet(PROCESSED_DIR / "master_feature_table.parquet", engine="pyarrow")
mft["date"] = pd.to_datetime(mft["date"])
print(f"\n  Loaded MFT: {len(mft):,} rows  ({time.time()-t0:.1f}s)")
print(f"  Date range: {mft['date'].min().date()} -> {mft['date'].max().date()}")
print(f"  Products:   {mft['product_id'].nunique()}")

# ── 3. Build leakage-free features ───────────────────────────────────────────
t1 = time.time()
feat_df = _build_expanded_features(mft)
print(f"\n  Feature build: {time.time()-t1:.1f}s  ->  {len(feat_df.columns)} columns")

# Confirm no banned column in actual built features
built_cols = set(feat_df.columns)
feat_list = _get_feature_cols(feat_df)
leaky_in_feats = BANNED & set(feat_list)
if leaky_in_feats:
    print(f"\n[FAIL] Banned columns appear as training features: {leaky_in_feats}")
    sys.exit(1)
print(f"  [PASS] Banned raw columns not used as training features")

# ── 4. Train/test split ───────────────────────────────────────────────────────
train, val, test, train_end, val_end, test_start = _make_splits(feat_df, cfg)
print(f"\n  Split: train <= {train_end.date()}  |  val <= {val_end.date()}  |  test >= {test_start.date()}")
print(f"         train={len(train):,}  val={len(val):,}  test={len(test):,}")

# ── 5. Encode + clean ─────────────────────────────────────────────────────────
feat_cols = _get_feature_cols(feat_df)
print(f"\n  Training features: {len(feat_cols)}")
train_enc, test_enc, _ = _label_encode_cats(train, test, CAT_COLS)
_, val_enc, _          = _label_encode_cats(train, val,  CAT_COLS)

lag_cols   = [c for c in feat_cols if c.startswith("lag_")]
avail_feat = [c for c in feat_cols if c in train_enc.columns]

train_clean = train_enc.dropna(subset=[c for c in lag_cols if c in train_enc.columns])
val_clean   = val_enc.dropna(subset=[c for c in lag_cols if c in val_enc.columns])
test_clean  = test_enc.dropna(subset=[c for c in lag_cols if c in test_enc.columns])

X_train = train_clean[avail_feat].fillna(0)
y_train = train_clean["units_sold"]
X_val   = val_clean[avail_feat].fillna(0)
y_val   = val_clean["units_sold"]
X_test  = test_clean[avail_feat].fillna(0)
y_test  = test_clean["units_sold"].values

print(f"  After lag-dropna: train={len(train_clean):,}  val={len(val_clean):,}  test={len(test_clean):,}")

# ── 6. Train §2.1 LightGBM ───────────────────────────────────────────────────
lgb_params = cfg["model"]["lightgbm"].copy()
n_est = lgb_params.pop("n_estimators", 1500)

t2 = time.time()
model = lgb.LGBMRegressor(n_estimators=n_est, **lgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
)
preds = np.maximum(model.predict(X_test), 0)
print(f"\n  LightGBM trained in {time.time()-t2:.1f}s  (best_iter={model.best_iteration_})")

# ── 7. Evaluate at all four grains ───────────────────────────────────────────
test_df_copy = test_clean.copy()
test_df_copy["pred"] = preds

# Grain 1: Row level (transaction grain)
row_r2 = compute_r2(y_test, preds)
row_mape = compute_mape(y_test, preds)
row_wape = compute_wape(y_test, preds)
row_bias = compute_bias_pct(y_test, preds)

# Grain 2: Product-Day (Sales days only)
pd_sales = (
    test_df_copy.groupby(["product_id", "date"])
    .agg(actual=("units_sold", "sum"), pred=("pred", "sum"))
    .reset_index()
)
pds_actual = pd_sales["actual"].values
pds_pred = pd_sales["pred"].values
pds_r2 = compute_r2(pds_actual, pds_pred)
pds_mape = compute_mape(pds_actual, pds_pred)
pds_wape = compute_wape(pds_actual, pds_pred)
pds_bias = compute_bias_pct(pds_actual, pds_pred)

# Grain 3: Product-Day (All calendar days in test window)
min_test_date = test_clean["date"].min()
max_test_date = test_clean["date"].max()
test_dates = pd.date_range(min_test_date, max_test_date, freq="D")
all_test_products = test_clean["product_id"].unique()

full_pd_idx = pd.MultiIndex.from_product([all_test_products, test_dates], names=["product_id", "date"])
full_pd = pd.DataFrame(index=full_pd_idx).reset_index()
full_pd = full_pd.merge(pd_sales, on=["product_id", "date"], how="left")
full_pd["actual"] = full_pd["actual"].fillna(0.0)
full_pd["pred"] = full_pd["pred"].fillna(0.0)

pda_actual = full_pd["actual"].values
pda_pred = full_pd["pred"].values
pda_r2 = compute_r2(pda_actual, pda_pred)
pda_mape = compute_mape(pda_actual, pda_pred)
pda_wape = compute_wape(pda_actual, pda_pred)
pda_bias = compute_bias_pct(pda_actual, pda_pred)

# Grain 4: Product-Week
test_df_copy["year_week"] = test_df_copy["date"].dt.to_period("W")
pw = (
    test_df_copy.groupby(["product_id", "year_week"])
    .agg(actual=("units_sold", "sum"), pred=("pred", "sum"))
    .reset_index()
)
pw_actual = pw["actual"].values
pw_pred = pw["pred"].values
pw_r2 = compute_r2(pw_actual, pw_pred)
pw_mape = compute_mape(pw_actual, pw_pred)
pw_wape = compute_wape(pw_actual, pw_pred)
pw_bias = compute_bias_pct(pw_actual, pw_pred)

print(f"\n{'='*78}")
print(f"  FOUR-GRAIN PERFORMANCE REPORT")
print(f"{'='*78}")
header = f"  {'Grain':<32} {'R2':>8} {'MAPE %':>10} {'WAPE %':>10} {'Bias %':>10} {'Samples':>10}"
print(header)
print("  " + "-" * 74)
print(f"  {'Row (transaction)':<32} {row_r2:>8.4f} {row_mape:>10.2f} {row_wape:>10.2f} {row_bias:>10.2f} {len(y_test):>10,}")
print(f"  {'Product-day (all days)':<32} {pda_r2:>8.4f} {pda_mape:>10.2f} {pda_wape:>10.2f} {pda_bias:>10.2f} {len(full_pd):>10,}")
print(f"  {'Product-day (sales days only)':<32} {pds_r2:>8.4f} {pds_mape:>10.2f} {pds_wape:>10.2f} {pds_bias:>10.2f} {len(pd_sales):>10,}")
print(f"  {'Product-week':<32} {pw_r2:>8.4f} {pw_mape:>10.2f} {pw_wape:>10.2f} {pw_bias:>10.2f} {len(pw):>10,}")
print("=" * 78)

# ── 8. Gate check with documented origins ─────────────────────────────────────
print("\n  CRITERIA & THRESHOLDS:")
print(f"  1. Row-level R2: {row_r2:.4f}")
print(f"     Target: ~= 0.58  |  Origin: User Fix 1 specification ('Row-level R2 ~= 0.58')")
print(f"     Status: {'PASS' if row_r2 >= 0.58 else 'FAIL'} (measured {row_r2:.4f} >= 0.58)")
print(f"  2. Product-Week R2: {pw_r2:.4f}")
print(f"     Target: >= 0.80  |  Origin: evaluation/metrics.py THRESHOLDS_PHASE2['r2_min'] / Spec 5.4 (C7)")
print(f"     Status: {'PASS' if pw_r2 >= 0.80 else 'FAIL'} (measured {pw_r2:.4f} >= 0.80)")
print(f"  3. Product-Day (all days): R2={pda_r2:.4f}, WAPE={pda_wape:.2f}%")
print(f"     Target: None set  |  Origin: N/A (no threshold set by user or spec for this grain)")
print(f"  4. Product-Day (sales days only): R2={pds_r2:.4f}, WAPE={pds_wape:.2f}%")
print(f"     Target: None set  |  Origin: N/A (no threshold set by user or spec for this grain)")

# ── 9. Final feature list annotation ──────────────────────────────────────────
print("\n" + "=" * 78)
print("  FINAL EXPANDED_FEATURE_COLS WITH ANNOTATIONS")
print("=" * 78)

CALENDAR_COLS = {
    "day_of_week", "month", "day_of_month", "week_of_year",
    "is_month_start", "is_month_end", "is_weekend"
}
LAGGED_COLS = {
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28", "lag_60", "lag_90",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7", "rolling_std_28",
    "rolling_min_28", "rolling_max_28", "ewm_7", "ewm_14", "ewm_28",
    "lag1_vs_mean7", "lag7_vs_mean28", "expanding_mean", "category_demand",
    "stockout_flag_lag1", "total_closing_stock_lag1", "total_ordered_quantity_lag1",
    "num_orders_lag1", "total_received_quantity_lag1"
}

for i, col in enumerate(EXPANDED_FEATURE_COLS, 1):
    if col in CALENDAR_COLS:
        tag = "[calendar]"
    elif col in LAGGED_COLS:
        tag = "[lagged]"
    else:
        tag = "[known-in-advance]"
    print(f"  {i:2d}. {col:<32} {tag}")

print("=" * 78)

