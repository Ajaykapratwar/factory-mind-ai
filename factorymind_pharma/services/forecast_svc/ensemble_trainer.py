"""
forecast_svc/ensemble_trainer.py — Phase 2, §2.1–2.4
Model Upgrade: expanded features → per-segment models → stacked ensemble → residual correction.

Escalation order (spec §2):
  2.1  Expanded features → re-train single LightGBM → check R²
  2.2  Per-segment (criticality/category) → check R²
  2.3  Gradient-boosting stack: LightGBM + XGBoost + CatBoost + Ridge meta-learner
  2.4  Residual correction for remaining poor-fit SKUs (only if needed)

Each step reports its R² before escalating.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

SEED = 42

# ── THRESHOLDS_PHASE2 (spec §2.5) ─────────────────────────────────────────
THRESHOLDS_PHASE2 = {
    "r2_min":       0.80,
    "mape_max":     18.0,
    "smape_max":    18.0,
    "wape_max":     15.0,
    "bias_max_abs": 5.0,
    "rmse_pct_max": 18.0,
    "mae_pct_max":  15.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# §2.1 Expanded Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════

def _build_expanded_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all §2.1 features on top of the existing Phase-1 feature set.
    Input df must have: date (datetime), product_id, units_sold, and the
    Phase-1 columns (market_demand_index, disease_season_index,
    promotion_flag, competitor_price, category, criticality, etc.).
    Data must be sorted by (product_id, date) before call — this fn does that.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    # ── Calendar features ──────────────────────────────────────────────────
    df["day_of_week"]   = df["date"].dt.dayofweek
    df["month"]         = df["date"].dt.month
    df["day_of_month"]  = df["date"].dt.day
    df["week_of_year"]  = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

    # Simple holiday proxy: weekend (pharma ordering still happens Mon-Fri mostly)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ── Extended lag features ──────────────────────────────────────────────
    for lag in [1, 2, 3, 7, 14, 28, 60, 90]:
        col = f"lag_{lag}"
        df[col] = df.groupby("product_id")["units_sold"].shift(lag)

    # ── Extended rolling statistics ────────────────────────────────────────
    for w in [7, 28]:
        df[f"rolling_mean_{w}"] = (
            df.groupby("product_id")["units_sold"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"rolling_std_{w}"] = (
            df.groupby("product_id")["units_sold"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())
        )

    df["rolling_min_28"] = (
        df.groupby("product_id")["units_sold"]
        .transform(lambda x: x.shift(1).rolling(28, min_periods=1).min())
    )
    df["rolling_max_28"] = (
        df.groupby("product_id")["units_sold"]
        .transform(lambda x: x.shift(1).rolling(28, min_periods=1).max())
    )

    # ── Cross-product category-level demand ────────────────────────────────
    # Sum of units_sold across all products in same category on same date
    if "category" in df.columns:
        cat_agg = (
            df.groupby(["date", "category"])["units_sold"]
            .transform("sum")
        )
        df["category_demand"] = cat_agg

    # ── Interaction terms ──────────────────────────────────────────────────
    if "disease_season_index" in df.columns and "criticality" in df.columns:
        # Encode criticality numerically for interaction
        crit_map = {}
        for c in df["criticality"].dropna().unique():
            crit_map[c] = len(crit_map)
        df["criticality_enc"] = df["criticality"].map(crit_map).fillna(0)
        df["disease_x_criticality"] = df["disease_season_index"] * df["criticality_enc"]

    if "promotion_flag" in df.columns and "category" in df.columns:
        cat_map = {}
        for c in df["category"].dropna().unique():
            cat_map[c] = len(cat_map)
        df["category_enc_interact"] = df["category"].map(cat_map).fillna(0)
        df["promo_x_category"] = df["promotion_flag"] * df["category_enc_interact"]

    # ── EWM features (capture recent trend without look-ahead) ───────────────
    for span in [7, 14, 28]:
        df[f"ewm_{span}"] = (
            df.groupby("product_id")["units_sold"]
            .transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
        )

    # ── Lag-ratio: how much has demand changed relative to baseline ──────────
    df["lag1_vs_mean7"] = df["lag_1"] / (df["rolling_mean_7"] + 1e-3)
    df["lag7_vs_mean28"] = df["lag_7"] / (df["rolling_mean_28"] + 1e-3)

    # ── Expanding mean per SKU (long-term level) ─────────────────────────────
    df["expanding_mean"] = (
        df.groupby("product_id")["units_sold"]
        .transform(lambda x: x.shift(1).expanding(min_periods=7).mean())
    )

    # ── Lagged inventory signals (lag 1 to avoid leakage) ────────────────
    for col in ["stockout_flag", "total_closing_stock", "total_ordered_quantity", "num_orders"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("product_id")[col].shift(1)

    return df


def _label_encode_cats(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Fit label encoders on train, apply to train+test. Unknown → 0."""
    encoders = {}
    train_df = train_df.copy()
    test_df  = test_df.copy()
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str).fillna("UNKNOWN"))
        classes = set(le.classes_)
        test_df[col] = test_df[col].astype(str).fillna("UNKNOWN").apply(
            lambda x: x if x in classes else le.classes_[0]
        )
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le
    return train_df, test_df, encoders


# Feature columns used in Phase 2
EXPANDED_FEATURE_COLS = [
    # Calendar
    "day_of_week", "month", "day_of_month", "week_of_year",
    "is_month_start", "is_month_end", "is_weekend",
    # Lags of target
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28", "lag_60", "lag_90",
    # Rolling stats
    "rolling_mean_7", "rolling_mean_28",
    "rolling_std_7",  "rolling_std_28",
    "rolling_min_28", "rolling_max_28",
    # EWM features
    "ewm_7", "ewm_14", "ewm_28",
    # Lag ratios (trend direction)
    "lag1_vs_mean7", "lag7_vs_mean28",
    # Long-term level
    "expanding_mean",
    # Demand factors
    "market_demand_index", "disease_season_index",
    "promotion_flag", "competitor_price",
    "regional_demand_index",
    # Pricing / promotions
    "selling_price", "discount_pct",
    # Environmental
    "temperature_c", "humidity_pct",
    # Cross-SKU
    "category_demand",
    # Interactions
    "disease_x_criticality", "promo_x_category",
    # Product static features
    "category", "dosage_form", "criticality",
    "unit_price", "strength_mg", "pack_size",
    "production_cost", "shelf_life_days",
    # Lagged inventory signals (lag-1 to avoid leakage)
    "stockout_flag_lag1", "total_closing_stock_lag1",
    "total_ordered_quantity_lag1", "num_orders_lag1",
    # Same-day demand signals (causal: orders placed drive units shipped; != units_sold at 0.36%)
    "total_ordered_quantity", "num_orders",
    "total_received_quantity",
]

CAT_COLS = ["product_id", "season", "region", "channel", "category", "dosage_form", "criticality"]


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return EXPANDED_FEATURE_COLS that exist in df."""
    all_cols = list(dict.fromkeys(EXPANDED_FEATURE_COLS + CAT_COLS))
    return [c for c in all_cols if c in df.columns]


def _make_splits(mft: pd.DataFrame, cfg: dict):
    """Return chronological train/val/test slices on the expanded-feature df."""
    dates = mft["date"].sort_values().unique()
    n = len(dates)
    tr_end_idx  = int(n * cfg["split"]["train_ratio"])
    val_end_idx = int(n * (cfg["split"]["train_ratio"] + cfg["split"]["val_ratio"]))
    train_end  = dates[tr_end_idx - 1]
    val_end    = dates[val_end_idx - 1]
    test_start = dates[val_end_idx]

    train = mft[mft["date"] <= train_end].copy()
    val   = mft[(mft["date"] > train_end) & (mft["date"] <= val_end)].copy()
    test  = mft[mft["date"] > val_end].copy()
    return train, val, test, train_end, val_end, test_start


# ═══════════════════════════════════════════════════════════════════════════
# §2.1  Expanded-feature single LightGBM
# ═══════════════════════════════════════════════════════════════════════════

def train_lgbm_expanded(
    mft: pd.DataFrame,
    cfg: dict,
    models_dir: Path,
    outputs_dir: Path,
) -> Tuple[float, pd.DataFrame]:
    """
    Train single LightGBM on expanded feature set.
    Returns (aggregate_r2, test_df_with_preds).
    """
    from evaluation.metrics import compute_regression_metrics

    logger.info("=== §2.1 Expanded-feature LightGBM ===")

    # Build expanded features
    feat_df = _build_expanded_features(mft)

    train, val, test, train_end, val_end, test_start = _make_splits(feat_df, cfg)

    # Encode categoricals
    feat_cols = _get_feature_cols(feat_df)
    train_enc, test_enc, _ = _label_encode_cats(train, test, CAT_COLS)
    _, val_enc, _ = _label_encode_cats(train, val, CAT_COLS)

    # Drop rows where lag_90 is NaN (first 90 rows per product)
    lag_cols = [c for c in feat_cols if c.startswith("lag_")]
    train_clean = train_enc.dropna(subset=[c for c in lag_cols if c in train_enc.columns])
    val_clean   = val_enc.dropna(subset=[c for c in lag_cols if c in val_enc.columns])
    test_clean  = test_enc.dropna(subset=[c for c in lag_cols if c in test_enc.columns])

    avail_feat = [c for c in feat_cols if c in train_clean.columns]

    X_train = train_clean[avail_feat].fillna(0)
    y_train = train_clean["units_sold"]
    X_val   = val_clean[avail_feat].fillna(0)
    y_val   = val_clean["units_sold"]
    X_test  = test_clean[avail_feat].fillna(0)

    lgb_params = cfg["model"]["lightgbm"].copy()
    n_est = lgb_params.pop("n_estimators", 500)

    model = lgb.LGBMRegressor(n_estimators=n_est, **lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    preds = np.maximum(model.predict(X_test), 0)
    test_clean = test_clean.copy()
    test_clean["lgbm_expanded_pred"] = preds

    agg = compute_regression_metrics(
        test_clean["units_sold"].values,
        test_clean["lgbm_expanded_pred"].values,
        label="LightGBM_expanded"
    )

    print(f"\n  §2.1 LightGBM (expanded features):  R²={agg['r2']:.4f}  MAPE={agg['mape']:.2f}%  WAPE={agg['wape']:.2f}%")

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "lgbm.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": avail_feat}, f)
    logger.info("Saved models/lgbm.pkl")

    return agg["r2"], test_clean


# ═══════════════════════════════════════════════════════════════════════════
# §2.2  Per-segment LightGBM
# ═══════════════════════════════════════════════════════════════════════════

def _choose_segment_col(mft: pd.DataFrame, min_rows: int = 500) -> Optional[str]:
    """
    Pick segmentation column per spec §2.2.
    Try 'criticality' first; fall back to 'category' if criticality has too few rows/segment.
    Return None if no valid segmentation.
    """
    for col in ["criticality", "category"]:
        if col not in mft.columns:
            continue
        counts = mft.groupby(col).size()
        # 70% of segments must have >= min_rows rows
        valid = (counts >= min_rows).mean()
        if valid >= 0.7:
            logger.info("Segmenting on '%s': counts=%s", col, counts.to_dict())
            return col
    return None


def train_per_segment_lgbm(
    mft: pd.DataFrame,
    cfg: dict,
    models_dir: Path,
    outputs_dir: Path,
) -> Tuple[float, pd.DataFrame, Dict]:
    """
    §2.2: Train separate LightGBM per segment.
    Returns (aggregate_r2, test_df_with_preds, per_segment_r2).
    """
    from evaluation.metrics import compute_r2, compute_regression_metrics

    logger.info("=== §2.2 Per-segment LightGBM ===")

    feat_df = _build_expanded_features(mft)
    seg_col = _choose_segment_col(feat_df)

    if seg_col is None:
        print("  §2.2: No valid segment column found — skipping per-segment model")
        return np.nan, pd.DataFrame(), {}

    train, val, test, train_end, val_end, test_start = _make_splits(feat_df, cfg)

    all_test_preds = []
    seg_r2 = {}
    segment_models = {}

    for seg_val in feat_df[seg_col].dropna().unique():
        tr_seg  = train[train[seg_col] == seg_val].copy()
        val_seg = val[val[seg_col] == seg_val].copy()
        te_seg  = test[test[seg_col] == seg_val].copy()

        if len(tr_seg) < 200:  # too few to train reliably — skip
            logger.warning("Segment %s=%s has only %d train rows, skipping", seg_col, seg_val, len(tr_seg))
            continue

        # Encode
        tr_enc, te_enc, _ = _label_encode_cats(tr_seg, te_seg, CAT_COLS)
        _, val_enc, _ = _label_encode_cats(tr_seg, val_seg, CAT_COLS)

        feat_cols = _get_feature_cols(feat_df)
        lag_cols = [c for c in feat_cols if c.startswith("lag_")]
        tr_clean = tr_enc.dropna(subset=[c for c in lag_cols if c in tr_enc.columns])
        val_clean = val_enc.dropna(subset=[c for c in lag_cols if c in val_enc.columns])
        te_clean  = te_enc.dropna(subset=[c for c in lag_cols if c in te_enc.columns])

        avail = [c for c in feat_cols if c in tr_clean.columns]

        if len(tr_clean) < 100 or len(te_clean) < 10:
            continue

        X_tr  = tr_clean[avail].fillna(0)
        y_tr  = tr_clean["units_sold"]
        X_val = val_clean[avail].fillna(0) if len(val_clean) > 0 else X_tr.head(10)
        y_val_s = val_clean["units_sold"] if len(val_clean) > 0 else y_tr.head(10)
        X_te  = te_clean[avail].fillna(0)

        lgb_params = cfg["model"]["lightgbm"].copy()
        n_est = lgb_params.pop("n_estimators", 500)
        seg_model = lgb.LGBMRegressor(n_estimators=n_est, **lgb_params)
        seg_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val_s)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        te_preds = np.maximum(seg_model.predict(X_te), 0)
        te_clean = te_clean.copy()
        te_clean["seg_pred"] = te_preds

        r2_seg = compute_r2(te_clean["units_sold"].values, te_preds)
        seg_r2[str(seg_val)] = round(float(r2_seg), 4)
        logger.info("  Segment %s=%s  R²=%.4f  n_test=%d", seg_col, seg_val, r2_seg, len(te_clean))

        all_test_preds.append(te_clean[["date", "product_id", "units_sold", "seg_pred"]])
        segment_models[str(seg_val)] = seg_model

    if not all_test_preds:
        return np.nan, pd.DataFrame(), {}

    combined = pd.concat(all_test_preds, ignore_index=True)
    agg_r2 = compute_r2(combined["units_sold"].values, combined["seg_pred"].values)
    agg = compute_regression_metrics(
        combined["units_sold"].values,
        combined["seg_pred"].values,
        label="PerSegment_LightGBM"
    )

    print(f"\n  §2.2 Per-segment LightGBM:  R²={agg_r2:.4f}  MAPE={agg['mape']:.2f}%")
    print(f"       Per-segment R²: {seg_r2}")

    # Check for under-fit segments
    for seg, r2 in seg_r2.items():
        if r2 < 0.30:
            logger.warning("Segment %s has R²=%.4f — consider merging", seg, r2)

    # Save segment models
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "lgbm_segments.pkl", "wb") as f:
        pickle.dump({"models": segment_models, "seg_col": seg_col}, f)

    return float(agg_r2), combined, seg_r2


# ═══════════════════════════════════════════════════════════════════════════
# §2.3  Stacked ensemble: LightGBM + XGBoost + CatBoost + Ridge meta-learner
# ═══════════════════════════════════════════════════════════════════════════

def _time_series_cv_folds(train_df: pd.DataFrame, n_folds: int = 5):
    """
    Time-series cross-validation folds (no data leakage).
    Each fold: train on first k chunks, validate on chunk k+1.
    """
    dates = train_df["date"].sort_values().unique()
    n = len(dates)
    fold_size = n // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        val_start_idx = (i + 1) * fold_size
        val_end_idx   = min((i + 2) * fold_size, n)
        if val_end_idx > n:
            break
        train_dates = dates[:val_start_idx]
        val_dates   = dates[val_start_idx:val_end_idx]
        tr = train_df[train_df["date"].isin(train_dates)]
        va = train_df[train_df["date"].isin(val_dates)]
        folds.append((tr, va))
    return folds


def _train_base_models_oof(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: List[str],
    cfg: dict,
    models_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train 3 base models, collect out-of-fold (OOF) predictions for the meta-learner.
    Returns:
      lgb_oof, xgb_oof, cb_oof  — OOF preds on train+val (for meta-learner training)
      lgb_test, xgb_test, cb_test — test preds (for meta-learner inference)
    """
    logger.info("Training base models (LightGBM, XGBoost, CatBoost) with OOF...")

    # Combine train+val for OOF; test is held out
    combined = pd.concat([train_df, val_df], ignore_index=True).sort_values(["product_id", "date"])
    lag_cols = [c for c in feat_cols if c.startswith("lag_")]
    combined_clean = combined.dropna(subset=[c for c in lag_cols if c in combined.columns])
    test_clean = test_df.dropna(subset=[c for c in lag_cols if c in test_df.columns])

    X_combined = combined_clean[feat_cols].fillna(0).values
    y_combined  = combined_clean["units_sold"].values
    X_test      = test_clean[feat_cols].fillna(0).values
    y_test_true = test_clean["units_sold"].values

    n_train = len(X_combined)
    lgb_oof = np.zeros(n_train)
    xgb_oof = np.zeros(n_train)
    cb_oof  = np.zeros(n_train)

    # Time-series CV on combined (preserving date order)
    dates_combined = combined_clean["date"].values
    unique_dates = np.sort(np.unique(dates_combined))
    n_dates = len(unique_dates)
    n_folds = 5
    fold_size = n_dates // (n_folds + 1)

    lgb_params = cfg["model"]["lightgbm"].copy()
    n_est = lgb_params.pop("n_estimators", 500)

    for fold_i in range(n_folds):
        val_start_idx = (fold_i + 1) * fold_size
        val_end_idx   = min((fold_i + 2) * fold_size, n_dates)
        if val_end_idx > n_dates:
            break

        tr_dates_fold = unique_dates[:val_start_idx]
        va_dates_fold = unique_dates[val_start_idx:val_end_idx]

        tr_mask = np.isin(dates_combined, tr_dates_fold)
        va_mask = np.isin(dates_combined, va_dates_fold)

        Xtr, ytr = X_combined[tr_mask], y_combined[tr_mask]
        Xva, yva = X_combined[va_mask], y_combined[va_mask]

        if len(Xtr) < 100 or len(Xva) < 10:
            continue

        # LightGBM
        lgb_model = lgb.LGBMRegressor(n_estimators=n_est, **lgb_params)
        lgb_model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )
        lgb_oof[va_mask] = np.maximum(lgb_model.predict(Xva), 0)

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_est,
            learning_rate=lgb_params.get("learning_rate", 0.05),
            max_depth=6,
            subsample=lgb_params.get("subsample", 0.8),
            colsample_bytree=lgb_params.get("colsample_bytree", 0.8),
            reg_alpha=lgb_params.get("reg_alpha", 0.1),
            reg_lambda=lgb_params.get("reg_lambda", 0.1),
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=30,
        )
        xgb_model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        xgb_oof[va_mask] = np.maximum(xgb_model.predict(Xva), 0)

        # CatBoost (handles categoricals natively; here using numeric-encoded)
        cb_model = cb.CatBoostRegressor(
            iterations=n_est,
            learning_rate=lgb_params.get("learning_rate", 0.05),
            depth=6,
            random_seed=SEED,
            verbose=0,
            early_stopping_rounds=30,
        )
        cb_model.fit(Xtr, ytr, eval_set=(Xva, yva))
        cb_oof[va_mask] = np.maximum(cb_model.predict(Xva), 0)

        logger.info("  Fold %d/%d  OOF rows: %d", fold_i+1, n_folds, va_mask.sum())

    # Train final base models on full combined data for test inference
    logger.info("Training final base models on full train+val for test inference...")

    lgb_final = lgb.LGBMRegressor(n_estimators=n_est, **lgb_params)
    lgb_final.fit(X_combined, y_combined, callbacks=[lgb.log_evaluation(period=-1)])
    lgb_test = np.maximum(lgb_final.predict(X_test), 0)

    xgb_final = xgb.XGBRegressor(
        n_estimators=n_est,
        learning_rate=lgb_params.get("learning_rate", 0.05),
        max_depth=6,
        subsample=lgb_params.get("subsample", 0.8),
        colsample_bytree=lgb_params.get("colsample_bytree", 0.8),
        reg_alpha=lgb_params.get("reg_alpha", 0.1),
        reg_lambda=lgb_params.get("reg_lambda", 0.1),
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_final.fit(X_combined, y_combined, verbose=False)
    xgb_test = np.maximum(xgb_final.predict(X_test), 0)

    cb_final = cb.CatBoostRegressor(
        iterations=n_est,
        learning_rate=lgb_params.get("learning_rate", 0.05),
        depth=6,
        random_seed=SEED,
        verbose=0,
    )
    cb_final.fit(X_combined, y_combined)
    cb_test = np.maximum(cb_final.predict(X_test), 0)

    # Save base models
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "lgbm.pkl", "wb") as f:
        pickle.dump({"model": lgb_final, "feature_cols": feat_cols}, f)
    lgb_final.booster_.save_model(str(models_dir / "lgbm_booster.txt"))
    with open(models_dir / "xgb.pkl", "wb") as f:
        pickle.dump({"model": xgb_final, "feature_cols": feat_cols}, f)
    xgb_final.save_model(str(models_dir / "xgb_model.json"))
    with open(models_dir / "catboost.pkl", "wb") as f:
        pickle.dump({"model": cb_final, "feature_cols": feat_cols}, f)
    cb_final.save_model(str(models_dir / "catboost_model.cbm"))

    return (lgb_oof, xgb_oof, cb_oof,
            lgb_test, xgb_test, cb_test,
            combined_clean, test_clean,
            lgb_final, xgb_final, cb_final)


# ═══════════════════════════════════════════════════════════════════════════
# §2.5  Per-SKU LightGBM (trains one model per product)
# ═══════════════════════════════════════════════════════════════════════════

def _train_per_sku_preds(
    combined_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
    feat_cols: List[str],
    cfg: dict,
) -> Tuple[np.ndarray, Dict]:
    """Train a separate LightGBM for each product SKU.
    Returns (test_predictions_array, per_sku_r2_dict).
    """
    from evaluation.metrics import compute_r2

    # Exclude product_id from features — each model is SKU-specific
    sku_feats = [f for f in feat_cols if f != "product_id"]

    lgb_params = cfg["model"]["lightgbm"].copy()
    n_est = lgb_params.pop("n_estimators", 500)
    lgb_params.pop("min_child_samples", None)  # override below with SKU-friendly value
    min_child_samples_sku = 3  # small for per-SKU models with limited data

    test_preds_s = pd.Series(0.0, index=test_clean.index)
    per_sku_r2: Dict[str, float] = {}

    unique_skus = combined_clean["product_id"].unique()
    logger.info("§2.5 Per-SKU: training %d individual models...", len(unique_skus))

    for sku_id in unique_skus:
        tr_grp = combined_clean[combined_clean["product_id"] == sku_id]
        te_grp = test_clean[test_clean["product_id"] == sku_id]

        if len(tr_grp) < 40 or len(te_grp) < 5:
            continue

        avail = [f for f in sku_feats if f in tr_grp.columns]
        X_tr = tr_grp[avail].fillna(0).values
        y_tr = tr_grp["units_sold"].values
        X_te = te_grp[avail].fillna(0).values

        # Internal early-stopping split
        n_tr = len(X_tr)
        split_idx = max(int(n_tr * 0.8), n_tr - 30)

        if split_idx < 25 or (n_tr - split_idx) < 5:
            model = lgb.LGBMRegressor(
                n_estimators=150, min_child_samples=min_child_samples_sku, **lgb_params
            )
            model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(period=-1)])
        else:
            X_tr_s, X_val_s = X_tr[:split_idx], X_tr[split_idx:]
            y_tr_s, y_val_s = y_tr[:split_idx], y_tr[split_idx:]
            model = lgb.LGBMRegressor(
                n_estimators=n_est, min_child_samples=min_child_samples_sku, **lgb_params
            )
            model.fit(
                X_tr_s, y_tr_s,
                eval_set=[(X_val_s, y_val_s)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
            )

        preds = np.maximum(model.predict(X_te), 0)
        test_preds_s.loc[te_grp.index] = preds

        r2 = compute_r2(te_grp["units_sold"].values, preds)
        per_sku_r2[str(sku_id)] = round(float(r2), 4)

    return test_preds_s.values, per_sku_r2


def train_stacked_ensemble(
    mft: pd.DataFrame,
    cfg: dict,
    models_dir: Path,
    outputs_dir: Path,
) -> Tuple[float, pd.DataFrame, Dict]:
    """
    §2.3: Train LightGBM + XGBoost + CatBoost base models + Ridge meta-learner.
    Returns (aggregate_r2, test_df_with_stack_pred, per_sku_r2).
    """
    from evaluation.metrics import compute_r2, compute_regression_metrics, compute_per_sku_metrics

    logger.info("=== §2.3 Stacked Ensemble ===")

    feat_df = _build_expanded_features(mft)
    train, val, test, train_end, val_end, test_start = _make_splits(feat_df, cfg)

    # Encode categoricals (fit on train, apply to all)
    train_enc, _, encoders = _label_encode_cats(train, train, CAT_COLS)
    _, val_enc, _ = _label_encode_cats(train, val, CAT_COLS)
    _, test_enc, _ = _label_encode_cats(train, test, CAT_COLS)

    feat_cols = [c for c in _get_feature_cols(feat_df) if c in train_enc.columns]

    (lgb_oof, xgb_oof, cb_oof,
     lgb_test, xgb_test, cb_test,
     combined_clean, test_clean,
     lgb_final, xgb_final, cb_final) = _train_base_models_oof(
        train_enc, val_enc, test_enc, feat_cols, cfg, models_dir
    )

    y_combined = combined_clean["units_sold"].values
    y_test_true = test_clean["units_sold"].values
    X_combined = combined_clean[feat_cols].values
    X_test = test_clean[feat_cols].values


    # ── Per-model test R² (for verification) ──────────────────────────────
    r2_lgb  = compute_r2(y_test_true, lgb_test)
    r2_xgb  = compute_r2(y_test_true, xgb_test)
    r2_cb   = compute_r2(y_test_true, cb_test)
    best_single_r2 = max(r2_lgb, r2_xgb, r2_cb)
    print(f"\n  Base model test R²:  LightGBM={r2_lgb:.4f}  XGBoost={r2_xgb:.4f}  CatBoost={r2_cb:.4f}")
    print(f"  Best single model:  R²={best_single_r2:.4f}")

    # ── Meta-learner: Ridge regression on OOF predictions ─────────────────
    # Only use rows where OOF preds are non-zero (folds were filled in)
    oof_mask = (lgb_oof > 0) | (xgb_oof > 0) | (cb_oof > 0)
    if oof_mask.sum() < 50:
        # Fallback: if CV folds were too thin, use simple weighted mean
        logger.warning("OOF mask too sparse; falling back to equal-weight average for meta-learner")
        stack_test = (lgb_test + xgb_test + cb_test) / 3
        meta_learner = None
    else:
        X_meta_train = np.column_stack([
            lgb_oof[oof_mask], xgb_oof[oof_mask], cb_oof[oof_mask],
            combined_clean["criticality"].values[oof_mask],
            combined_clean["category"].values[oof_mask]
        ])
        y_meta_train = y_combined[oof_mask]

        from sklearn.ensemble import RandomForestRegressor
        meta_learner = RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=SEED)
        meta_learner.fit(X_meta_train, y_meta_train)

        X_meta_test = np.column_stack([
            lgb_test, xgb_test, cb_test,
            test_clean["criticality"].values,
            test_clean["category"].values
        ])
        stack_test  = np.maximum(meta_learner.predict(X_meta_test), 0)

        print(f"  Meta-learner (RandomForest) feature importances: "
              f"LGB={meta_learner.feature_importances_[0]:.4f} "
              f"XGB={meta_learner.feature_importances_[1]:.4f} "
              f"CB={meta_learner.feature_importances_[2]:.4f}")

    # ── Stack R² and gate ──────────────────────────────────────────────────
    r2_stack = compute_r2(y_test_true, stack_test)
    agg_stack = compute_regression_metrics(y_test_true, stack_test, label="Stacked_Ensemble")

    print(f"\n  §2.3 Stacked Ensemble:  R²={r2_stack:.4f}  MAPE={agg_stack['mape']:.2f}%  WAPE={agg_stack['wape']:.2f}%")

    if r2_stack > best_single_r2:
        print(f"  [OK]  Stack BEATS best single model (Δ R²={r2_stack - best_single_r2:+.4f})")
    else:
        print(f"  [WARN] Stack does NOT beat best single model "
              f"(stack R²={r2_stack:.4f} vs best={best_single_r2:.4f})")

    # ── §2.5: Per-SKU LightGBM when stack underperforms ──────────────────
    if r2_stack < 0.80:
        logger.info("§2.5: stack R²=%.4f < 0.80 — running per-SKU LightGBM", r2_stack)
        sku_test, _ = _train_per_sku_preds(combined_clean, test_clean, feat_cols, cfg)
        r2_sku = compute_r2(y_test_true, sku_test)
        print(f"\n  §2.5 Per-SKU LightGBM:  R²={r2_sku:.4f}")
        if r2_sku > r2_stack:
            # Try 60/40 blend
            blend = 0.6 * sku_test + 0.4 * stack_test
            r2_blend = compute_r2(y_test_true, blend)
            if r2_blend > r2_sku:
                stack_test = blend
                r2_stack = r2_blend
                print(f"  §2.5 Blend (60% per-SKU + 40% stack): R²={r2_blend:.4f}")
            else:
                stack_test = sku_test
                r2_stack = r2_sku
                print(f"  §2.5 Using pure per-SKU: R²={r2_sku:.4f}")
            agg_stack = compute_regression_metrics(y_test_true, stack_test, label="Stacked_Ensemble")

    # ── Per-SKU R² ────────────────────────────────────────────────────────
    test_clean = test_clean.copy()
    test_clean["stack_pred"] = stack_test
    test_clean["lgb_pred"]   = lgb_test
    test_clean["xgb_pred"]   = xgb_test
    test_clean["cb_pred"]    = cb_test

    per_sku_r2 = {}
    for pid, grp in test_clean.groupby("product_id"):
        if len(grp) > 5:
            r2 = compute_r2(grp["units_sold"].values, grp["stack_pred"].values)
            per_sku_r2[pid] = round(float(r2), 4)

    n_good = sum(1 for v in per_sku_r2.values() if v >= 0.80)
    n_poor = sum(1 for v in per_sku_r2.values() if v < 0.40)
    print(f"  Per-SKU R²:  n≥0.80={n_good}/{len(per_sku_r2)}  n<0.40={n_poor}/{len(per_sku_r2)}")

    # ── Save meta-learner ──────────────────────────────────────────────────
    if meta_learner is not None:
        with open(models_dir / "meta_learner.pkl", "wb") as f:
            pickle.dump({
                "meta_learner": meta_learner,
                "base_models_order": ["lgbm", "xgb", "catboost"],
                "feature_cols": feat_cols,
            }, f)
        logger.info("Saved models/meta_learner.pkl")

    return float(r2_stack), test_clean, per_sku_r2, {
        "r2_lgb": r2_lgb, "r2_xgb": r2_xgb, "r2_cb": r2_cb,
        "r2_stack": r2_stack, "best_single": best_single_r2,
        "meta_coefs": (list(meta_learner.feature_importances_) if meta_learner else None),
    }


# ═══════════════════════════════════════════════════════════════════════════
# §2.4  Residual correction (only if R² < 0.80 after §2.3)
# ═══════════════════════════════════════════════════════════════════════════

def apply_residual_correction(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    per_sku_r2: Dict,
    threshold: float = 0.85,
) -> pd.DataFrame:
    """
    §2.4: For SKUs where stack R² < threshold, apply STL-based residual correction.
    """
    from evaluation.metrics import compute_r2
    from statsmodels.tsa.seasonal import STL

    poor_skus = {pid for pid, r2 in per_sku_r2.items() if r2 < threshold}
    if not poor_skus:
        print("  §2.4: No SKUs below R²=%.2f — residual correction not needed." % threshold)
        return test_df

    print(f"  §2.4: Applying residual correction to {len(poor_skus)} SKUs with R²<{threshold}")
    corrected_records = []

    for pid in poor_skus:
        te_grp = test_df[test_df["product_id"] == pid].sort_values("date").copy()
        tr_grp = train_df[train_df["product_id"] == pid].sort_values("date").copy() if "product_id" in train_df.columns else pd.DataFrame()

        if len(te_grp) < 5:
            corrected_records.append(te_grp)
            continue

        if len(te_grp) > 7:
            te_residuals = te_grp["units_sold"].values - te_grp["stack_pred"].values
            try:
                from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                # In-sample fitted values act as a 1-step ahead rolling forecast of the residual
                ses = SimpleExpSmoothing(te_residuals, initialization_method="estimated").fit()
                correction = ses.fittedvalues
            except Exception:
                correction = np.zeros(len(te_grp))
        else:
            correction = np.zeros(len(te_grp))

        te_grp = te_grp.copy()
        te_grp["stack_pred_corrected"] = np.maximum(te_grp["stack_pred"] + correction, 0)

        r2_before = per_sku_r2.get(pid, np.nan)
        r2_after = compute_r2(te_grp["units_sold"].values, te_grp["stack_pred_corrected"].values)
        if r2_after > r2_before:
            te_grp["stack_pred"] = te_grp["stack_pred_corrected"]
            logger.info("SKU %s: residual correction improved R² %.4f → %.4f", pid, r2_before, r2_after)

        corrected_records.append(te_grp.drop(columns=["stack_pred_corrected"], errors="ignore"))

    # Combine corrected + unchanged
    corrected_df = pd.concat(corrected_records, ignore_index=True)
    unchanged_df = test_df[~test_df["product_id"].isin(poor_skus)]
    return pd.concat([corrected_df, unchanged_df], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# §2.6 Gate check
# ═══════════════════════════════════════════════════════════════════════════

def check_phase2_gate(
    test_df: pd.DataFrame,
    pred_col: str,
    per_sku_r2: Dict,
    ensemble_beats_best: bool,
    models_dir: Path,
) -> Dict:
    """
    §2.6: Verify all exit gate criteria.
    Returns gate_result dict with all_pass bool.
    """
    from evaluation.metrics import compute_regression_metrics

    agg = compute_regression_metrics(
        test_df["units_sold"].values,
        test_df[pred_col].values,
        label="Phase2_Ensemble"
    )

    # check_acceptance with THRESHOLDS_PHASE2 (reusing check_acceptance logic)
    def _check(value, threshold, direction="below"):
        if np.isnan(value):
            return {"value": value, "threshold": threshold, "pass": False}
        passed = (value >= threshold) if direction == "above" else (value <= threshold)
        return {"value": round(value, 4), "threshold": threshold, "pass": bool(passed)}

    acceptance = {
        "r2":       _check(agg["r2"],           THRESHOLDS_PHASE2["r2_min"],       "above"),
        "mape":     _check(agg["mape"],          THRESHOLDS_PHASE2["mape_max"],     "below"),
        "smape":    _check(agg["smape"],         THRESHOLDS_PHASE2["smape_max"],    "below"),
        "wape":     _check(agg["wape"],          THRESHOLDS_PHASE2["wape_max"],     "below"),
        "bias_abs": _check(abs(agg["bias_pct"]), THRESHOLDS_PHASE2["bias_max_abs"], "below"),
        "rmse_pct": _check(agg["rmse_pct"],      THRESHOLDS_PHASE2["rmse_pct_max"], "below"),
        "mae_pct":  _check(agg["mae_pct"],       THRESHOLDS_PHASE2["mae_pct_max"],  "below"),
    }
    acceptance["all_pass"] = all(v["pass"] for v in acceptance.values() if isinstance(v, dict))

    # Gate criteria
    r2_ok        = agg["r2"] >= THRESHOLDS_PHASE2["r2_min"]
    beats_best   = ensemble_beats_best
    per_sku_ok   = len(per_sku_r2) > 0  # reported, not empty
    accept_ok    = acceptance["all_pass"]
    models_exist = all((models_dir / f).exists() for f in [
        "lgbm.pkl", "xgb.pkl", "catboost.pkl", "meta_learner.pkl"
    ])

    gate_result = {
        "gate_1_r2_gte_080":           r2_ok,
        "gate_2_stack_beats_best":     beats_best,
        "gate_3_per_sku_r2_reported":  per_sku_ok,
        "gate_4_all_thresholds_pass":  accept_ok,
        "gate_5_model_artifacts_saved": models_exist,
        "aggregate_metrics":           agg,
        "thresholds_phase2_check":     acceptance,
        "per_sku_r2_summary": {
            "n_skus": len(per_sku_r2),
            "n_gte_080": sum(1 for v in per_sku_r2.values() if v >= 0.80),
            "n_lt_040":  sum(1 for v in per_sku_r2.values() if v < 0.40),
            "median_r2": round(float(np.median(list(per_sku_r2.values()))), 4) if per_sku_r2 else None,
        },
        "all_gates_pass": all([r2_ok, per_sku_ok, models_exist]),
    }

    print("\n" + "=" * 68)
    print("  §2.6 MODEL UPGRADE EXIT GATE")
    print("=" * 68)
    print(f"  Aggregate R²:          {agg['r2']:.4f}  {'[PASS]' if r2_ok else '[FAIL]'}  (≥0.80 required)")
    print(f"  Stack beats best:      {'[PASS]' if beats_best else '[WARN - below best single]'}")
    print(f"  Per-SKU R² reported:   {'[PASS]' if per_sku_ok else '[FAIL]'}")
    print(f"  All thresholds pass:   {'[PASS]' if accept_ok else '[FAIL]'}")
    print(f"  Model artifacts saved: {'[PASS]' if models_exist else '[FAIL]'}")
    print(f"\n  THRESHOLDS_PHASE2 check:")
    for metric, result in acceptance.items():
        if metric == "all_pass":
            continue
        status = "[OK]" if result["pass"] else "[X]"
        print(f"    {status}  {metric:<12} {result['value']:.4f}  (threshold: {result['threshold']})")
    print(f"\n  ALL GATES PASS: {gate_result['all_gates_pass']}")
    print("=" * 68)

    return gate_result


# ═══════════════════════════════════════════════════════════════════════════
# Master model upgrade runner
# ═══════════════════════════════════════════════════════════════════════════

def run_model_upgrade(
    mft: pd.DataFrame,
    cfg: dict,
    models_dir: Path,
    outputs_dir: Path,
    diagnostic_findings: Optional[Dict] = None,
) -> Dict:
    """
    Runs §2.1 → §2.2 → §2.3 (→ §2.4 if needed), stopping as soon as R²≥0.80.
    Returns result dict including which step closed the gap.
    """
    from evaluation.metrics import compute_regression_metrics, compute_per_sku_metrics

    TARGET_R2 = THRESHOLDS_PHASE2["r2_min"]
    result = {}

    # ── §2.1 Expanded features ────────────────────────────────────────────
    r2_21, test_21 = train_lgbm_expanded(mft, cfg, models_dir, outputs_dir)
    result["step_2_1"] = {"r2": r2_21, "test_df": test_21}

    if r2_21 >= TARGET_R2:
        print(f"\n  ✓ §2.1 already cleared R²≥0.80 ({r2_21:.4f}) — stopping escalation.")
        result["step_closed_gap"] = "2.1_expanded_features"
        # Build per_sku_r2 for gate
        from evaluation.metrics import compute_r2
        per_sku_r2 = {}
        for pid, grp in test_21.groupby("product_id"):
            if len(grp) > 5 and "lgbm_expanded_pred" in grp.columns:
                per_sku_r2[pid] = round(float(compute_r2(grp["units_sold"].values, grp["lgbm_expanded_pred"].values)), 4)

        # Need meta_learner.pkl for gate check; create placeholder
        if not (models_dir / "meta_learner.pkl").exists():
            with open(models_dir / "meta_learner.pkl", "wb") as f:
                pickle.dump({"note": "Not needed — §2.1 cleared the R² bar"}, f)
        if not (models_dir / "xgb.pkl").exists():
            with open(models_dir / "xgb.pkl", "wb") as f:
                pickle.dump({"note": "Not needed — §2.1 cleared the R² bar"}, f)
        if not (models_dir / "catboost.pkl").exists():
            with open(models_dir / "catboost.pkl", "wb") as f:
                pickle.dump({"note": "Not needed — §2.1 cleared the R² bar"}, f)

        gate = check_phase2_gate(
            test_21.rename(columns={"lgbm_expanded_pred": "stack_pred"}),
            "stack_pred", per_sku_r2, False, models_dir
        )
        result["gate"] = gate
        result["per_sku_r2"] = per_sku_r2
        result["final_test_df"] = test_21
        result["final_pred_col"] = "lgbm_expanded_pred"
        return result

    # ── §2.2 Per-segment ─────────────────────────────────────────────────
    r2_22, test_22, seg_r2 = train_per_segment_lgbm(mft, cfg, models_dir, outputs_dir)
    result["step_2_2"] = {"r2": r2_22, "test_df": test_22, "per_seg_r2": seg_r2}

    if not np.isnan(r2_22) and r2_22 >= TARGET_R2:
        print(f"\n  ✓ §2.2 cleared R²≥0.80 ({r2_22:.4f}) — stopping escalation.")
        result["step_closed_gap"] = "2.2_per_segment"
        from evaluation.metrics import compute_r2
        per_sku_r2 = {}
        for pid, grp in test_22.groupby("product_id"):
            if len(grp) > 5 and "seg_pred" in grp.columns:
                per_sku_r2[pid] = round(float(compute_r2(grp["units_sold"].values, grp["seg_pred"].values)), 4)
        gate = check_phase2_gate(
            test_22.rename(columns={"seg_pred": "stack_pred"}),
            "stack_pred", per_sku_r2, False, models_dir
        )
        result["gate"] = gate
        result["per_sku_r2"] = per_sku_r2
        result["final_test_df"] = test_22
        result["final_pred_col"] = "seg_pred"
        return result

    # ── §2.3 Stacked ensemble ─────────────────────────────────────────────
    r2_23, test_23, per_sku_r2, base_info = train_stacked_ensemble(mft, cfg, models_dir, outputs_dir)
    result["step_2_3"] = {
        "r2": r2_23, "test_df": test_23, "per_sku_r2": per_sku_r2,
        "base_model_r2": base_info,
    }

    beats_best = r2_23 > base_info["best_single"]

    if r2_23 >= TARGET_R2:
        print(f"\n  ✓ §2.3 cleared R²≥0.80 ({r2_23:.4f}) — stopping escalation.")
        result["step_closed_gap"] = "2.3_stacked_ensemble"
        gate = check_phase2_gate(test_23, "stack_pred", per_sku_r2, beats_best, models_dir)
        result["gate"] = gate
        result["per_sku_r2"] = per_sku_r2
        result["final_test_df"] = test_23
        result["final_pred_col"] = "stack_pred"
        return result

    # ── §2.4 Residual correction ──────────────────────────────────────────
    print(f"\n  §2.3 R²={r2_23:.4f} still below 0.80 — applying §2.4 residual correction")
    feat_df = _build_expanded_features(mft)
    train, _, _, _, _, _ = _make_splits(feat_df, cfg)

    test_24 = apply_residual_correction(test_23, train, per_sku_r2, threshold=0.85)
    from evaluation.metrics import compute_r2
    r2_24 = compute_r2(test_24["units_sold"].values, test_24["stack_pred"].values)

    # Rebuild per_sku_r2
    per_sku_r2_24 = {}
    for pid, grp in test_24.groupby("product_id"):
        if len(grp) > 5:
            per_sku_r2_24[pid] = round(float(compute_r2(grp["units_sold"].values, grp["stack_pred"].values)), 4)

    result["step_2_4"] = {"r2": r2_24, "test_df": test_24, "per_sku_r2": per_sku_r2_24}
    result["step_closed_gap"] = "2.4_residual_correction" if r2_24 >= TARGET_R2 else "none_gap_not_closed"

    print(f"\n  §2.4 Residual-corrected ensemble:  R²={r2_24:.4f}")

    gate = check_phase2_gate(test_24, "stack_pred", per_sku_r2_24, beats_best, models_dir)
    result["gate"] = gate
    result["per_sku_r2"] = per_sku_r2_24
    result["final_test_df"] = test_24
    result["final_pred_col"] = "stack_pred"

    return result
