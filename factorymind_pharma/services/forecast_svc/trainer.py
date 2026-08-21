"""
forecast_svc/trainer.py — Step 1.2
Baseline forecast: Naive Seasonal + LightGBM.

Chronological train/val/test split (never shuffle — time series).
Features: day-of-week, month, lag-7, lag-28, rolling-mean-7, rolling-mean-28
          of units_sold, plus demand factor columns.
Target:   units_sold.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def _add_lag_roll_features(df: pd.DataFrame, target: str = "units_sold",
                           lags: list = None, windows: list = None) -> pd.DataFrame:
    """
    Add per-product lag and rolling-mean features.
    Must be computed on training data then applied forward — never leak future.
    Data must be sorted by (product_id, date) before calling.
    """
    if lags is None:
        lags = [7, 28]
    if windows is None:
        windows = [7, 28]

    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("product_id")[target].shift(lag)

    for w in windows:
        df[f"rolling_mean_{w}"] = (
            df.groupby("product_id")[target]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    return df


def _encode_categoricals(df: pd.DataFrame, cat_cols: list) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("UNKNOWN"))
            encoders[col] = le
    return df, encoders


# ---------------------------------------------------------------------------
# Naive Seasonal Baseline
# ---------------------------------------------------------------------------

def naive_seasonal_forecast(df: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    """
    Naive seasonal: for each (product_id, test_date), look up the same day-of-week
    from 52 weeks ago (364 days). If unavailable, fall back to 28-day lag, then median.

    Returns a DataFrame with columns: date, product_id, naive_pred.
    """
    df = df.sort_values(["product_id", "date"]).copy()
    df_hist = df[df["date"] < test_start].copy()
    df_test = df[df["date"] >= test_start].copy()

    records = []
    for pid, grp in df_test.groupby("product_id"):
        # Aggregate history by date (sum units_sold across region/channel)
        hist = (
            df_hist[df_hist["product_id"] == pid]
            .groupby("date")["units_sold"]
            .sum()
        )
        # Get unique test dates for this product
        test_dates = grp["date"].unique()
        for target_date in test_dates:
            # Try 364 days back (same day-of-week, ~1 year)
            lookup_date = target_date - pd.Timedelta(days=364)
            if lookup_date in hist.index:
                pred = float(hist[lookup_date])
            else:
                # Fallback: 28-day lag
                lookup_date = target_date - pd.Timedelta(days=28)
                if lookup_date in hist.index:
                    pred = float(hist[lookup_date])
                else:
                    pred = float(hist.median()) if len(hist) > 0 else 0.0
            records.append({"date": target_date, "product_id": pid, "naive_pred": max(pred, 0)})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# LightGBM Model
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "day_of_week", "month", "day_of_month", "week_of_year",
    "lag_7", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "market_demand_index", "disease_season_index",
    "promotion_flag", "competitor_price",
]

CAT_COLS = ["region", "channel", "category", "dosage_form", "criticality"]


def _build_feature_df(df: pd.DataFrame, feature_cols: list, cat_cols: list) -> Tuple[pd.DataFrame, list]:
    """Adds time + lag + roll features and encodes categoricals."""
    df = _add_time_features(df)
    df = _add_lag_roll_features(df)
    df, _ = _encode_categoricals(df, cat_cols)

    # Combine base feature cols + any encoded categorical cols that exist
    all_feat = [c for c in feature_cols if c in df.columns]
    all_feat += [c for c in cat_cols if c in df.columns]

    # Drop rows where lag features are NaN (first few rows per product)
    lag_cols = [c for c in all_feat if c.startswith("lag_")]
    if lag_cols:
        df = df.dropna(subset=lag_cols)

    return df, all_feat


def run_forecast_pipeline(mft: pd.DataFrame, config: dict, outputs_dir: str) -> Dict:
    """
    Full forecast pipeline:
      1. Chronological split.
      2. Naive seasonal baseline.
      3. LightGBM training + inference.
      4. Save predictions.

    Returns dict with train/val/test DataFrames and both model predictions.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    mft = mft.copy()
    mft["date"] = pd.to_datetime(mft["date"])
    mft = mft.sort_values("date")

    # -----------------------------------------------------------------------
    # Chronological split
    # -----------------------------------------------------------------------
    dates = mft["date"].sort_values().unique()
    n = len(dates)
    train_end_idx = int(n * config["split"]["train_ratio"])
    val_end_idx   = int(n * (config["split"]["train_ratio"] + config["split"]["val_ratio"]))

    train_end = dates[train_end_idx - 1]
    val_end   = dates[val_end_idx - 1]
    test_start = dates[val_end_idx]

    train_df = mft[mft["date"] <= train_end].copy()
    val_df   = mft[(mft["date"] > train_end) & (mft["date"] <= val_end)].copy()
    test_df  = mft[mft["date"] > val_end].copy()

    print(f"\n  Chronological Split:")
    print(f"  Train: {train_df['date'].min().date()} → {train_df['date'].max().date()}  ({len(train_df):,} rows)")
    print(f"  Val:   {val_df['date'].min().date()} → {val_df['date'].max().date()}  ({len(val_df):,} rows)")
    print(f"  Test:  {test_df['date'].min().date()} → {test_df['date'].max().date()}  ({len(test_df):,} rows)")

    # -----------------------------------------------------------------------
    # Naive Seasonal Forecast (on test set)
    # -----------------------------------------------------------------------
    print("\n  Running naive seasonal forecast...")
    naive_preds = naive_seasonal_forecast(mft, test_start)

    # Merge naive preds back onto test_df
    test_df = test_df.merge(
        naive_preds[["date", "product_id", "naive_pred"]],
        on=["date", "product_id"], how="left"
    )

    # -----------------------------------------------------------------------
    # LightGBM
    # -----------------------------------------------------------------------
    print("  Building LightGBM features...")

    # Build features on the full dataset (sorted by product+date for lag integrity)
    full_feat_df = _add_time_features(mft)
    full_feat_df = _add_lag_roll_features(full_feat_df, lags=[7, 28], windows=[7, 28])
    full_feat_df, _ = _encode_categoricals(full_feat_df, CAT_COLS)

    all_feat = [c for c in FEATURE_COLS if c in full_feat_df.columns]
    all_feat += [c for c in CAT_COLS if c in full_feat_df.columns]

    # Drop rows where any lag is NaN (initial rows per product)
    lag_cols = [c for c in all_feat if c.startswith("lag_")]
    full_feat_df = full_feat_df.dropna(subset=lag_cols)

    X_train = full_feat_df[full_feat_df["date"] <= train_end][all_feat]
    y_train = full_feat_df[full_feat_df["date"] <= train_end]["units_sold"]
    X_val   = full_feat_df[(full_feat_df["date"] > train_end) & (full_feat_df["date"] <= val_end)][all_feat]
    y_val   = full_feat_df[(full_feat_df["date"] > train_end) & (full_feat_df["date"] <= val_end)]["units_sold"]
    X_test  = full_feat_df[full_feat_df["date"] > val_end][all_feat]

    lgb_params = config["model"]["lightgbm"].copy()
    n_estimators = lgb_params.pop("n_estimators", 500)

    model = lgb.LGBMRegressor(n_estimators=n_estimators, **lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    y_test_pred = np.maximum(model.predict(X_test), 0)  # non-negative demand

    # Attach lgb preds to test slice
    test_feat_df = full_feat_df[full_feat_df["date"] > val_end].copy()
    test_feat_df["lgb_pred"] = y_test_pred

    # -----------------------------------------------------------------------
    # Merge everything onto a combined test output
    # -----------------------------------------------------------------------
    test_out = mft[mft["date"] > val_end][["date", "product_id", "units_sold"]].copy()
    test_out = test_out.merge(
        naive_preds[["date", "product_id", "naive_pred"]],
        on=["date", "product_id"], how="left"
    )
    test_out = test_out.merge(
        test_feat_df[["date", "product_id", "lgb_pred"]],
        on=["date", "product_id"], how="left"
    )

    # Save predictions
    pred_path = outputs_dir / "test_predictions.parquet"
    test_out.to_parquet(pred_path, index=False, engine="pyarrow")
    print(f"\n  ✓  Forecasts saved: {pred_path}")
    print(f"     Test rows: {len(test_out):,}  |  Products: {test_out['product_id'].nunique()}")

    return {
        "model": model,
        "feature_cols": all_feat,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_out,
        "train_end": train_end,
        "val_end": val_end,
        "test_start": test_start,
    }
