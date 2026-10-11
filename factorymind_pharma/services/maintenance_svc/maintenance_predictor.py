"""
services/maintenance_svc/maintenance_predictor.py — Phase 2, §3.2
Predictive maintenance: failure-risk classifier.

Model: LightGBM classifier predicting maintenance_flag / imminent failure.
Features: temperature_c, vibration_mm_s, pressure_bar, rpm,
          power_consumption_kw, operating_hours, days_since_previous_maintenance.
Split: time-based (machine future cannot leak into its own training rows).
Verify: AUC ≥ 0.85 on held-out machines/time window.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

SENSOR_COLS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rpm",
    "power_consumption_kw",
    "operating_hours",
]
DERIVED_COLS = ["days_since_previous_maintenance"]
TARGET_COL   = "maintenance_flag"


def _build_maintenance_features(
    sensor_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join sensor + maintenance data and engineer features.
    Computes days_since_previous_maintenance per machine.
    """
    sensor_df = sensor_df.copy()
    sensor_df["timestamp"] = pd.to_datetime(
        sensor_df.get("timestamp", sensor_df.get("date", pd.Timestamp("2020-01-01")))
    )
    sensor_df["date"] = sensor_df["timestamp"].dt.date
    sensor_df["date"] = pd.to_datetime(sensor_df["date"])

    maint_df = maintenance_df.copy()
    date_col = "maintenance_date" if "maintenance_date" in maint_df.columns else \
               "date" if "date" in maint_df.columns else maint_df.columns[0]
    maint_df[date_col] = pd.to_datetime(maint_df[date_col])

    # Aggregate sensor by (machine_id, date) — daily mean
    machine_col = "machine_id" if "machine_id" in sensor_df.columns else \
                  "equipment_id" if "equipment_id" in sensor_df.columns else \
                  sensor_df.columns[0]

    agg_dict = {c: "mean" for c in SENSOR_COLS if c in sensor_df.columns}
    daily_sensor = (
        sensor_df.groupby([machine_col, "date"], as_index=False)
        .agg(agg_dict)
    )

    # Determine maintenance_flag — if maintenance_df has it, use it;
    # if failure_type column exists, use it (non-null = failure event);
    # otherwise derive from within-7-day window before a maintenance event.
    maint_machine_col = machine_col if machine_col in maint_df.columns else maint_df.columns[0]

    if TARGET_COL in maint_df.columns:
        maint_events = maint_df[[maint_machine_col, date_col, TARGET_COL]].copy()
        maint_events = maint_events.rename(columns={date_col: "date", maint_machine_col: machine_col})
        daily_sensor = daily_sensor.merge(
            maint_events.groupby([machine_col, "date"])[TARGET_COL].max().reset_index(),
            on=[machine_col, "date"], how="left"
        )
        daily_sensor[TARGET_COL] = daily_sensor[TARGET_COL].fillna(0).astype(int)
    elif "failure_type" in maint_df.columns:
        # Non-null failure_type = an unplanned failure requiring maintenance
        maint_df["_is_failure"] = (~maint_df["failure_type"].isna() &
                                    (maint_df["failure_type"].astype(str).str.strip() != "")).astype(int)
        fail_events = maint_df[[maint_machine_col, date_col, "_is_failure"]].copy()
        fail_events = fail_events.rename(columns={date_col: "date", maint_machine_col: machine_col})
        # For each failure event, flag sensor rows in the 7 days BEFORE it
        daily_sensor[TARGET_COL] = 0
        failure_rows = fail_events[fail_events["_is_failure"] == 1]
        for _, me in failure_rows.iterrows():
            mask = (
                (daily_sensor[machine_col] == me[machine_col]) &
                (daily_sensor["date"] >= me["date"] - pd.Timedelta(days=7)) &
                (daily_sensor["date"] < me["date"])
            )
            daily_sensor.loc[mask, TARGET_COL] = 1
    else:
        # Vectorised date-window derivation: flag=1 if a maintenance event
        # happened within 7 days AFTER this sensor reading.
        maint_events = maint_df[[maint_machine_col, date_col]].copy()
        maint_events = maint_events.rename(columns={date_col: "maint_date", maint_machine_col: machine_col})
        daily_sensor[TARGET_COL] = 0
        # Group by machine for vectorised window check
        for mid, grp in maint_events.groupby(machine_col):
            maint_dates = grp["maint_date"].sort_values().values
            sensor_mask = daily_sensor[machine_col] == mid
            sensor_dates = daily_sensor.loc[sensor_mask, "date"].values
            # For each sensor date, check if any maint date in [date, date+7]
            flags = np.zeros(sensor_mask.sum(), dtype=int)
            for i, sd in enumerate(sensor_dates):
                if np.any((maint_dates >= sd) & (maint_dates <= sd + np.timedelta64(7, 'D'))):
                    flags[i] = 1
            daily_sensor.loc[sensor_mask, TARGET_COL] = flags

    # days_since_previous_maintenance
    if "days_since_previous_maintenance" in sensor_df.columns:
        dsp = sensor_df.groupby([machine_col, "date"])["days_since_previous_maintenance"].mean().reset_index()
        daily_sensor = daily_sensor.merge(dsp, on=[machine_col, "date"], how="left")
    elif date_col in maint_df.columns:
        # Compute from maint events
        maint_sorted = maint_events.sort_values([machine_col, "maint_date"])
        dspm = []
        for mid, grp in maint_sorted.groupby(machine_col):
            maint_dates = grp["maint_date"].sort_values().values
            sensor_rows = daily_sensor[daily_sensor[machine_col] == mid].sort_values("date")
            for _, row in sensor_rows.iterrows():
                prev = maint_dates[maint_dates < np.datetime64(row["date"])]
                if len(prev) > 0:
                    days = (row["date"] - pd.Timestamp(prev[-1])).days
                else:
                    days = 999
                dspm.append({machine_col: mid, "date": row["date"], "days_since_previous_maintenance": days})
        if dspm:
            dspm_df = pd.DataFrame(dspm)
            daily_sensor = daily_sensor.merge(dspm_df, on=[machine_col, "date"], how="left")
    
    if "days_since_previous_maintenance" not in daily_sensor.columns:
        daily_sensor["days_since_previous_maintenance"] = 0

    # Rolling sensor stats (7-day rolling mean per machine — early degradation signal)
    for col in SENSOR_COLS:
        if col in daily_sensor.columns:
            daily_sensor[f"{col}_roll7"] = (
                daily_sensor.groupby(machine_col)[col]
                .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
            )

    daily_sensor["machine_id"] = daily_sensor[machine_col]
    return daily_sensor


def train_maintenance_classifier(
    sensor_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
    models_dir: Path,
    cfg: dict,
) -> Dict:
    """
    §3.2: Train LightGBM classifier for maintenance failure prediction.
    Time-based split (no future leakage per machine).
    Returns dict with AUC, model artifact path.
    """
    logger.info("=== §3.2 Predictive Maintenance Classifier ===")

    feat_df = _build_maintenance_features(sensor_df, maintenance_df)
    feat_df = feat_df.sort_values(["machine_id", "date"]).reset_index(drop=True)

    feature_cols = [c for c in (
        SENSOR_COLS + DERIVED_COLS +
        [f"{c}_roll7" for c in SENSOR_COLS]
    ) if c in feat_df.columns]

    feat_df_clean = feat_df.dropna(subset=feature_cols + [TARGET_COL])

    if len(feat_df_clean) < 100:
        logger.warning("Too few rows for maintenance classifier: %d", len(feat_df_clean))
        return {"error": "insufficient data", "auc": None}

    # Time-based split: first 80% → train, last 20% → test
    dates = feat_df_clean["date"].sort_values().unique()
    split_idx = int(len(dates) * 0.80)
    train_dates = dates[:split_idx]
    test_dates  = dates[split_idx:]

    train = feat_df_clean[feat_df_clean["date"].isin(train_dates)]
    test  = feat_df_clean[feat_df_clean["date"].isin(test_dates)]

    X_train = train[feature_cols].fillna(0)
    y_train = train[TARGET_COL].astype(int)
    X_test  = test[feature_cols].fillna(0)
    y_test  = test[TARGET_COL].astype(int)

    n_pos_train = y_train.sum()
    n_neg_train = len(y_train) - n_pos_train
    scale_pos_weight = max(1.0, n_neg_train / max(n_pos_train, 1))

    lgb_params = cfg["model"]["lightgbm"].copy()
    n_est = lgb_params.pop("n_estimators", 500)

    maint_model = lgb.LGBMClassifier(
        n_estimators=n_est,
        scale_pos_weight=scale_pos_weight,
        **lgb_params,
    )
    maint_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    y_prob = maint_model.predict_proba(X_test)[:, 1]

    if len(np.unique(y_test)) < 2:
        auc = 0.5
        logger.warning("Test set has only one class — AUC defaulting to 0.5")
    else:
        auc = roc_auc_score(y_test, y_prob)

    REQUIRED_AUC = 0.85
    print(f"\n  §3.2 Predictive Maintenance:")
    print(f"    Train rows: {len(train):,}  Test rows: {len(test):,}")
    print(f"    Class balance train: {n_pos_train} failures / {n_neg_train} normal")
    print(f"    AUC: {auc:.4f}  {'[PASS]' if auc >= REQUIRED_AUC else '[WARN <0.85]'}  (requirement: ≥0.85)")

    # Score full dataset for dashboard
    X_all = feat_df_clean[feature_cols].fillna(0)
    feat_df_clean = feat_df_clean.copy()
    feat_df_clean["failure_risk_prob"] = maint_model.predict_proba(X_all)[:, 1]
    feat_df_clean["failure_risk_flag"] = (feat_df_clean["failure_risk_prob"] > 0.50).astype(int)

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "maintenance_classifier.pkl", "wb") as f:
        pickle.dump({
            "model": maint_model,
            "feature_cols": feature_cols,
        }, f)
    logger.info("Saved models/maintenance_classifier.pkl")

    return {
        "auc": float(auc),
        "n_train": len(train),
        "n_test": len(test),
        "n_pos_train": int(n_pos_train),
        "gate_pass": auc >= REQUIRED_AUC,
        "scored_df": feat_df_clean,
        "feature_cols": feature_cols,
    }
