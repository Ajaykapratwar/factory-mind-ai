"""
services/qc_svc/qc_anomaly.py — Phase 2, §3.1
Quality Control anomaly detection using Isolation Forest.

Model: Isolation Forest (unsupervised) on numeric QC fields.
Validation: scored against release_status (PASS/FAIL) as a label —
            NOT a training label (spec §3.1).
Output: anomaly score per batch.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# QC numeric features to use
QC_NUMERIC_COLS = ["assay_pct", "dissolution_pct", "impurity_level"]
QC_TARGET_COL   = "release_status"
QC_PASS_VALUE   = "PASS"   # Will be confirmed from data


def _confirm_release_status_values(qc_df: pd.DataFrame) -> Tuple[str, str]:
    """Confirm actual PASS/FAIL field values from data."""
    if QC_TARGET_COL not in qc_df.columns:
        raise ValueError(f"Column '{QC_TARGET_COL}' not found in QC data")
    unique_vals = qc_df[QC_TARGET_COL].dropna().unique()
    logger.info("release_status unique values: %s", unique_vals)
    # Try common patterns
    for val in unique_vals:
        if str(val).upper() == "PASS":
            pass_val = val
            break
    else:
        pass_val = unique_vals[0]  # fallback: first value is PASS
    fail_vals = [v for v in unique_vals if v != pass_val]
    logger.info("Using PASS='%s', FAIL=%s", pass_val, fail_vals)
    return str(pass_val), [str(v) for v in fail_vals]


def train_qc_model(
    qc_df: pd.DataFrame,
    models_dir: Path,
) -> Tuple["IsolationForest", "StandardScaler", Dict]:
    """
    §3.1: Train Isolation Forest on QC numeric features.
    Validate against release_status on held-out set.
    """
    from evaluation.metrics import compute_classification_metrics

    logger.info("=== §3.1 QC Anomaly Detection ===")
    qc_df = qc_df.copy()

    # Confirm field values from data
    pass_val, fail_vals = _confirm_release_status_values(qc_df)

    # Drop rows with all QC features missing
    avail_cols = [c for c in QC_NUMERIC_COLS if c in qc_df.columns]
    if not avail_cols:
        raise ValueError(f"None of the expected QC columns {QC_NUMERIC_COLS} found. "
                         f"Available: {list(qc_df.columns)}")

    qc_clean = qc_df.dropna(subset=avail_cols).copy()
    logger.info("QC data: %d rows after dropping rows with all-null QC features", len(qc_clean))

    # Binary label for evaluation
    qc_clean["is_fail"] = (~qc_clean[QC_TARGET_COL].astype(str).isin([pass_val])).astype(int)
    n_fail = qc_clean["is_fail"].sum()
    n_pass = len(qc_clean) - n_fail
    logger.info("Release status: %d PASS, %d FAIL", n_pass, n_fail)

    # Time-based split: train on first 80%, test on last 20%
    qc_clean = qc_clean.sort_values("batch_date" if "batch_date" in qc_clean.columns else
                                     "test_date" if "test_date" in qc_clean.columns else
                                     qc_clean.columns[0])
    split_idx = int(len(qc_clean) * 0.80)
    qc_train = qc_clean.iloc[:split_idx]
    qc_test  = qc_clean.iloc[split_idx:]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(qc_train[avail_cols].fillna(0))
    X_test  = scaler.transform(qc_test[avail_cols].fillna(0))

    # Isolation Forest (contamination = proportion of FAIL in train if available)
    contamination = max(0.01, min(0.50, n_fail / len(qc_clean)))
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_train)

    # Scores: -1=anomaly, +1=normal (from sklearn convention)
    test_scores = iso_forest.score_samples(X_test)   # more negative = more anomalous
    test_flags  = (iso_forest.predict(X_test) == -1).astype(int)  # 1 = anomaly

    # Evaluation against release_status
    y_true_fail = qc_test["is_fail"].values
    clf_metrics = compute_classification_metrics(y_true_fail, test_flags)

    print(f"\n  §3.1 QC Anomaly Detection:")
    print(f"    Train rows: {len(qc_train):,}  Test rows: {len(qc_test):,}")
    print(f"    PASS: {n_pass}  FAIL: {n_fail}  contamination={contamination:.4f}")
    print(f"    Precision: {clf_metrics['precision']:.4f}")
    print(f"    Recall:    {clf_metrics['recall']:.4f}  (recall against FAIL = key metric)")
    print(f"    F1:        {clf_metrics['f1']:.4f}")

    # Full dataset scores (for operational use)
    X_all = scaler.transform(qc_clean[avail_cols].fillna(0))
    all_scores = iso_forest.score_samples(X_all)
    qc_clean = qc_clean.copy()
    qc_clean["anomaly_score"] = all_scores
    qc_clean["anomaly_flag"]  = (iso_forest.predict(X_all) == -1).astype(int)

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "qc_anomaly_model.pkl", "wb") as f:
        pickle.dump({
            "iso_forest": iso_forest,
            "scaler": scaler,
            "feature_cols": avail_cols,
            "pass_val": pass_val,
            "fail_vals": fail_vals,
        }, f)
    logger.info("Saved models/qc_anomaly_model.pkl")

    # Save scored output
    score_df = qc_clean[["batch_id"] + avail_cols + [QC_TARGET_COL, "is_fail",
                                                       "anomaly_score", "anomaly_flag"]
                        if "batch_id" in qc_clean.columns else
                        avail_cols + [QC_TARGET_COL, "is_fail", "anomaly_score", "anomaly_flag"]]

    return iso_forest, scaler, {
        "precision": clf_metrics["precision"],
        "recall": clf_metrics["recall"],
        "f1": clf_metrics["f1"],
        "n_train": len(qc_train),
        "n_test": len(qc_test),
        "n_fail": n_fail,
        "contamination": contamination,
        "pass_val": pass_val,
        "fail_vals": fail_vals,
        "scored_df": qc_clean,
    }


def score_qc_batch(
    batch_data: pd.DataFrame,
    models_dir: Path,
) -> pd.DataFrame:
    """
    Score a new batch through the trained QC model.
    Returns df with anomaly_score and anomaly_flag columns.
    """
    with open(models_dir / "qc_anomaly_model.pkl", "rb") as f:
        pkg = pickle.load(f)

    iso_forest  = pkg["iso_forest"]
    scaler      = pkg["scaler"]
    feature_cols = pkg["feature_cols"]

    batch_data = batch_data.copy()
    avail = [c for c in feature_cols if c in batch_data.columns]
    X = scaler.transform(batch_data[avail].fillna(0))
    batch_data["anomaly_score"] = iso_forest.score_samples(X)
    batch_data["anomaly_flag"]  = (iso_forest.predict(X) == -1).astype(int)
    return batch_data
