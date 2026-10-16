"""
tests/test_no_leakage.py
========================
Verifies that _build_expanded_features() honours the 1-day-ahead forecast
contract:  features available on date t may only use information from dates ≤ t-1.

Two independent checks:
  1. Truncation-invariance  — dropping the last row of a product's history must
     NOT change any feature value for all earlier rows.  If a same-day signal
     is present the transform will be recomputed over a different window and
     at least one earlier-row value will change.
  2. Banned-column list     — raw (un-lagged) columns that carry same-day
     demand information must never appear as features used for training.

Each check includes a mandatory negative test that proves the assertion is
capable of failing (i.e. it is not rigged to always pass).
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from services.forecast_svc.ensemble_trainer import (
    _build_expanded_features,
    EXPANDED_FEATURE_COLS,
)

# ── helpers ───────────────────────────────────────────────────────────────────

BANNED_RAW_COLS = {"total_ordered_quantity", "num_orders", "total_received_quantity"}

_CATEGORIES = ["Analgesics", "Antibiotics", "Vitamins"]


def _make_synthetic_df(n_products: int = 3, n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Synthetic pharma-like daily transaction data."""
    rng = np.random.default_rng(seed)
    records = []
    base = pd.Timestamp("2024-01-01")
    products = [f"P{i:03d}" for i in range(n_products)]
    cats = {p: _CATEGORIES[i % len(_CATEGORIES)] for i, p in enumerate(products)}

    for pid in products:
        for d in range(n_days):
            date = base + pd.Timedelta(days=d)
            units = max(0, int(rng.poisson(50) + rng.normal(0, 5)))
            records.append({
                "date": date,
                "product_id": pid,
                "units_sold": units,
                "category": cats[pid],
                "criticality": rng.choice(["High", "Medium", "Low"]),
                "market_demand_index": rng.uniform(0.8, 1.2),
                "disease_season_index": rng.uniform(0.5, 1.5),
                "promotion_flag": int(rng.random() < 0.1),
                "competitor_price": rng.uniform(10, 100),
                "regional_demand_index": rng.uniform(0.9, 1.1),
                "selling_price": rng.uniform(5, 50),
                "discount_pct": rng.uniform(0, 0.3),
                "temperature_c": rng.uniform(15, 35),
                "humidity_pct": rng.uniform(30, 80),
                "total_ordered_quantity": int(rng.poisson(45)),
                "num_orders": int(rng.poisson(5)),
                "total_received_quantity": int(rng.poisson(40)),
                "total_closing_stock": int(rng.poisson(200)),
                "stockout_flag": int(rng.random() < 0.05),
            })
    return pd.DataFrame(records)


# =============================================================================
# Check 1: Truncation-invariance
# =============================================================================

class TestTruncationInvariance:
    """
    Dropping the last row for one product must not alter feature values for any
    earlier row of that product.  Leaky same-day features fail this test because
    they implicitly aggregate over the full date-group (including the last row).
    """

    @staticmethod
    def _feature_snapshot(df: pd.DataFrame, pid: str, drop_last: bool) -> pd.DataFrame:
        """Return feature-engineered rows for a single product."""
        d = df.copy()
        if drop_last:
            last_date = d[d["product_id"] == pid]["date"].max()
            d = d[~((d["product_id"] == pid) & (d["date"] == last_date))]
        feat = _build_expanded_features(d)
        return feat[feat["product_id"] == pid].sort_values("date").reset_index(drop=True)

    def test_lag_features_are_invariant_to_truncation(self):
        """Lag features must be identical whether or not the last row is present."""
        df = _make_synthetic_df()
        pid = "P000"

        full = self._feature_snapshot(df, pid, drop_last=False)
        trunc = self._feature_snapshot(df, pid, drop_last=True)

        # Compare only rows that exist in both (trunc has one fewer row)
        n = len(trunc)
        lag_cols = [c for c in full.columns if c.startswith("lag_")]
        assert lag_cols, "No lag columns found — feature builder may be broken"

        for col in lag_cols:
            full_vals = full[col].iloc[:n].values
            trunc_vals = trunc[col].values
            np.testing.assert_array_equal(
                full_vals, trunc_vals,
                err_msg=f"Lag column '{col}' is NOT truncation-invariant — possible leakage"
            )

    def test_category_demand_is_invariant_to_truncation(self):
        """category_demand must only reflect previous-day totals."""
        df = _make_synthetic_df()
        pid = "P000"

        full = self._feature_snapshot(df, pid, drop_last=False)
        trunc = self._feature_snapshot(df, pid, drop_last=True)

        if "category_demand" not in full.columns:
            pytest.skip("category_demand not present — skipping")

        n = len(trunc)
        full_vals = full["category_demand"].iloc[:n].values
        trunc_vals = trunc["category_demand"].values
        np.testing.assert_array_almost_equal(
            full_vals, trunc_vals, decimal=6,
            err_msg="category_demand is NOT truncation-invariant — same-day leakage detected"
        )

    # ── Negative test: prove the assertion CAN fail ──────────────────────────

    def test_negative__same_day_agg_fails_truncation_check(self):
        """
        Demonstrate that a same-day cross-product aggregation FAILS truncation invariance.
        When P000's final date is truncated, its category peer P003 (in the same category)
        has its same-day category demand on that date corrupted (reduced), whereas the
        properly lagged category_demand is 100% invariant.
        """
        df = _make_synthetic_df(n_products=6)
        # P000 and P003 both belong to "Analgesics"
        assert df[df["product_id"] == "P000"]["category"].iloc[0] == df[df["product_id"] == "P003"]["category"].iloc[0]

        d_full = df.copy()
        last_date = d_full[d_full["product_id"] == "P000"]["date"].max()
        # Truncate only P000's record on last_date
        d_trunc = d_full[~((d_full["product_id"] == "P000") & (d_full["date"] == last_date))].copy()

        # 1. Properly lagged feature (production code): invariant for peer on last_date
        feat_full = _build_expanded_features(d_full)
        feat_trunc = _build_expanded_features(d_trunc)
        val_clean_full = feat_full[(feat_full["product_id"] == "P003") & (feat_full["date"] == last_date)]["category_demand"].values[0]
        val_clean_trunc = feat_trunc[(feat_trunc["product_id"] == "P003") & (feat_trunc["date"] == last_date)]["category_demand"].values[0]
        np.testing.assert_equal(
            val_clean_full, val_clean_trunc,
            err_msg="Clean category_demand must be invariant to peer truncation"
        )

        # 2. Leaky same-day aggregation: FAILS invariance because P000's same-day units are lost
        leaky_full = d_full.groupby(["date", "category"])["units_sold"].transform("sum")
        leaky_trunc = d_trunc.groupby(["date", "category"])["units_sold"].transform("sum")
        d_full_leaky = d_full.assign(leaky_cat=leaky_full)
        d_trunc_leaky = d_trunc.assign(leaky_cat=leaky_trunc)

        val_leaky_full = d_full_leaky[(d_full_leaky["product_id"] == "P003") & (d_full_leaky["date"] == last_date)]["leaky_cat"].values[0]
        val_leaky_trunc = d_trunc_leaky[(d_trunc_leaky["product_id"] == "P003") & (d_trunc_leaky["date"] == last_date)]["leaky_cat"].values[0]

        assert val_leaky_full != val_leaky_trunc, (
            "Negative test failed: leaky same-day transform did not detect truncation"
        )


# =============================================================================
# Check 2: Banned-column enforcement
# =============================================================================

class TestBannedColumns:
    """
    The training feature list (EXPANDED_FEATURE_COLS) must not contain any
    raw same-day demand columns.  Only their lag-1 variants are permitted.
    """

    def test_banned_raw_cols_absent_from_feature_list(self):
        """No banned raw columns appear in EXPANDED_FEATURE_COLS."""
        present = BANNED_RAW_COLS & set(EXPANDED_FEATURE_COLS)
        assert not present, (
            f"Same-day banned columns still in EXPANDED_FEATURE_COLS: {present}\n"
            "These columns cause data leakage in 1-day-ahead forecasting."
        )

    def test_lag1_variants_are_present_in_feature_list(self):
        """The lag-1 counterparts of all banned columns must be in the feature list."""
        for raw_col in BANNED_RAW_COLS:
            lag_col = f"{raw_col}_lag1"
            assert lag_col in EXPANDED_FEATURE_COLS, (
                f"Lag-1 replacement '{lag_col}' missing from EXPANDED_FEATURE_COLS. "
                "Add it to preserve the signal without leakage."
            )

    def test_features_built_do_not_contain_banned_raw_cols(self):
        """Running _build_expanded_features must not produce raw banned cols as features."""
        df = _make_synthetic_df()
        feat = _build_expanded_features(df)
        feat_cols_used = set(EXPANDED_FEATURE_COLS) & set(feat.columns)
        leaky = feat_cols_used & BANNED_RAW_COLS
        assert not leaky, (
            f"After feature building, banned raw columns are still used as features: {leaky}"
        )

    # ── Negative test ─────────────────────────────────────────────────────────

    def test_negative__banned_col_detection_works(self):
        """
        Temporarily inject a banned column into a synthetic feature list and
        confirm our detection logic catches it.
        """
        synthetic_bad_feature_list = list(EXPANDED_FEATURE_COLS) + ["total_ordered_quantity"]
        present = BANNED_RAW_COLS & set(synthetic_bad_feature_list)
        assert present, (
            "Negative test failed: injected 'total_ordered_quantity' but detection "
            "did not find it — the banned-column check is broken"
        )


# =============================================================================
# Check 3: Existence Canary (Sale-existence leakage rejection)
# =============================================================================

class TestExistenceCanary:
    """
    Features evaluated on the product x calendar-day grid must not leak whether
    a sale occurred on that date via missingness patterns or row counts.
    """

    @staticmethod
    def _check_canary(feat_df: pd.DataFrame, target_series: pd.Series):
        """
        Canary assertion:
        1. No feature contains 'row_count', 'is_sale', or transaction identifiers.
        2. No feature's missingness correlates with target presence (units_sold > 0).
        """
        sale_occurred = (target_series > 0).astype(int)
        banned_terms = {"row_count", "num_rows", "is_sale", "sale_exists", "transaction_count"}

        for col in feat_df.columns:
            # Check 1: Banned terminology
            for term in banned_terms:
                assert term not in col.lower(), f"Feature '{col}' contains sale-existence indicator '{term}'"

            # Check 2: Missingness mask leakage
            isna_mask = feat_df[col].isna().astype(int)
            if isna_mask.nunique() > 1:
                corr = np.corrcoef(isna_mask, sale_occurred)[0, 1]
                assert abs(corr) < 0.10, (
                    f"Feature '{col}' missingness correlates with sale existence (corr={corr:.3f}). "
                    "Features must not use missingness to signal zero-demand days."
                )

    def test_calendar_grid_features_pass_existence_canary(self):
        """Clean calendar grid features must have zero missingness correlation with sale existence."""
        df = _make_synthetic_df(n_products=6, n_days=60)
        feat = _build_expanded_features(df)
        # Drop initial lag warm-up
        clean = feat.dropna(subset=[c for c in feat.columns if c.startswith("lag_")])
        self._check_canary(clean, clean["units_sold"])

    def test_negative__leaky_missingness_fails_canary(self):
        """
        Negative test: A feature where NaNs align with non-sale days (e.g. unshifted demand factors)
        MUST be detected and rejected by the canary.
        """
        df = _make_synthetic_df(n_products=6, n_days=60)
        leaky_df = df.copy()
        # Create zero-demand days (intermittent demand pattern)
        leaky_df.loc[::3, "units_sold"] = 0
        # Deliberately construct leaky feature: present on sale days, NaN on zero-sale days
        leaky_df["leaky_factor"] = np.where(leaky_df["units_sold"] > 0, 100.0, np.nan)

        with pytest.raises(AssertionError, match="missingness correlates with sale existence"):
            self._check_canary(leaky_df[["leaky_factor"]], leaky_df["units_sold"])

    def test_negative__row_count_feature_fails_canary(self):
        """Negative test: Explicit row count feature MUST be rejected."""
        df = _make_synthetic_df(n_products=6, n_days=60)
        leaky_df = df.copy()
        leaky_df["daily_row_count"] = 1.0

        with pytest.raises(AssertionError, match="contains sale-existence indicator"):
            self._check_canary(leaky_df[["daily_row_count"]], leaky_df["units_sold"])


# =============================================================================
# Check 4: No Backfill (bfill) in Feature Engineering
# =============================================================================

class TestNoBfill:
    """
    Backfill (.bfill()) propagates future information into the past and is strictly
    forbidden in time series feature construction. Only forward fill (.ffill()) is allowed.
    """

    @staticmethod
    def _scan_for_bfill(filepath: str) -> list[str]:
        """Return lines containing .bfill( in code (excluding comments)."""
        import re
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        violations = []
        for idx, line in enumerate(lines, 1):
            clean_line = line.split("#")[0].strip()
            if re.search(r"\.bfill\(", clean_line) or re.search(r"\bbackward_fill\b", clean_line):
                violations.append(f"Line {idx}: {line.strip()}")
        return violations

    def test_no_bfill_in_ensemble_trainer(self):
        """ensemble_trainer.py must not contain any .bfill() calls in code."""
        trainer_path = os.path.join(ROOT, "services", "forecast_svc", "ensemble_trainer.py")
        violations = self._scan_for_bfill(trainer_path)
        assert not violations, f"Found .bfill() in {trainer_path}:\n" + "\n".join(violations)

    def test_no_bfill_in_eval_full_grid(self):
        """eval_full_grid.py must not contain any .bfill() calls in code."""
        eval_path = os.path.join(ROOT, "eval_full_grid.py")
        violations = self._scan_for_bfill(eval_path)
        assert not violations, f"Found .bfill() in {eval_path}:\n" + "\n".join(violations)

    def test_negative__bfill_detector_catches_violation(self, tmp_path):
        """
        Negative test: Synthetically injecting a .bfill() call must be caught and fail.
        """
        bad_file = tmp_path / "leaky_feature_builder.py"
        bad_file.write_text("df['feature'] = df.groupby('pid')['val'].bfill()\n", encoding="utf-8")
        violations = self._scan_for_bfill(str(bad_file))
        assert len(violations) == 1, (
            "Negative test failed: .bfill() was injected but scanner failed to detect it."
        )


