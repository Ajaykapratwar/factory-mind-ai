"""
tests/test_data_facts.py -- Phase 4-6 spec section 1 carry-forward corrections.

Asserts C1, C2, C3, C5, C6, C8 against the raw CSVs. A data change will
fail these tests loudly rather than silently propagating wrong assumptions.

C4 and C7 are not asserted here (topology is config-only; R2 grain is panel
context, not a raw-CSV property).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Locate raw data relative to this file (factorymind_pharma/tests -> ../data)
_HERE = Path(__file__).parent
_DATA = _HERE.parent.parent / "data"


@pytest.fixture(scope="module")
def compliance_df():
    return pd.read_csv(_DATA / "14_compliance.csv", low_memory=False)


@pytest.fixture(scope="module")
def inventory_df():
    return pd.read_csv(_DATA / "05_inventory.csv", low_memory=False)


@pytest.fixture(scope="module")
def financials_df():
    return pd.read_csv(_DATA / "15_financials.csv", low_memory=False)


@pytest.fixture(scope="module")
def qc_df():
    return pd.read_csv(_DATA / "08_quality_control.csv", low_memory=False)


# ---------------------------------------------------------------------------
# C1  compliance audit_status values
# ---------------------------------------------------------------------------

class TestC1AuditStatus:
    """
    14_compliance.audit_status values are Compliant, Minor Deviation,
    Major Deviation. There is NO 'Passed' value.
    Phase 3 gate that checked != 'Passed' would block every batch.
    Fixed by tiered gate in spec section 10.2.
    """

    def test_no_passed_value(self, compliance_df):
        assert "Passed" not in compliance_df["audit_status"].unique(), (
            "C1 violated: 'Passed' found in audit_status -- "
            "Phase 3 gate logic must stay updated to the tiered rule (spec 10.2)."
        )

    def test_expected_values_present(self, compliance_df):
        vals = set(compliance_df["audit_status"].unique())
        assert "Compliant" in vals, "C1: 'Compliant' missing from audit_status"
        assert "Minor Deviation" in vals, "C1: 'Minor Deviation' missing from audit_status"
        assert "Major Deviation" in vals, "C1: 'Major Deviation' missing from audit_status"

    def test_compliant_pct_approx_80(self, compliance_df):
        pct = (compliance_df["audit_status"] == "Compliant").mean() * 100
        assert 70 <= pct <= 90, (
            f"C1: Compliant % changed significantly: {pct:.1f}% (expected approx 80.3%)"
        )

    def test_deviation_count_range(self, compliance_df):
        dmin = int(compliance_df["deviation_count"].min())
        dmax = int(compliance_df["deviation_count"].max())
        assert dmin == 0, f"C1: deviation_count min changed: {dmin}"
        assert dmax <= 10, f"C1: deviation_count max unusually high: {dmax}"

    def test_compliant_rows_can_have_positive_deviation_count(self, compliance_df):
        """Compliant rows sometimes have deviation_count >= 1 -- this is correct data."""
        compliant = compliance_df[compliance_df["audit_status"] == "Compliant"]
        pct_with_deviation = (compliant["deviation_count"] >= 1).mean() * 100
        assert pct_with_deviation > 10, (
            f"C1: Expected many Compliant rows with deviation_count>=1, got {pct_with_deviation:.1f}%"
        )


# ---------------------------------------------------------------------------
# C2  inventory stockout_flag is 0 on all rows
# ---------------------------------------------------------------------------

class TestC2StockoutFlag:
    """
    05_inventory.stockout_flag is 0 on all 35,000 rows.
    Historical stockout count KPI = 0; backtest by ledger replay is impossible.
    Use derived stockout_risk_event = closing_stock <= 0 OR closing_stock < safety_stock.
    """

    def test_all_zero(self, inventory_df):
        nonzero = (inventory_df["stockout_flag"] != 0).sum()
        assert nonzero == 0, (
            f"C2: stockout_flag now has {nonzero} non-zero rows. "
            "The derived metric logic may need revisiting."
        )

    def test_row_count_35000(self, inventory_df):
        assert len(inventory_df) == 35_000, (
            f"C2: inventory row count changed: {len(inventory_df)} (expected 35,000)"
        )

    def test_derived_stockout_risk_computable(self, inventory_df):
        """Can compute the derived metric without NaNs on key columns."""
        required = {"closing_stock", "safety_stock"}
        assert required.issubset(inventory_df.columns), (
            f"C2: columns for derived stockout_risk_event missing: "
            f"{required - set(inventory_df.columns)}"
        )
        stockout_risk = (
            (inventory_df["closing_stock"] <= 0)
            | (inventory_df["closing_stock"] < inventory_df["safety_stock"])
        )
        assert stockout_risk.notna().all()


# ---------------------------------------------------------------------------
# C3  inventory ledger identity does NOT hold
# ---------------------------------------------------------------------------

class TestC3LedgerIdentity:
    """
    opening + received + production_received - units_sold = closing_stock holds
    on 0% of rows. Rows are independently sampled; ledger replay is invalid.
    Twin must be validated distributionally (spec section 4.7).
    """

    def test_ledger_identity_fails(self, inventory_df):
        df = inventory_df
        check = (
            df["opening_stock"]
            + df["received_quantity"]
            + df["production_received"]
            - df["units_sold"]
        )
        pct_holds = ((check - df["closing_stock"]).abs() < 0.01).mean() * 100
        assert pct_holds < 5.0, (
            f"C3: Ledger identity now holds on {pct_holds:.1f}% of rows. "
            "If this significantly increased, the data may have changed and "
            "the twin validation strategy should be re-evaluated."
        )


# ---------------------------------------------------------------------------
# C5  duplicate keys in financials and inventory
# ---------------------------------------------------------------------------

class TestC5Duplicates:
    """
    15_financials has approx 7,204 duplicate (date, product_id) rows.
    05_inventory has approx 794 duplicate (date, product_id, warehouse_id) rows.
    Naive joins would fan out and double-count money. Must aggregate before joining.
    """

    def test_financials_has_duplicates(self, financials_df):
        n_dups = financials_df.duplicated(subset=["date", "product_id"]).sum()
        assert n_dups > 0, (
            "C5: Expected duplicate (date, product_id) in financials, found none. "
            "Verify aggregation logic is still appropriate."
        )
        assert 5_000 <= n_dups <= 10_000, (
            f"C5: financials duplicate count changed significantly: {n_dups} (expected approx 7,204)"
        )

    def test_inventory_has_duplicates(self, inventory_df):
        n_dups = inventory_df.duplicated(subset=["date", "product_id", "warehouse_id"]).sum()
        assert n_dups > 0, (
            "C5: Expected duplicate (date, product_id, warehouse_id) in inventory, found none."
        )
        assert 500 <= n_dups <= 1_200, (
            f"C5: inventory duplicate count changed significantly: {n_dups} (expected approx 794)"
        )


# ---------------------------------------------------------------------------
# C6  profit identity and negative profit
# ---------------------------------------------------------------------------

class TestC6ProfitIdentity:
    """
    profit = revenue - (production_cost + raw_material_cost + labor_cost +
                        energy_cost + storage_cost + logistics_cost)
    holds on 100% of rows. 48.1% of rows have negative profit.
    Do not clip negatives; surface the loss-making share prominently.
    """

    COST_COLS = [
        "production_cost", "raw_material_cost", "labor_cost",
        "energy_cost", "storage_cost", "logistics_cost",
    ]

    def test_cost_columns_present(self, financials_df):
        missing = [c for c in self.COST_COLS if c not in financials_df.columns]
        assert not missing, f"C6: cost columns missing: {missing}"

    def test_profit_identity_holds(self, financials_df):
        derived = financials_df["revenue"] - financials_df[self.COST_COLS].sum(axis=1)
        pct_match = ((derived - financials_df["profit"]).abs() < 0.01).mean() * 100
        assert pct_match >= 99.0, (
            f"C6: profit identity holds on {pct_match:.1f}% of rows (expected approx 100%). "
            "Check for data changes or aggregation errors."
        )

    def test_negative_profit_not_clipped(self, financials_df):
        """Assert there are negative profit rows -- clipping would fail this."""
        pct_neg = (financials_df["profit"] < 0).mean() * 100
        assert pct_neg >= 40.0, (
            f"C6: Only {pct_neg:.1f}% negative-profit rows found (expected approx 48.1%). "
            "If data was clipped or corrected this test will fail."
        )

    def test_negative_profit_not_too_high(self, financials_df):
        """Sanity bound: not more than 70% negative (would suggest data corruption)."""
        pct_neg = (financials_df["profit"] < 0).mean() * 100
        assert pct_neg <= 70.0, (
            f"C6: {pct_neg:.1f}% negative-profit rows -- unexpectedly high, check data."
        )


# ---------------------------------------------------------------------------
# C8  QC release_status FAIL is rare
# ---------------------------------------------------------------------------

class TestC8ReleaseStatus:
    """
    release_status: PASS 81.4%, RETEST 18.1%, FAIL 0.5% (approx 175 rows).
    FAIL is too rare for a reliable supervised classifier.
    Use RETEST union FAIL as the 'non-pass' label; report FAIL recall separately.
    """

    def test_expected_statuses_present(self, qc_df):
        vals = set(qc_df["release_status"].unique())
        assert {"PASS", "RETEST", "FAIL"}.issubset(vals), (
            f"C8: Expected PASS/RETEST/FAIL, got {vals}"
        )

    def test_fail_count_small(self, qc_df):
        n_fail = int((qc_df["release_status"] == "FAIL").sum())
        assert n_fail < 500, (
            f"C8: FAIL count is {n_fail}, much higher than expected approx 175. "
            "Reconsider whether supervised FAIL classifier is now feasible."
        )
        assert n_fail > 50, (
            f"C8: FAIL count dropped to {n_fail} -- check data integrity."
        )

    def test_pass_pct_approx_81(self, qc_df):
        pct = (qc_df["release_status"] == "PASS").mean() * 100
        assert 70 <= pct <= 90, (
            f"C8: PASS % changed significantly: {pct:.1f}% (expected approx 81.4%)"
        )

    def test_nonpass_label_is_retest_union_fail(self, qc_df):
        """RETEST union FAIL should cover all non-PASS rows exactly."""
        nonpass = qc_df[qc_df["release_status"] != "PASS"]
        all_covered = nonpass["release_status"].isin(["RETEST", "FAIL"]).all()
        assert all_covered, (
            "C8: Non-PASS rows contain statuses other than RETEST/FAIL. "
            "The evaluation label definition must be updated."
        )