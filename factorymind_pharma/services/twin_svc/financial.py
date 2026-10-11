"""
Financial Intelligence Layer per Spec §5.
Builds fact_financial, unit economics, modeled costs, and forward-looking financial metrics.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2].parent / "data"

COST_COLS = [
    "production_cost",
    "raw_material_cost",
    "labor_cost",
    "energy_cost",
    "storage_cost",
    "logistics_cost",
]


class FinancialIntelligence:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self._fact_financial: Optional[pd.DataFrame] = None
        self._unit_margins: Optional[pd.Series] = None
        self._coverage: float = 0.0

    def build_fact_financial(self) -> pd.DataFrame:
        """
        §5.1: Aggregate 15_financials by (date, product_id) using sum (C5).
        Validates C6 identity on the aggregated table.
        Never clips negative profit.
        """
        if self._fact_financial is not None:
            return self._fact_financial

        fin_path = self.data_dir / "15_financials.csv"
        fin = pd.read_csv(fin_path)

        agg_cols = COST_COLS + ["revenue", "profit"]
        fact_fin = fin.groupby(["date", "product_id"], as_index=False)[agg_cols].sum()

        # Derived fields
        total_cost = fact_fin[COST_COLS].sum(axis=1)
        fact_fin["total_cost"] = total_cost

        # C6 identity check: profit = revenue - sum(costs) on 100% of rows
        computed_profit = fact_fin["revenue"] - total_cost
        diff = (fact_fin["profit"] - computed_profit).abs()
        max_diff = diff.max()
        assert max_diff < 1e-4, f"C6 identity violated in fact_financial: max diff = {max_diff}"

        # margin_pct: null when revenue == 0, never imputed
        fact_fin["margin_pct"] = np.where(
            fact_fin["revenue"] == 0,
            np.nan,
            fact_fin["profit"] / fact_fin["revenue"]
        )

        for c in COST_COLS:
            fact_fin[f"cost_share_{c}"] = np.where(
                total_cost == 0,
                0.0,
                fact_fin[c] / total_cost
            )

        fact_fin["is_loss_making"] = fact_fin["profit"] < 0
        self._fact_financial = fact_fin
        return fact_fin

    def compute_unit_economics(self) -> Tuple[pd.Series, float]:
        """
        §5.2: Join product-day units_sold where both exist; report coverage.
        Unit margin per product = trailing-90-day median of profit / units.
        Rows without unit counts are excluded from unit metrics, not zero-filled.
        """
        if self._unit_margins is not None:
            return self._unit_margins, self._coverage

        fact_fin = self.build_fact_financial().copy()
        sales = pd.read_csv(self.data_dir / "02_sales_demand.csv")
        sales_agg = sales.groupby(["date", "product_id"], as_index=False)["units_sold"].sum()

        merged = pd.merge(fact_fin, sales_agg, on=["date", "product_id"], how="inner")
        self._coverage = len(merged) / len(fact_fin)

        merged["date"] = pd.to_datetime(merged["date"])
        max_date = merged["date"].max()
        cutoff = max_date - pd.Timedelta(days=90)
        recent = merged[merged["date"] >= cutoff].copy()

        recent["unit_margin"] = recent["profit"] / recent["units_sold"]
        # Median per product over trailing 90 days
        margins = recent.groupby("product_id")["unit_margin"].median()

        # If any SKU not in recent, fill from full merged median (not zero)
        all_skus = fact_fin["product_id"].unique()
        full_margins = merged.groupby("product_id")["profit"].sum() / np.maximum(
            merged.groupby("product_id")["units_sold"].sum(), 1
        )
        for sku in all_skus:
            if sku not in margins or pd.isna(margins[sku]):
                margins[sku] = full_margins.get(sku, merged["profit"].median() / max(merged["units_sold"].median(), 1))

        self._unit_margins = margins
        return margins, self._coverage

    def compute_modeled_costs(
        self,
        retest_delay_days: int = 3,
        holding_cost_per_unit_day: float = 0.05,
    ) -> Dict:
        """
        §5.3: Modeled cost estimates (always labelled 'modeled').
        - Cost of poor quality
        - Downtime cost
        - Supplier risk cost
        """
        unit_margins, _ = self.compute_unit_economics()

        # 1. Cost of poor quality:
        # (FAIL batches * batch value scrapped) + (RETEST batches * retest_delay_days * holding_cost)
        qc = pd.read_csv(self.data_dir / "08_quality_control.csv")
        bm = pd.read_csv(self.data_dir / "07_batch_manufacturing.csv")
        pm = pd.read_csv(self.data_dir / "01_product_master.csv")

        unit_cost_map = pm.set_index("product_id")["production_cost"].to_dict()
        batch_size_map = bm.set_index("batch_id")["batch_size"].to_dict() if "batch_size" in bm.columns else {}
        batch_prod_map = bm.set_index("batch_id")["product_id"].to_dict() if "product_id" in bm.columns else {}

        n_fail = int((qc["release_status"] == "FAIL").sum())
        n_retest = int((qc["release_status"] == "RETEST").sum())

        avg_batch_size = float(bm["batch_size"].median()) if "batch_size" in bm.columns else 5000.0
        avg_unit_cost = float(pm["production_cost"].median()) if "production_cost" in pm.columns else 50.0
        avg_batch_value = avg_batch_size * avg_unit_cost

        scrapped_value = n_fail * avg_batch_value
        retest_holding_cost = n_retest * retest_delay_days * avg_batch_size * holding_cost_per_unit_day
        cost_poor_quality = scrapped_value + retest_holding_cost

        # 2. Downtime cost:
        # 10.maintenance_cost + downtime_hours * lost_margin_per_hour
        maint = pd.read_csv(self.data_dir / "10_maintenance.csv")
        direct_maint_cost = float(maint["maintenance_cost"].sum())
        downtime_hours = float(maint["downtime_hours"].sum())

        # Line throughput per hour proxy
        prod = pd.read_csv(self.data_dir / "06_production.csv")
        daily_units = prod["actual_quantity"].median()
        units_per_hour = daily_units / 24.0
        median_unit_margin = max(float(unit_margins.median()), 10.0)
        lost_margin_per_hour = units_per_hour * median_unit_margin
        lost_margin_total = downtime_hours * lost_margin_per_hour
        total_downtime_cost = direct_maint_cost + lost_margin_total

        # 3. Supplier risk cost: expected late-delivery days * holding_cost_proxy
        supp = pd.read_csv(self.data_dir / "12_suppliers.csv")
        unreliability = 1.0 - (supp["delivery_reliability_pct"].mean() / 100.0)
        mean_delivery_days = float(supp["delivery_time_days"].mean())
        expected_late_days = unreliability * mean_delivery_days * len(supp)
        supplier_risk_cost = expected_late_days * avg_batch_size * holding_cost_per_unit_day

        return {
            "cost_of_poor_quality": {
                "value": cost_poor_quality,
                "label": "modeled",
                "components": {
                    "fail_batches": n_fail,
                    "retest_batches": n_retest,
                    "scrapped_value": scrapped_value,
                    "retest_holding_cost": retest_holding_cost,
                },
                "assumptions": f"retest_delay={retest_delay_days}d, holding_cost={holding_cost_per_unit_day}/u-d, batch_size={avg_batch_size:.0f}",
            },
            "downtime_cost": {
                "value": total_downtime_cost,
                "label": "modeled",
                "components": {
                    "direct_maintenance_cost": direct_maint_cost,
                    "downtime_hours": downtime_hours,
                    "lost_margin": lost_margin_total,
                },
                "assumptions": f"throughput={units_per_hour:.1f} u/h, unit_margin={median_unit_margin:.2f}",
            },
            "supplier_risk_cost": {
                "value": supplier_risk_cost,
                "label": "modeled",
                "components": {
                    "expected_late_days": expected_late_days,
                },
                "assumptions": f"unreliability={unreliability:.1%}, holding_cost={holding_cost_per_unit_day}/u-d",
            },
        }

    def compute_forward_financials(
        self,
        demand_forecast: Dict[str, float],
        twin_profit_samples: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        §5.4: Expected profit (next 30 days) = ensemble forecast * unit margin.
        Profit-at-Risk (PaR) = P5 of baseline-minus-scenario profit.
        """
        unit_margins, _ = self.compute_unit_economics()
        expected_profit = 0.0

        for sku, fcast in demand_forecast.items():
            margin = unit_margins.get(sku, unit_margins.median())
            expected_profit += fcast * margin

        par_p5 = 0.0
        profit_p5 = expected_profit * 0.85
        profit_p95 = expected_profit * 1.15

        if twin_profit_samples is not None and len(twin_profit_samples) > 0:
            profit_p5 = float(np.percentile(twin_profit_samples, 5))
            profit_p95 = float(np.percentile(twin_profit_samples, 95))
            # PaR is 5th percentile delta vs expected
            par_p5 = float(expected_profit - profit_p5)

        return {
            "expected_profit_30d": expected_profit,
            "profit_p5": profit_p5,
            "profit_p95": profit_p95,
            "profit_at_risk_p5": par_p5,
        }
