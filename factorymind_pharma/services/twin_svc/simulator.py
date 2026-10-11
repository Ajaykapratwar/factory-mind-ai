"""
Vectorized Stochastic Digital Twin Simulator per Spec §4.
Implements NumPy tensor daily-step simulation, Common Random Numbers,
shock composition, and mitigation search.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import numpy as np
import pandas as pd

from factorymind_pharma.contracts.scenario import (
    Scenario,
    MachineOutage,
    SupplierDelay,
    DemandShock,
    ColdChainEvent,
    QualityShift,
    MitigationAction,
    ScenarioMetricSummary,
    ScenarioComparisonResult,
)
from factorymind_pharma.services.twin_svc.distributions import (
    DemandResidualModel,
    YieldSampler,
    MaintenanceSampler,
    QCSampler,
    SupplierSampler,
    ColdChainSampler,
)
from factorymind_pharma.services.twin_svc.financial import FinancialIntelligence


DATA_DIR = Path(__file__).resolve().parents[2].parent / "data"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "twin.yaml"


class DigitalTwinSimulator:
    def __init__(self, config_path: Optional[Path] = None, data_dir: Optional[Path] = None):
        self.config_path = config_path or CONFIG_PATH
        self.data_dir = data_dir or DATA_DIR

        # Load config
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Stochastic models
        self.demand_model = DemandResidualModel(data_dir=self.data_dir)
        self.yield_sampler = YieldSampler(data_dir=self.data_dir)
        self.maint_sampler = MaintenanceSampler(data_dir=self.data_dir)
        self.qc_sampler = QCSampler(data_dir=self.data_dir)
        self.supp_sampler = SupplierSampler(data_dir=self.data_dir)
        self.coldchain_sampler = ColdChainSampler(data_dir=self.data_dir)
        self.financial = FinancialIntelligence(data_dir=self.data_dir)

        # Precompute static topology
        self.lines = [l["id"] for l in self.config["topology"]["lines"]]
        self.line_capacities = {l["id"]: l["capacity_units_per_day"] for l in self.config["topology"]["lines"]}
        self.warehouses = self.config["topology"]["warehouses"]["ids"]
        self.n_wh = len(self.warehouses)
        self.wh_to_idx = {w: i for i, w in enumerate(self.warehouses)}

        # Load SKUs and unit margins
        self.pm = pd.read_csv(self.data_dir / "01_product_master.csv")
        self.skus = sorted(self.pm["product_id"].unique().tolist())
        self.n_skus = len(self.skus)
        self.sku_to_idx = {s: i for i, s in enumerate(self.skus)}
        self.sku_category = self.pm.set_index("product_id")["category"].to_dict()
        self.sku_criticality = self.pm.set_index("product_id")["criticality"].to_dict()

        self.unit_margins, _ = self.financial.compute_unit_economics()
        self.sku_margin_arr = np.array([self.unit_margins.get(s, 50.0) for s in self.skus], dtype=np.float32)
        price_map = self.pm.set_index("product_id")["unit_price"].to_dict()
        self.sku_price_arr = np.array([price_map.get(s, 100.0) for s in self.skus], dtype=np.float32)

        # Precompute base daily demand mean per SKU
        sales = pd.read_csv(self.data_dir / "02_sales_demand.csv")
        mean_sales = sales.groupby("product_id")["units_sold"].mean()
        self.base_demand = np.array([mean_sales.get(s, 200.0) for s in self.skus], dtype=np.float32)

        # Load initial inventory & safety stock per SKU x warehouse
        self._load_inventory_state()

    def _load_inventory_state(self):
        inv = pd.read_csv(self.data_dir / "05_inventory.csv")
        # Median closing stock and safety stock per SKU x warehouse
        stock_pivot = inv.groupby(["product_id", "warehouse_id"])["closing_stock"].median().unstack(fill_value=2000.0)
        ss_pivot = inv.groupby(["product_id", "warehouse_id"])["safety_stock"].median().unstack(fill_value=500.0)

        # Safety stock optimized if available
        opt_ss_path = self.data_dir / "outputs" / "safety_stock_optimized.csv"
        if opt_ss_path.exists():
            opt_df = pd.read_csv(opt_ss_path)
            if "optimized_safety_stock" in opt_df.columns:
                opt_map = opt_df.set_index("product_id")["optimized_safety_stock"].to_dict()
                for s in self.skus:
                    if s in opt_map:
                        # allocate across warehouses
                        ss_pivot.loc[s] = opt_map[s] / float(self.n_wh)

        self.initial_stock = np.zeros((self.n_skus, self.n_wh), dtype=np.float32)
        self.safety_stock = np.zeros((self.n_skus, self.n_wh), dtype=np.float32)

        for s_idx, s in enumerate(self.skus):
            for w_idx, w in enumerate(self.warehouses):
                self.initial_stock[s_idx, w_idx] = float(stock_pivot.get(w, {}).get(s, 2500.0))
                self.safety_stock[s_idx, w_idx] = float(ss_pivot.get(w, {}).get(s, 500.0))

    def _generate_demand_tensor(
        self,
        horizon_days: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generates demand tensor: [n_paths, n_skus, n_wh, horizon_days]
        using base forecast + block-bootstrapped residuals.
        """
        base_demand = self.base_demand

        # Residuals [n_paths, n_skus, horizon_days]
        res_sample = self.demand_model.sample_residuals(self.skus, horizon_days, n_paths, rng=rng)

        # Total SKU demand [n_paths, n_skus, horizon_days]
        sku_demand = np.maximum(0.0, base_demand[None, :, None] + res_sample)

        # Each warehouse experiences empirical daily demand (~327 u/day)
        demand_tensor = np.repeat(sku_demand[:, :, None, :], self.n_wh, axis=2)
        return demand_tensor

    def simulate_run(
        self,
        scenario: Scenario,
        is_baseline: bool = False,
        mitigation: Optional[str] = None,
    ) -> Dict:
        """
        Executes single vectorized simulation run.
        """
        rng = np.random.default_rng(scenario.seed)
        horizon = scenario.horizon_days
        n_paths = scenario.n_paths

        # Generate stochastic demand [n_paths, n_skus, n_wh, horizon]
        demand = self._generate_demand_tensor(horizon, n_paths, rng)

        # Base daily replenishment
        # Average replenishment per SKU-WH-day to sustain steady state
        base_replenish = demand.mean(axis=(0, 3), keepdims=True) # [1, n_skus, n_wh, 1]
        replenish = np.broadcast_to(base_replenish, (n_paths, self.n_skus, self.n_wh, horizon)).copy()

        # Apply Shocks if not baseline
        if not is_baseline:
            for shock in scenario.shocks:
                if isinstance(shock, DemandShock):
                    pct = shock.pct_change
                    s_day = shock.start_day
                    e_day = min(horizon, s_day + shock.duration_days)
                    if s_day < horizon:
                        # Find affected SKUs
                        for s_idx, s in enumerate(self.skus):
                            cat_match = shock.category and self.sku_category.get(s) == shock.category
                            sku_match = shock.product_id and s == shock.product_id
                            if cat_match or sku_match:
                                demand[:, s_idx, :, s_day:e_day] *= (1.0 + pct)

                elif isinstance(shock, MachineOutage):
                    # Outage causes production reduction
                    s_day = shock.start_day
                    duration_days = max(1, int(np.ceil(shock.duration_h / 24.0)))
                    e_day = min(horizon, s_day + duration_days)
                    if s_day < horizon:
                        capacity_loss = min(1.0, shock.duration_h / (24.0 * duration_days))
                        # Reduces replenishment across SKUs by capacity share
                        replenish[:, :, :, s_day:e_day] *= (1.0 - 0.25 * capacity_loss)

                elif isinstance(shock, SupplierDelay):
                    s_day = shock.start_day
                    delay = shock.extra_days
                    if s_day < horizon:
                        e_day = min(horizon, s_day + delay)
                        # Zero replenishment during delay, delayed arrival after
                        delayed_vol = replenish[:, :, :, s_day:e_day].copy()
                        replenish[:, :, :, s_day:e_day] = 0.0
                        if e_day < horizon:
                            replenish[:, :, :, e_day:min(horizon, e_day + delay)] += delayed_vol

                elif isinstance(shock, QualityShift):
                    # Multiplies retest delay / quarantine
                    retest_delay = self.config["parameters"]["retest_delay_days"]["value"]
                    mult = shock.retest_rate_multiplier
                    # Reduces on-time batch release by delay fraction
                    replenish *= max(0.5, 1.0 - (0.05 * mult))

                elif isinstance(shock, ColdChainEvent):
                    s_day = shock.start_day
                    e_day = min(horizon, s_day + shock.duration_days)
                    if s_day < horizon:
                        # Excursions quarantine batches (reduce replenishment)
                        replenish[:, :, :, s_day:e_day] *= 0.70

        # Apply Mitigations if specified
        mitigation_cost = 0.0
        if mitigation == "expedite_raw_material":
            mitigation_cost = self.config["parameters"]["raw_material_expedite_cost"]["value"]
            # Recovers delayed replenishment back to baseline
            replenish = np.maximum(replenish, np.broadcast_to(base_replenish, (n_paths, self.n_skus, self.n_wh, horizon)))

        elif mitigation == "pull_forward_maintenance":
            mitigation_cost = self.config["parameters"]["maintenance_pull_forward_cost"]["value"]
            # Prevents outage capacity loss
            replenish = np.maximum(replenish, np.broadcast_to(base_replenish, (n_paths, self.n_skus, self.n_wh, horizon)))

        elif mitigation == "reprioritize_skus":
            # Reallocate line capacity to high-margin / high-criticality SKUs
            high_priority = [i for i, s in enumerate(self.skus) if self.sku_criticality.get(s) in ["Critical", "High"]]
            replenish[:, high_priority, :, :] *= 1.25

        elif mitigation == "inter_warehouse_transfer":
            transfer_unit_cost = self.config["parameters"]["transfer_cost_per_unit"]["value"]
            # Active inventory rebalancing between warehouses (assumed lead time 2 days)
            mitigation_cost = transfer_unit_cost * 2000.0

        # Simulate inventory step-by-step
        stock = np.broadcast_to(self.initial_stock, (n_paths, self.n_skus, self.n_wh)).copy()
        safety_stock_tensor = np.broadcast_to(self.safety_stock, (n_paths, self.n_skus, self.n_wh))

        total_demand = np.sum(demand, axis=3) # [n_paths, n_skus, n_wh]
        total_sold = np.zeros_like(total_demand)
        stockout_occurred = np.zeros_like(total_demand, dtype=bool)

        for t in range(horizon):
            d_t = demand[:, :, :, t]
            r_t = replenish[:, :, :, t]

            # Fulfill demand
            sold_t = np.minimum(stock, d_t)
            stock = stock - sold_t + r_t

            total_sold += sold_t
            # Stockout event: closing stock <= 0 OR stock < safety_stock (C2)
            stockout_occurred |= (stock <= 0.0) | (stock < safety_stock_tensor)

            if mitigation == "inter_warehouse_transfer" and t % 3 == 0:
                # Rebalance across warehouses towards mean
                mean_wh_stock = np.mean(stock, axis=2, keepdims=True)
                stock = 0.8 * stock + 0.2 * mean_wh_stock

        total_lost = np.maximum(0.0, total_demand - total_sold) # [n_paths, n_skus, n_wh]

        # Financial valuation:
        # Earned potential margin on full demand minus lost value on unfulfilled units
        demand_per_sku = np.sum(total_demand, axis=2) # [n_paths, n_skus]
        lost_per_sku = np.sum(total_lost, axis=2) # [n_paths, n_skus]
        earned_profit_per_path = np.sum(demand_per_sku * self.sku_margin_arr[None, :], axis=1)
        unit_loss_value = np.maximum(self.sku_margin_arr, self.sku_price_arr * 0.15)
        lost_profit_per_path = np.sum(lost_per_sku * unit_loss_value[None, :], axis=1)
        profit_per_path = earned_profit_per_path - lost_profit_per_path - mitigation_cost
        lost_margin_per_path = lost_profit_per_path

        fill_rate_per_path = np.sum(total_sold, axis=(1, 2)) / np.maximum(np.sum(total_demand, axis=(1, 2)), 1.0)
        p_stockout_per_sku_wh = np.mean(stockout_occurred, axis=0) # [n_skus, n_wh]
        overall_p_stockout = float(np.mean(p_stockout_per_sku_wh))

        return {
            "p_stockout_mean": overall_p_stockout,
            "p_stockout_p95": float(np.percentile(np.mean(stockout_occurred, axis=(1, 2)), 95)),
            "expected_lost_units": float(np.mean(np.sum(total_lost, axis=(1, 2)))),
            "fill_rate_p5": float(np.percentile(fill_rate_per_path, 5)),
            "fill_rate_p50": float(np.median(fill_rate_per_path)),
            "fill_rate_p95": float(np.percentile(fill_rate_per_path, 95)),
            "days_of_supply_mean": float(np.mean(stock) / max(float(demand.mean()), 1.0)),
            "expected_profit": float(np.mean(profit_per_path)),
            "profit_p5": float(np.percentile(profit_per_path, 5)),
            "profit_p95": float(np.percentile(profit_per_path, 95)),
            "profit_per_path": profit_per_path,
        }

    def run_scenario(self, scenario: Scenario) -> ScenarioComparisonResult:
        """
        §4.5: Runs baseline and shocked runs with Common Random Numbers (identical seeds).
        §4.6: Evaluates fixed menu of mitigation actions.
        """
        # Baseline run (no shocks)
        base_res = self.simulate_run(scenario, is_baseline=True)

        if not scenario.shocks:
            shock_res = base_res
            profit_delta = 0.0
            par_p5 = 0.0
            p_stockout_delta = 0.0
            fill_rate_delta = 0.0
            ranked_mitigations: List[MitigationAction] = []
        else:
            # Shocked run
            shock_res = self.simulate_run(scenario, is_baseline=False)

            profit_delta = shock_res["expected_profit"] - base_res["expected_profit"]
            # PaR: P5 of baseline-minus-scenario profit (Spec §5.4 / §4.5)
            paired_profit_diff = base_res["profit_per_path"] - shock_res["profit_per_path"]
            par_p5 = float(np.percentile(paired_profit_diff, 95)) # 95th percentile loss = P5 downside
            p_stockout_delta = shock_res["p_stockout_mean"] - base_res["p_stockout_mean"]
            fill_rate_delta = shock_res["fill_rate_p50"] - base_res["fill_rate_p50"]

            # §4.6 Mitigation search
            candidate_mitigations = [
                ("expedite_raw_material", "Expedite Raw Material", "Pay express supplier dispatch to eliminate component lead time delay", self.config["parameters"]["raw_material_expedite_cost"]["value"]),
                ("inter_warehouse_transfer", "Inter-Warehouse Stock Transfer", "Balance inventory across regional distribution centers", self.config["parameters"]["transfer_cost_per_unit"]["value"] * 2000.0),
                ("reprioritize_skus", "Reprioritize Line Capacity", "Dynamically allocate finite production slots to critical and high-margin SKUs", 0.0),
                ("pull_forward_maintenance", "Pull Forward Maintenance", "Execute proactive maintenance during non-peak hours to avert catastrophic corrective outage", self.config["parameters"]["maintenance_pull_forward_cost"]["value"]),
            ]

            ranked_mitigations: List[MitigationAction] = []
            for act_id, act_name, act_desc, act_cost in candidate_mitigations:
                mit_res = self.simulate_run(scenario, is_baseline=False, mitigation=act_id)
                expected_saved = mit_res["expected_profit"] - shock_res["expected_profit"]
                ps_change = mit_res["p_stockout_mean"] - shock_res["p_stockout_mean"]

                ranked_mitigations.append(
                    MitigationAction(
                        id=act_id,
                        action_type=act_id,
                        name=act_name,
                        description=act_desc,
                        assumed_cost=float(act_cost),
                        expected_profit_saved=float(expected_saved),
                        p_stockout_delta=float(ps_change),
                    )
                )

        # Rank mitigations by expected profit saved descending
        ranked_mitigations.sort(key=lambda m: m.expected_profit_saved, reverse=True)

        return ScenarioComparisonResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            horizon_days=scenario.horizon_days,
            n_paths=scenario.n_paths,
            baseline_metrics=ScenarioMetricSummary(
                p_stockout_mean=base_res["p_stockout_mean"],
                p_stockout_p95=base_res["p_stockout_p95"],
                expected_lost_units=base_res["expected_lost_units"],
                fill_rate_p5=base_res["fill_rate_p5"],
                fill_rate_p50=base_res["fill_rate_p50"],
                fill_rate_p95=base_res["fill_rate_p95"],
                days_of_supply_mean=base_res["days_of_supply_mean"],
                expected_profit=base_res["expected_profit"],
                profit_p5=base_res["profit_p5"],
                profit_p95=base_res["profit_p95"],
            ),
            shocked_metrics=ScenarioMetricSummary(
                p_stockout_mean=shock_res["p_stockout_mean"],
                p_stockout_p95=shock_res["p_stockout_p95"],
                expected_lost_units=shock_res["expected_lost_units"],
                fill_rate_p5=shock_res["fill_rate_p5"],
                fill_rate_p50=shock_res["fill_rate_p50"],
                fill_rate_p95=shock_res["fill_rate_p95"],
                days_of_supply_mean=shock_res["days_of_supply_mean"],
                expected_profit=shock_res["expected_profit"],
                profit_p5=shock_res["profit_p5"],
                profit_p95=shock_res["profit_p95"],
            ),
            profit_delta=profit_delta,
            profit_at_risk_p5=par_p5,
            p_stockout_delta=p_stockout_delta,
            fill_rate_delta=fill_rate_delta,
            ranked_mitigations=ranked_mitigations,
        )
