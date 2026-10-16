"""
Empirical stochastic component models for Digital Twin per Spec §4.3.
Fitted from historical CSVs without hand-picked Gaussians.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2].parent / "data"


class DemandResidualModel:
    """
    Block-bootstrapped residuals per SKU / segment (block length 7)
    preserving autocorrelation per Spec §4.3.
    """
    def __init__(self, block_len: int = 7, data_dir: Optional[Path] = None):
        self.block_len = block_len
        self.data_dir = data_dir or DATA_DIR
        self.residuals_by_sku: Dict[str, np.ndarray] = {}
        self.blocks_by_sku: Dict[str, List[np.ndarray]] = {}
        self.global_residuals: np.ndarray = np.array([])
        self.global_blocks: List[np.ndarray] = []
        self._fit()

    def _fit(self):
        full_grid_path = self.data_dir / "outputs" / "test_predictions_full_grid.parquet"
        pred_path = self.data_dir / "outputs" / "test_predictions.parquet"
        if full_grid_path.exists():
            tp = pd.read_parquet(full_grid_path)
            # Use hurdle or lgb pred
            pred_col = "pred_hurdle" if "pred_hurdle" in tp.columns else ("pred_lgb" if "pred_lgb" in tp.columns else "units_sold")
            tp["residual"] = tp["units_sold"].values - tp[pred_col].values
            self.global_residuals = tp["residual"].values
            for sku, grp in tp.groupby("product_id"):
                res = grp.sort_values("date")["residual"].values
                self.residuals_by_sku[sku] = res
                if len(res) >= self.block_len:
                    blocks = [res[i:i + self.block_len] for i in range(len(res) - self.block_len + 1)]
                    self.blocks_by_sku[sku] = blocks
        elif pred_path.exists():
            tp = pd.read_parquet(pred_path)
            pred_col = "lgb_pred" if "lgb_pred" in tp.columns else "units_sold"
            tp["residual"] = tp["units_sold"].values - tp[pred_col].values
            self.global_residuals = tp["residual"].values
            for sku, grp in tp.groupby("product_id"):
                res = grp.sort_values("date")["residual"].values
                self.residuals_by_sku[sku] = res
                if len(res) >= self.block_len:
                    blocks = [res[i:i + self.block_len] for i in range(len(res) - self.block_len + 1)]
                    self.blocks_by_sku[sku] = blocks
        else:
            # Fallback to 02_sales_demand demeaned
            sales = pd.read_csv(self.data_dir / "02_sales_demand.csv")
            sales["date"] = pd.to_datetime(sales["date"])
            means = sales.groupby("product_id")["units_sold"].transform("mean")
            sales["residual"] = sales["units_sold"] - means
            self.global_residuals = sales["residual"].values
            for sku, grp in sales.groupby("product_id"):
                res = grp.sort_values("date")["residual"].values
                self.residuals_by_sku[sku] = res
                if len(res) >= self.block_len:
                    blocks = [res[i:i + self.block_len] for i in range(len(res) - self.block_len + 1)]
                    self.blocks_by_sku[sku] = blocks

        if len(self.global_residuals) >= self.block_len:
            self.global_blocks = [
                self.global_residuals[i:i + self.block_len]
                for i in range(0, len(self.global_residuals) - self.block_len + 1, self.block_len)
            ]

    def sample_residuals(
        self,
        sku_list: List[str],
        horizon_days: int,
        n_paths: int = 500,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Returns tensor of shape [n_paths, len(sku_list), horizon_days]
        using vectorized block bootstrap.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n_skus = len(sku_list)
        n_blocks = int(np.ceil(horizon_days / self.block_len))
        out = np.zeros((n_paths, n_skus, horizon_days), dtype=np.float32)

        for s_idx, sku in enumerate(sku_list):
            blocks = self.blocks_by_sku.get(sku, self.global_blocks)
            if not blocks:
                # Fallback if no blocks
                continue
            n_avail = len(blocks)
            sampled_idx = rng.integers(0, n_avail, size=(n_paths, n_blocks))
            # Shape [n_paths, n_blocks, block_len]
            block_arr = np.array(blocks)[sampled_idx]
            # Flatten to [n_paths, n_blocks * block_len] and slice to horizon_days
            flat = block_arr.reshape(n_paths, -1)[:, :horizon_days]
            out[:, s_idx, :] = flat

        return out


class YieldSampler:
    """
    Empirical resample of yield_percentage per production line from 06_production.csv.
    """
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.yield_by_line: Dict[str, np.ndarray] = {}
        self.global_yield: np.ndarray = np.array([])
        self._fit()

    def _fit(self):
        prod = pd.read_csv(self.data_dir / "06_production.csv")
        self.global_yield = prod["yield_percentage"].dropna().values
        for line, grp in prod.groupby("production_line"):
            self.yield_by_line[line] = grp["yield_percentage"].dropna().values

    def sample_yield(
        self,
        line_id: str,
        size: Union[int, Tuple[int, ...]],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(42)
        vals = self.yield_by_line.get(line_id, self.global_yield)
        return rng.choice(vals, size=size, replace=True)

    def mean_yield(self, line_id: Optional[str] = None) -> float:
        if line_id and line_id in self.yield_by_line:
            return float(np.mean(self.yield_by_line[line_id]))
        return float(np.mean(self.global_yield))


class MaintenanceSampler:
    """
    Per-machine failure hazard and conditioned downtime hours from 09 & 10.
    """
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.daily_hazard_by_machine: Dict[str, float] = {}
        self.downtime_by_failure_type: Dict[str, np.ndarray] = {}
        self.global_downtime_hours: np.ndarray = np.array([])
        self.global_hazard: float = 0.05
        self._fit()

    def _fit(self):
        maint = pd.read_csv(self.data_dir / "10_maintenance.csv")
        maint["date"] = pd.to_datetime(maint["date"])
        n_days = (maint["date"].max() - maint["date"].min()).days + 1

        corr = maint[maint["maintenance_type"] == "Corrective"]
        self.global_downtime_hours = corr["downtime_hours"].dropna().values

        # Per failure type downtime
        if "failure_type" in corr.columns:
            for f_type, grp in corr.groupby("failure_type"):
                self.downtime_by_failure_type[f_type] = grp["downtime_hours"].dropna().values

        # Hazard per machine
        for m_id, grp in corr.groupby("machine_id"):
            self.daily_hazard_by_machine[m_id] = len(grp) / max(n_days, 1)

        self.global_hazard = len(corr) / (max(n_days, 1) * max(len(maint["machine_id"].unique()), 1))

    def sample_outage(
        self,
        machine_id: str,
        days: int,
        n_paths: int = 500,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            failure_events: boolean array [n_paths, days]
            downtime_hours: float array [n_paths, days]
        """
        if rng is None:
            rng = np.random.default_rng(42)

        hazard = self.daily_hazard_by_machine.get(machine_id, self.global_hazard)
        failures = rng.random(size=(n_paths, days)) < hazard

        downtimes = np.zeros((n_paths, days), dtype=np.float32)
        n_fails = np.sum(failures)
        if n_fails > 0:
            sampled_dt = rng.choice(self.global_downtime_hours, size=n_fails, replace=True)
            downtimes[failures] = sampled_dt

        return failures, downtimes


class QCSampler:
    """
    Per-product outcome sampled from empirical release_status rates from 08_quality_control.csv.
    PASS: 81.4%, RETEST: 18.1%, FAIL: 0.5%.
    """
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.p_pass = 0.814371
        self.p_retest = 0.180629
        self.p_fail = 0.005000
        self._fit()

    def _fit(self):
        qc = pd.read_csv(self.data_dir / "08_quality_control.csv")
        vc = qc["release_status"].value_counts(normalize=True).to_dict()
        self.p_pass = vc.get("PASS", 0.814)
        self.p_retest = vc.get("RETEST", 0.181)
        self.p_fail = vc.get("FAIL", 0.005)

    def sample_status(
        self,
        size: Union[int, Tuple[int, ...]],
        retest_multiplier: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        0 = PASS, 1 = RETEST, 2 = FAIL
        """
        if rng is None:
            rng = np.random.default_rng(42)

        retest_p = min(0.99, self.p_retest * retest_multiplier)
        rem = max(0.001, 1.0 - retest_p)
        fail_p = min(rem, self.p_fail)
        pass_p = rem - fail_p

        probs = [pass_p, retest_p, fail_p]
        return rng.choice([0, 1, 2], size=size, p=probs)


class SupplierSampler:
    """
    Bernoulli(delivery_reliability_pct) on-time; if late, delay ~ resampled excess over delivery_time_days.
    """
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.reliability_by_supplier: Dict[str, float] = {}
        self.delays_by_supplier: Dict[str, np.ndarray] = {}
        self.global_reliability: float = 0.82
        self.global_delays: np.ndarray = np.array([2, 3, 5, 7])
        self._fit()

    def _fit(self):
        supp = pd.read_csv(self.data_dir / "12_suppliers.csv")
        self.global_reliability = float(supp["delivery_reliability_pct"].mean() / 100.0)

        # In supplier data, delivery_time_days has variance
        lead_times = supp["delivery_time_days"].dropna().values
        median_lt = np.median(lead_times)
        excess = lead_times[lead_times > median_lt] - median_lt
        if len(excess) > 0:
            self.global_delays = excess

        for s_id, grp in supp.groupby("supplier_id"):
            self.reliability_by_supplier[s_id] = float(grp["delivery_reliability_pct"].mean() / 100.0)
            lt = grp["delivery_time_days"].dropna().values
            m_lt = np.median(lt)
            ex = lt[lt > m_lt] - m_lt
            if len(ex) > 0:
                self.delays_by_supplier[s_id] = ex

    def sample_delay(
        self,
        supplier_id: str,
        size: Union[int, Tuple[int, ...]],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Returns extra days of delay (0 if on-time, >0 if late).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        rel = self.reliability_by_supplier.get(supplier_id, self.global_reliability)
        is_late = rng.random(size=size) > rel

        delays = np.zeros(size, dtype=np.int32)
        n_late = np.sum(is_late)
        if n_late > 0:
            pool = self.delays_by_supplier.get(supplier_id, self.global_delays)
            sampled = rng.choice(pool, size=n_late, replace=True)
            delays[is_late] = np.maximum(1, np.round(sampled).astype(np.int32))

        return delays


class ColdChainSampler:
    """
    Per-shipment excursion ~ Bernoulli(empirical rate ≈ 4.0%).
    """
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.excursion_rate = 0.039886
        self._fit()

    def _fit(self):
        log = pd.read_csv(self.data_dir / "13_logistics.csv")
        self.excursion_rate = float(log["temperature_excursion_flag"].mean())

    def sample_excursion(
        self,
        size: Union[int, Tuple[int, ...]],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(42)
        return rng.random(size=size) < self.excursion_rate
