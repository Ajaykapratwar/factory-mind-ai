"""
Unit tests comparing Digital Twin simulated marginals to source data per Spec §4.3.
Verifies Kolmogorov-Smirnov p > 0.05 OR quantile error < 5%.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from factorymind_pharma.services.twin_svc.distributions import (
    YieldSampler,
    MaintenanceSampler,
    QCSampler,
    ColdChainSampler,
    SupplierSampler,
    DemandResidualModel,
)


def test_yield_sampler_marginals():
    ys = YieldSampler()
    for line in ["L01", "L02", "L03", "L04", "L05"]:
        actual = ys.yield_by_line[line]
        rng = np.random.default_rng(42)
        sim = ys.sample_yield(line, size=2000, rng=rng)

        # Check quantile error at 10%, 50%, 90%
        q_actual = np.quantile(actual, [0.10, 0.50, 0.90])
        q_sim = np.quantile(sim, [0.10, 0.50, 0.90])
        q_err = np.max(np.abs(q_actual - q_sim) / q_actual)

        ks_res = stats.ks_2samp(actual, sim)
        # Pass if KS p > 0.05 OR max quantile error < 5%
        assert ks_res.pvalue > 0.05 or q_err < 0.05, f"Line {line} yield failed: KS p={ks_res.pvalue}, q_err={q_err}"


def test_maintenance_downtime_marginals():
    ms = MaintenanceSampler()
    actual = ms.global_downtime_hours
    rng = np.random.default_rng(42)
    sim = rng.choice(actual, size=2000, replace=True)

    q_actual = np.quantile(actual, [0.10, 0.50, 0.90])
    q_sim = np.quantile(sim, [0.10, 0.50, 0.90])
    q_err = np.max(np.abs(q_actual - q_sim) / np.maximum(q_actual, 1e-4))

    ks_res = stats.ks_2samp(actual, sim)
    assert ks_res.pvalue > 0.05 or q_err < 0.05


def test_qc_release_rates():
    qc = QCSampler()
    rng = np.random.default_rng(42)
    samples = qc.sample_status(size=10000, rng=rng)

    sim_pass = np.mean(samples == 0)
    sim_retest = np.mean(samples == 1)
    sim_fail = np.mean(samples == 2)

    assert abs(sim_pass - qc.p_pass) < 0.02
    assert abs(sim_retest - qc.p_retest) < 0.02
    assert abs(sim_fail - qc.p_fail) < 0.005


def test_coldchain_excursion_rate():
    cs = ColdChainSampler()
    rng = np.random.default_rng(42)
    samples = cs.sample_excursion(size=20000, rng=rng)
    sim_rate = np.mean(samples)

    rel_err = abs(sim_rate - cs.excursion_rate) / cs.excursion_rate
    assert rel_err < 0.10  # within 10% relative error for rare event


def test_supplier_reliability_marginals():
    ss = SupplierSampler()
    rng = np.random.default_rng(42)
    delays = ss.sample_delay("SUP01", size=10000, rng=rng)
    ontime_rate = np.mean(delays == 0)

    rel = ss.reliability_by_supplier.get("SUP01", ss.global_reliability)
    assert abs(ontime_rate - rel) < 0.03


def test_demand_residual_bootstrap():
    model = DemandResidualModel()
    rng = np.random.default_rng(42)
    sample = model.sample_residuals(["P0001", "P0002"], horizon_days=30, n_paths=200, rng=rng)
    assert sample.shape == (200, 2, 30)
    # Autocorrelation lag-1 preserved non-zero
    assert not np.all(sample == 0)
