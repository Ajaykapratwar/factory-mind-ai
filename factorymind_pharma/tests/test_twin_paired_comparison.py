"""
Unit test asserting Common Random Numbers and Paired Comparison per Spec §4.5 & §6 (Criterion 5).
"""
import numpy as np
import pytest
from factorymind_pharma.services.twin_svc import DigitalTwinSimulator
from factorymind_pharma.contracts.scenario import Scenario, SupplierDelay, MachineOutage


def test_twin_paired_comparison_common_random_numbers():
    sim = DigitalTwinSimulator()
    sc = Scenario(
        id="SC_PAIRED_TEST",
        name="Paired CRN Test",
        horizon_days=30,
        n_paths=200,
        seed=12345,
        shocks=[
            SupplierDelay(supplier_id="S004", extra_days=5, start_day=3)
        ],
    )

    # Run twice with same seed
    res1 = sim.run_scenario(sc)
    res2 = sim.run_scenario(sc)

    # Identical results due to seed determinism
    assert res1.baseline_metrics.expected_profit == pytest.approx(res2.baseline_metrics.expected_profit)
    assert res1.shocked_metrics.expected_profit == pytest.approx(res2.shocked_metrics.expected_profit)
    assert res1.profit_delta == pytest.approx(res2.profit_delta)
    assert res1.profit_at_risk_p5 == pytest.approx(res2.profit_at_risk_p5)

    # Baseline with no shocks should have higher or equal profit compared to shocked
    assert res1.baseline_metrics.expected_profit >= res1.shocked_metrics.expected_profit

    # Paired run with different seed produces slightly different numbers
    sc_diff = Scenario(
        id="SC_DIFF_SEED",
        name="Diff Seed",
        horizon_days=30,
        n_paths=200,
        seed=99999,
        shocks=[
            SupplierDelay(supplier_id="S004", extra_days=5, start_day=3)
        ],
    )
    res_diff = sim.run_scenario(sc_diff)
    assert res1.baseline_metrics.expected_profit != res_diff.baseline_metrics.expected_profit


def test_fact_financial_c6_identity():
    fi = sim = DigitalTwinSimulator().financial
    df = fi.build_fact_financial()
    assert len(df) == 27796
    assert (df["profit"] < 0).mean() > 0.40  # ~48% loss-making share preserved without clipping
    modeled = fi.compute_modeled_costs()
    for k, v in modeled.items():
        assert v["label"] == "modeled"
