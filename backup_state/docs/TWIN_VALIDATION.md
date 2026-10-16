# FactoryMind AI (Pharma) — Digital Twin Validation Report (Spec §4.7)

**Validation Window:** Held-out test period 2022-10-20 to 2022-12-31 (final 10% chronologically, 73 days).
**Validation Methodology:** Distributional backtest on held-out actuals per Carry-Forward Correction C3.

---

## 1. Summary of Results

| Validation Metric | Actual / Reference | Simulated Twin | Target / Tolerance | Status |
|---|---|---|---|---|
| **Demand P10–P90 Coverage** | Actual sales in heldout | 81.48% | 75.0% – 85.0% | **PASS** |
| **Production Mean Yield** | 95.79% | 95.80% | ±1.00 percentage point | **PASS** |
| **Corrective Downtime Hours** | 4,534.36 h | 4,612.70 h | ±20.0% (1.73% err) | **PASS** |

---

## 2. Metric Details

### 2.1 Demand Interval Calibration
- **Model:** Phase 2 ensemble point forecast + block-bootstrapped validation residuals (block length 7).
- **Result:** Simulated P10–P90 coverage is **81.48%**, falling strictly within the mandated 75–85% envelope.
- **Autocorrelation:** Block length 7 preserves weekly cyclicality and lag-1 autocorrelation in demand residuals.

### 2.2 Production Yield
- **Model:** Empirical resampling of line-level `yield_percentage` from `06_production.csv`.
- **Actual Held-Out Mean:** 95.79%.
- **Simulated Mean:** 95.80% (difference: +0.01 pp, well within the ±1.0 pp target).

### 2.3 Machine Downtime
- **Model:** Machine failure hazard and conditioned downtime hours from `10_maintenance.csv` corrective maintenance events.
- **Actual Held-Out Downtime:** 4,534.36 hours.
- **Simulated Downtime:** 4,612.70 hours (relative error: 1.73%, well within the ±20% budget).
