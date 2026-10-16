# FactoryMind AI (Pharma) — Model Cards

> Phase 6 handover artifact · §13.2
> All metrics are on held-out data (final 10% chronologically). Numbers from `models/registry.json` and `evaluation/` outputs.

---

## MC-1: Demand Forecast Ensemble (Champion v1)

| Field | Value |
|---|---|
| **ID** | `ensemble-champion-v1` |
| **Status** | Champion |
| **Type** | Stacked Ensemble — LightGBM + XGBoost + CatBoost + Ridge meta-learner |
| **Purpose** | Point demand forecast per SKU per day; provides P10/P50/P90 intervals via block-bootstrapped residuals |
| **Training window** | 2021-01-01 → 2022-10-19 |
| **Held-out window** | 2022-10-20 → 2022-12-31 |
| **Input features** | Lag-1/7/14/28 demand, day-of-week, month, disease index, product category, warehouse |
| **Metrics (held-out)** | WAPE 20.2% · RMSE 35.8 · Bias -0.85% · R² 0.65 (row grain) |
| **R² grain note** | R² 0.65 is at row (product-day) grain. Product-week grain is higher; see `evaluation_report_phase2.csv`. The Phase 2 R² ≥ 0.80 bar is met at product-week grain (C7). |
| **Retrain trigger** | `|rolling_bias_30d| > 5%` sustained 14 days, or rolling WAPE > threshold × 1.25, or PSI > 0.2 on key features |
| **Rollback** | `models/registry.json` — promote previous champion; one command via `/v1/models/promote` |
| **Failure modes** | Sparse SKUs (< 100 training rows) have higher WAPE; bias may spike on disease-index events unseen in training |
| **Serialised to** | `models/lgbm.pkl`, `models/xgb.pkl`, `models/catboost.pkl`, `models/meta_learner.pkl` |

---

## MC-2: Demand Forecast Challenger (Candidate v2)

| Field | Value |
|---|---|
| **ID** | `ensemble-challenger-v2` |
| **Status** | Candidate (not promoted) |
| **Type** | Retrained Ensemble with enhanced lag features and disease index |
| **Training window** | 2021-01-01 → 2022-11-15 |
| **Metrics (held-out)** | WAPE 17.8% · RMSE 31.2 · Bias -0.4% · R² 0.72 |
| **Promotion gate** | Must beat champion WAPE by ≥ 2% on last-28-day holdout AND pass bias gate. Currently meets the WAPE gate (20.2% → 17.8% = 2.4 pp). Promotion pending QA sign-off. |

---

## MC-3: QC Anomaly Detector

| Field | Value |
|---|---|
| **ID** | `qc_anomaly_model` |
| **Type** | Isolation Forest (scikit-learn, contamination=0.05) |
| **Purpose** | Flag batches with unusual QC parameter profiles; generate anomaly scores and robust z-scores per field |
| **Training data** | `08_batch_quality` — PASS batches only (n ≈ 34 000, after RETEST/FAIL excluded from fit) |
| **Input features** | All numeric QC parameters (assay %, dissolution %, moisture %, hardness, pH, microbial count, particle size) |
| **Output** | `anomaly_score ∈ [-1, +1]`; fields outside P1–P99 PASS range ranked by robust z-score |
| **Label definition** | `RETEST ∪ FAIL` = "non-pass" (C8). FAIL n = 175; FAIL recall reported separately with support stated. |
| **Evaluation** | Precision/recall on held-out non-pass batches. FAIL recall is informational only (tiny support). |
| **Reference ranges** | Empirical P1–P99 of PASS batches per product category (≥ 200 batch support), else global. **Not regulatory spec limits.** |
| **Failure modes** | Novel process changes outside historical distribution will not be flagged reliably; drift detection required |
| **Serialised to** | `models/qc_anomaly_model.pkl` |

---

## MC-4: Maintenance Failure Classifier

| Field | Value |
|---|---|
| **ID** | `maintenance_classifier` |
| **Type** | Random Forest (scikit-learn) |
| **Purpose** | Predict next-30-day failure probability per machine; drives twin machine-availability simulation |
| **Training data** | `09_machine_sensors` + `10_maintenance` joined on `machine_id` / `date` |
| **Input features** | Vibration, temperature, operating hours, last maintenance gap, machine type |
| **Output** | `failure_prob ∈ [0, 1]`; risk tier LOW / MEDIUM / HIGH / CRITICAL |
| **Evaluation** | AUC on held-out machines (final 10% time window). Calibration checked via Brier score. |
| **Failure modes** | Machine types with sparse failure events (< 20 failures) have unreliable probability calibration |
| **Retrain policy** | Retrain with 30-day rolling window when drift flagged in sensor distributions (PSI > 0.2) |
| **Serialised to** | `models/maintenance_classifier.pkl` |

---

## MC-5: Supplier Risk Scorer

| Field | Value |
|---|---|
| **ID** | `supplier_risk_scorer` |
| **Type** | Rule-based composite (no ML) |
| **Purpose** | Score each supplier for delivery risk; drives twin supply-delay simulation and REVIEW-tier decisions |
| **Logic** | `risk_score = 0.5 × (1 - delivery_reliability_pct) + 0.5 × rejection_rate_pct`; normalised to [0,1] |
| **Data sources** | `12_suppliers` · `11_raw_materials` |
| **Tiers** | LOW < 0.3 · MEDIUM < 0.6 · HIGH ≥ 0.6 |
| **Failure modes** | Score is static per supplier; does not capture temporal degradation within the dataset window |
| **Serialised to** | `models/supplier_risk_scorer.pkl` (lightweight; stores per-supplier scores) |

---

## MC-6: Digital Twin Stochastic Components

These are not learned models but empirical fits; included here for completeness.

| Component | Method | Source data | Validation criterion |
|---|---|---|---|
| Demand residual bootstrap | Block bootstrap, block=7 days, per SKU segment | Ensemble validation residuals | KS p > 0.05 vs. held-out residuals |
| Production yield | Empirical resample of `yield_percentage` per line | `06_production` | Quantile error < 5% |
| Machine availability | Failure hazard from MC-4; outage duration from `downtime_hours` conditioned on `failure_type` | `09`, `10` | Simulated downtime within ±20% of actual |
| QC release rates | Empirical PASS/RETEST/FAIL rates per product; RETEST delays 3 days (assumed) | `08_batch_quality` | — |
| Raw-material delay | Bernoulli(`delivery_reliability_pct`); delay duration resampled from excess delivery days | `12_suppliers`, `11_raw_materials` | — |
| Cold-chain excursion | Bernoulli(empirical rate ≈ 4.0%) per shipment | `13_logistics` | — |

Twin interval calibration (back-test on held-out 10%): P10–P90 coverage target 75–85%. See `docs/TWIN_VALIDATION.md`.
