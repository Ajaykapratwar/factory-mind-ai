

## Phase 2 — Intelligence Layer

### Step 2A.1 — §1 Residual Diagnostic
- Analysed residuals vs. predicted value and product_id (grouped).
- Confirmed which of 4 causes applied (see MODEL_UPGRADE_REPORT.md).

### Step 2A.2 — §2 Model Upgrade
- Escalation path: expanded features → per-segment → stacked ensemble → residual correction.
- Gap-closing step documented in MODEL_UPGRADE_REPORT.md.
- Artifacts: models/lgbm.pkl, models/xgb.pkl, models/catboost.pkl, models/meta_learner.pkl.
- Output: data/outputs/evaluation_report_phase2.csv (per-SKU full metric suite).

### Step 2B.1 — §3.1 QC Anomaly Detection
- Isolation Forest on assay_pct, dissolution_pct, impurity_level.
- Validated against release_status (PASS/FAIL) — precision/recall/F1 reported.
- Output: data/outputs/qc_anomaly_scores.parquet.

### Step 2B.2 — §3.2 Predictive Maintenance
- LightGBM classifier on sensor data + maintenance history.
- Time-based split (no future leakage). AUC requirement: ≥0.85.
- Output: data/outputs/maintenance_risk_scores.parquet.

### Step 2B.3 — §3.3 Supplier Risk Scoring
- Weighted composite: rejection_rate_pct × 0.6 + delivery_unreliability × 0.4.
- Deterministic rule (not a model) — scores normalised [0–100].
- Output: data/outputs/supplier_risk_scores.parquet.

### Step 2B.4 — Dashboards 3, 5, 7
- Dashboard 3: per-SKU forecast vs actual, confidence bands from ensemble variance.
- Dashboard 5: machine health risk scores, sensor trend charts.
- Dashboard 7: supplier risk scores, QC anomaly flags.
- All dashboards reference dashboard/assets/tokens.css unchanged (same light theme).

## Phase 3 — Optimization & Compliance

### Step 3.1 — §4.1 Safety Stock Optimization
- safety_stock = z × σ_demand × sqrt(lead_time).
- σ_demand from per-SKU ensemble residual std-dev on validation set.
- z per criticality tier: critical=2.33, standard=1.65, non-critical=1.28.
- Warehouse hierarchy: checked in 05_inventory.csv — result documented in ASSUMPTIONS.md.
- Backtest: simulated stockout reduction vs baseline reorder logic.
- Output: data/outputs/safety_stock_optimized.parquet.

### Step 3.2 — §4.2 Compliance Gate
- Deterministic hard rule: deviation_count > 0 OR audit_status != 'Passed' → blocked.
- Zero compliance-gated batches in auto-approved output (verified with synthetic test).

### Step 3.3 — §4.3 Cold-Chain Gate
- Deterministic hard rule: temperature_excursion_flag = 1 → blocked.
- Zero excursion-flagged shipments in auto-approved output (verified with synthetic test).

### Step 3.4 — Dashboards 4, 8, 9
- Dashboard 4: new vs old safety stock/reorder point, simulated stockout reduction.
- Dashboard 8/9: shipment delay trends, cold-chain excursion table by region.
- Same design tokens as Phase 0–1.
