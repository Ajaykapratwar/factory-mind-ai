# CHANGELOG.md — FactoryMind Pharma Phase 0–1

## Phase 0 — Scaffold & Contracts

### Step 0.1 — Repo Scaffold
- Created `factorymind_pharma/` directory tree as per spec.
- Services: `sensing_svc/`, `forecast_svc/` (implemented), `twin_svc/`, `policy_svc/`, `tower_svc/` (stubs).
- Dashboard: `dashboard/app.py`, `pages/executive.py`, `pages/operational.py`, `components/charts.py`, `assets/tokens.css`, `assets/style.css`.
- Evaluation: `evaluation/metrics.py`.
- Tests: `tests/test_metrics.py`.

### Step 0.2 — Config
- `configs/config.yaml` defines: data paths, train/val/test ratios (0.80/0.10/0.10), seed 42, LightGBM params, feature lists.

### Step 0.3 — Load & Validate All 15 Source Files
- Per-file schema dict with expected columns for all 15 files.
- FK validation: product_id in files 02–06 confirmed subset of product_master.
- Batch uniqueness validated (no duplicate batch_id for same date+product in production).
- Row counts printed on load.
- Files 07–15 (Phase 2+ stubs): schema validated only — no feature engineering applied.
- **Stub schema surprises:** None. All 9 stub files have well-formed headers matching expected schemas.

## Phase 1 — Foundation: Data Layer + Core Dashboards

### Step 1.1 — Master Feature Table
- Anchor: `sales_demand` (date × product_id × region × channel).
- Left-join: `demand_factors` on (date, product_id).
- Left-join: `orders` aggregated (daily ordered_quantity sum, order count) on (date, product_id).
- Left-join: `inventory` aggregated (sum closing/opening/received stock, max stockout_flag) on (date, product_id).
- Product dimensions attached from `product_master`.
- Null imputation: forward-fill per product_id; imputed flags added.
- Output: `data/processed/master_feature_table.parquet`.

### Step 1.2 — Baseline Forecast
- Naive seasonal: last-year-same-day-of-week (364-day lag), with 28-day fallback then median.
- LightGBM: lag-7, lag-28, rolling-mean-7/28 of units_sold + demand factor columns + time features + encoded categoricals.
- Chronological split (no shuffle): train 80% → val 10% → test 10%.
- Predictions saved: `data/outputs/test_predictions.parquet`.

### Step 1.3 — Evaluation Metrics Module
- Full metric suite: MAE, MSE, RMSE, MAPE, SMAPE (new vs reference), WAPE, R², RMSE%, MAE%, Bias%, Rolling bias, Pinball loss stub.
- Per-SKU report saved: `data/outputs/evaluation_report.csv`.
- `check_acceptance()` with Phase 1 thresholds.
- Unit tests: `tests/test_metrics.py`.

### Step 1.4 — Data Access Layer
- `dashboard/data_loader.py` exposes: `get_kpi_summary()`, `get_demand_vs_actual()`, `get_inventory_status()`, `get_per_sku_mape_table()`.

### Step 1.5 — Dashboard 1: Executive Summary
- KPI tiles: revenue, units sold, forecast accuracy (1−WAPE), avg days-of-supply, stockout count.
- Plan-vs-actual line chart (all SKUs aggregate).
- Top/bottom 5 SKUs by MAPE table.
- Custom light palette, DM Serif Display headings, Inter body.

### Step 1.6 — Dashboard 2: Operational Planning
- Warehouse inventory table (opening/received/sold/closing/safety/reorder/stockout).
- Order fulfillment horizontal bar chart by status.
- Product drill-down: Dropdown → demand history + forecast + inventory trend with safety stock and reorder point overlays.
