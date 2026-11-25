# FactoryMind AI — Pharma Control Tower

**FactoryMind AI** is a full-stack pharmaceutical manufacturing intelligence platform. It ingests 15 operational data streams, runs a multi-model ML ensemble for demand forecasting, and surfaces insights through 10 live dashboards backed by a FastAPI control-tower REST service, an AI Copilot, and a governance audit chain — all across a 7-phase build (Phase 0 → Phase 6).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [ML Models & Forecasting](#ml-models--forecasting)
- [Dashboard Pages](#dashboard-pages)
- [REST API — Control Tower](#rest-api--control-tower)
- [AI Copilot](#ai-copilot)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Configuration](#configuration)
- [Known Caveats](#known-caveats)

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────┐
│                   Raw Data (15 CSVs)                  │
└───────────────────┬───────────────────────────────────┘
                    │  sensing_svc (loader + feature_builder)
                    ▼
         master_feature_table.parquet
                    │
        ┌───────────┴────────────┐
        │                        │
   forecast_svc             policy_svc
  (Phase 1: LightGBM         (inventory
   Phase 2: Ensemble)         optimizer)
        │
   models/*.pkl  ◄──── ensemble_trainer
        │
   ┌────┴───────────────────────────────────────┐
   │               tower_svc (FastAPI)           │
   │  audit_log · decision_engine · replay_clock │
   │  copilot/client · event_bus                 │
   └────────────────┬───────────────────────────┘
                    │
   ┌────────────────▼────────────────────────────┐
   │      Dash Enterprise Dashboard (port 8050)  │
   │  10 pages · AI Copilot drawer · CSS tokens  │
   └─────────────────────────────────────────────┘
```

---

## Project Structure

```
factory-mind-ai/
├── factorymind_pharma/
│   ├── run_pipeline.py          # Phase 0 + Phase 1 end-to-end runner
│   ├── run_phase2.py            # Phase 2 + Phase 3 ensemble & risk runner
│   ├── requirements.txt
│   ├── configs/
│   │   ├── config.yaml          # Data paths, model params, split ratios
│   │   ├── copilot.yaml         # Groq LLM & guard model config
│   │   ├── policy.yaml          # Safety-stock & reorder policy thresholds
│   │   └── twin.yaml            # Digital twin simulation parameters
│   ├── contracts/
│   │   ├── decision.py          # Pydantic decision state model
│   │   ├── scenario.py          # Pydantic scenario/twin contract
│   │   └── openapi.json         # OpenAPI spec snapshot
│   ├── dashboard/
│   │   ├── app.py               # Dash multi-page app entrypoint
│   │   ├── data_loader.py       # Shared data access layer
│   │   ├── assets/style.css     # CSS design token variables
│   │   └── pages/
│   │       ├── executive.py
│   │       ├── operational.py
│   │       ├── forecast_analytics.py
│   │       ├── machine_health.py
│   │       ├── supplier_qc.py
│   │       ├── inventory_optimization.py
│   │       ├── compliance_logistics.py
│   │       ├── financial_impact.py
│   │       ├── scenario_lab.py
│   │       └── control_tower.py
│   ├── evaluation/
│   │   └── metrics.py           # Full metric suite (MAE, RMSE, WAPE, SMAPE, Bias …)
│   ├── scripts/
│   │   └── generate_output_csvs.py   # Phase 3 re-measure & output tables
│   ├── services/
│   │   ├── sensing_svc/         # Data ingestion, schema validation, feature engineering
│   │   ├── forecast_svc/        # Baseline (LightGBM), ensemble trainer, diagnostics
│   │   ├── policy_svc/          # Inventory optimizer (safety stock, reorder points)
│   │   ├── maintenance_svc/     # Machine failure hazard models
│   │   ├── qc_svc/              # Isolation Forest QC anomaly scoring
│   │   ├── supplier_svc/        # Supplier risk scoring
│   │   ├── twin_svc/            # Stochastic digital twin (Monte Carlo, CRN, financials)
│   │   └── tower_svc/
│   │       ├── api.py           # FastAPI app (all REST endpoints)
│   │       ├── audit_log.py     # Tamper-evident SHA-256 audit chain (SQLite)
│   │       ├── decision_engine.py  # L5 approval queue (BLOCK / REVIEW / AUTO)
│   │       ├── replay_clock.py  # Closed-loop simulation clock
│   │       ├── event_bus.py
│   │       └── copilot/
│   │           ├── client.py    # Groq LLM client + offline deterministic fallback
│   │           ├── grounding.py # Data grounding + provenance strips
│   │           └── tools.py     # Tool definitions for AI function calling
│   ├── models/                  # Trained model artifacts (git-ignored)
│   └── tests/
│       ├── test_metrics.py
│       ├── test_data_facts.py
│       ├── test_no_leakage.py
│       ├── test_no_rigged_checks.py
│       ├── test_closed_loop_30day.py
│       ├── test_twin_distributions.py
│       ├── test_twin_paired_comparison.py
│       ├── test_property_block_tier.py
│       ├── test_phase6_orchestration.py
│       ├── test_copilot_eval.py
│       └── acceptance/
│           └── test_phase4_6_acceptance.py
├── data/
│   ├── raw/                     # 15 source CSVs (place here before running)
│   ├── processed/               # master_feature_table.parquet (auto-generated)
│   └── outputs/                 # evaluation reports, predictions, risk scores
├── docs/                        # MODEL_UPGRADE_REPORT, ACCEPTANCE_REPORT, ASSUMPTIONS …
├── backup_state/                # Pre-run artifact snapshots
├── .env.example                 # API key template
├── GETTING_STARTED.md           # Full step-by-step run guide
└── README.md
```

---

## Data Sources

The platform ingests **15 CSV files** covering 100 pharmaceutical SKUs at daily granularity (2021-01-01 → 2022-12-31, ~35 000 rows per operational table).

| # | File | Grain | Key Columns |
|---|------|-------|-------------|
| 01 | `01_product_master.csv` | 1 row/product | `product_id`, `category`, `dosage_form`, `criticality`, `unit_price` |
| 02 | `02_sales_demand.csv` | date × product × region × channel | `units_sold`, `revenue` |
| 03 | `03_demand_factors.csv` | date × product | `market_demand_index`, `disease_season_index`, `promotion_flag`, `competitor_price` |
| 04 | `04_orders.csv` | order | `order_id`, `ordered_quantity`, `order_status` |
| 05 | `05_inventory.csv` | date × product × warehouse | `closing_stock`, `stockout_flag`, `reorder_point` |
| 06 | `06_production.csv` | date × batch | `batch_id`, `produced_quantity` |
| 07 | `07_batch_manufacturing.csv` | batch | `batch_id`, `yield_pct` |
| 08 | `08_quality_control.csv` | batch | `pass_fail`, `defect_rate` |
| 09 | `09_machine_sensor_data.csv` | timestamp × machine | `machine_id`, sensor telemetry |
| 10 | `10_maintenance.csv` | event | `machine_id`, `maintenance_type` |
| 11 | `11_raw_materials.csv` | material dim | `raw_material_id` |
| 12 | `12_suppliers.csv` | supplier × material | `supplier_id`, `lead_time`, `reliability_score` |
| 13 | `13_logistics.csv` | shipment | `shipment_id`, `temperature_reading` |
| 14 | `14_compliance.csv` | batch inspection | `batch_id`, `audit_flag` |
| 15 | `15_financials.csv` | date × product | `cogs`, `gross_profit`, `revenue` |

> [!NOTE]
> Place all 15 CSVs into `data/raw/` before running the pipeline. Schema validation runs on all 15 files; foreign-key integrity is enforced for files 01–06.

---

## ML Models & Forecasting

### Phase 1 — Baseline

| Model | Description |
|-------|-------------|
| **Naive Seasonal** | Last-year same-day-of-week per SKU — the floor every later model must beat |
| **LightGBM** | Lag-7, lag-28, rolling-mean-7/28, demand factor features; chronological 80/10/10 split |

### Phase 2 — Multi-Model Ensemble

| Model | Artifact |
|-------|----------|
| LightGBM (tuned) | `models/lgbm.pkl` / `lgbm_booster.txt` |
| XGBoost | `models/xgb.pkl` / `xgb_model.json` |
| CatBoost | `models/catboost.pkl` / `catboost_model.cbm` |
| Holt-Winters (per-SKU) | In-memory, 100 SKU models |
| Stacking Meta-Learner | `models/meta_learner.pkl` |
| Segment LightGBM | `models/lgbm_segments.pkl` |

### Phase 2 — Operational Risk Models

| Service | Method | Output |
|---------|--------|--------|
| **Machine Health** (`maintenance_svc`) | Survival/hazard model | `data/outputs/maintenance_risk_scores.parquet` |
| **QC Anomaly** (`qc_svc`) | Isolation Forest | `data/outputs/qc_anomaly_scores.parquet` |
| **Supplier Risk** (`supplier_svc`) | Composite risk scoring | `data/outputs/supplier_risk_scores.parquet` |

### Evaluation Metrics

Implemented in `evaluation/metrics.py` — covers point-forecast accuracy, bias, and probabilistic metrics:

`MAE · MSE · RMSE · MAPE · SMAPE · WAPE · R² · RMSE% · MAE% · Bias% · Rolling Bias (30-day) · Pinball Loss`

**Phase 1 acceptance thresholds:**

| Metric | Threshold |
|--------|-----------|
| R² | ≥ 0.55 |
| MAPE | ≤ 25% |
| SMAPE | ≤ 25% |
| WAPE | ≤ 20% |
| Bias (abs) | ≤ 8% |
| RMSE% | ≤ 25% |
| MAE% | ≤ 20% |

---

## Dashboard Pages

The Dash app (`factorymind_pharma/dashboard/app.py`) serves **10 pages** at `http://localhost:8050`:

| Route | Page | Key Elements |
|-------|------|--------------|
| `/` | **Executive Summary** | Revenue KPI, WAPE badge, service level, top/bottom 5 SKUs by MAPE, plan-vs-actual chart |
| `/operational` | **Operational Telemetry** | Warehouse-level inventory table, order fulfillment funnel, per-SKU drill-down |
| `/forecast` | **Forecast Analytics** | Ensemble predictions per SKU, confidence bands, WAPE & R² |
| `/machine-health` | **Machine Health** | Failure hazard probabilities, sensor telemetry trends, wear heuristics |
| `/supplier-qc` | **Supplier & QC** | Vendor risk matrix, delivery unreliability, Isolation Forest QC scatter |
| `/inventory-opt` | **Inventory Optimization** | Safety stock bands, reorder point status, stockout risk distribution |
| `/compliance` | **Compliance & Logistics** | Cold-chain excursion alerts, batch audit flags |
| `/financial-impact` | **Financial Impact** | COGS breakdown, gross margin by category, loss-making SKU share |
| `/scenario-lab` | **Scenario Lab** | Monte Carlo "what-if" shock builder, CRN-paired comparison, mitigation ranking |
| `/control-tower` | **Control Tower** | L5 approval queue (BLOCK / REVIEW / AUTO), SHA-256 audit badge, replay clock |

The floating **AI Copilot** button (✦) is available on every page.

---

## REST API — Control Tower

The FastAPI service runs at `http://127.0.0.1:8000`. Docs at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/readyz` | Readiness probe |
| `GET` | `/v1/decisions` | Control Tower approval queue |
| `GET` | `/v1/audit/verify` | SHA-256 cryptographic audit chain verification |
| `GET` | `/v1/audit/entries` | Full audit trail entries |
| `GET` | `/v1/replay/status` | Simulation replay clock status |
| `POST` | `/v1/replay/step` | Advance replay clock by 1 day |
| `GET` | `/v1/models/registry` | Champion/challenger model governance registry |
| `GET` | `/v1/sensing/machine/{machine_id}` | Machine telemetry & risk |
| `GET` | `/v1/forecast/{product_id}` | Demand forecast & confidence intervals |
| `GET` | `/v1/policy/inventory/{product_id}` | Inventory position & reorder point |
| `POST` | `/v1/twin/simulate` | Stochastic digital twin Monte Carlo simulation |

**Start the API:**
```powershell
python -m uvicorn factorymind_pharma.services.tower_svc.api:app --host 127.0.0.1 --port 8000
```

---

## AI Copilot

| Setting | Value |
|---------|-------|
| **Provider** | Groq (`https://api.groq.com/openai/v1`) |
| **Primary LLM** | `openai/gpt-oss-120b` |
| **Fallback LLM** | `llama-3.3-70b-versatile` |
| **Offline mode** | `deterministic-engine-v1` (activates automatically when no `GROQ_API_KEY`) |
| **Prompt defense** | In-process regex heuristics in `copilot/client.py` (`_check_security_and_intent()`) |

Every response includes a `model_used` field and a provenance strip grounded in real data.

**Configure the API key (optional — offline mode works without it):**

```env
# .env
GROQ_API_KEY=gsk_your_actual_key_here
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+ (tested on 3.14)
- The 15 source CSVs placed in `data/raw/`

### 1. Activate the virtual environment

**PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**Bash / macOS / Linux:**
```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r factorymind_pharma/requirements.txt
```

**Key packages:** `pandas · numpy · pyarrow · lightgbm · xgboost · catboost · scikit-learn · statsmodels · scipy · dash · plotly · fastapi · uvicorn · httpx · pydantic · groq · pytest · hypothesis · pyyaml`

### 3. (Optional) Copy and configure `.env`

```powershell
Copy-Item .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
```

---

## Running the Project

> [!IMPORTANT]
> Run steps in order. Each phase's outputs are inputs to the next.

### Step 1 — Phase 0 & 1: Data pipeline + baseline models (~16 s)

```powershell
cd factorymind_pharma
python -X utf8 run_pipeline.py
cd ..
```

**Outputs:** `data/processed/master_feature_table.parquet` · `data/outputs/evaluation_report.csv` · `data/outputs/test_predictions.parquet`

### Step 2 — Phase 2 & 3: Ensemble, risk models + policy gates (~4 min)

```powershell
cd factorymind_pharma
python -X utf8 run_phase2.py
cd ..
```

**Outputs:** `models/*.pkl` · `data/outputs/maintenance_risk_scores.parquet` · `supplier_risk_scores.parquet` · `qc_anomaly_scores.parquet` · `docs/MODEL_UPGRADE_REPORT.md`

### Step 3 — Generate output CSVs (~6 s)

```powershell
python -X utf8 factorymind_pharma/scripts/generate_output_csvs.py
```

**Outputs:** `data/outputs/phase3_remeasure.csv` · `data/outputs/evaluation_report_phase2.csv`

### Step 4 — Launch Control Tower REST API

```powershell
python -m uvicorn factorymind_pharma.services.tower_svc.api:app --host 127.0.0.1 --port 8000
```

→ API: http://127.0.0.1:8000 · Swagger: http://127.0.0.1:8000/docs

### Step 5 — Launch Dashboard

```powershell
python -X utf8 factorymind_pharma/dashboard/app.py
```

→ Dashboard: http://127.0.0.1:8050

> [!TIP]
> Before running steps 1–3 for the first time, back up tracked artifacts:
> ```powershell
> New-Item -ItemType Directory -Force -Path .\backup_state | Out-Null
> Copy-Item -Recurse -Force .\factorymind_pharma\models .\backup_state\models
> Copy-Item -Recurse -Force .\factorymind_pharma\docs .\backup_state\docs
> if (Test-Path .\data\outputs) { Copy-Item -Recurse -Force .\data\outputs .\backup_state\outputs }
> ```

---

## Testing

```powershell
# Full test suite — unit, invariants, closed-loop, property tests (97/98 pass)
python -X utf8 -m pytest factorymind_pharma/tests/ -v --tb=short

# Formal acceptance suite — Phase 4–6 exit criteria
python -X utf8 -m pytest factorymind_pharma/tests/acceptance/ -v --tb=short
```

**Test suite summary (98 total):**

| Test File | Coverage |
|-----------|----------|
| `test_metrics.py` | All metric functions against hand-computed values |
| `test_data_facts.py` | Schema integrity, FK checks, row counts |
| `test_no_leakage.py` | No future data leaks into training features |
| `test_no_rigged_checks.py` | Acceptance checks are not gamed |
| `test_closed_loop_30day.py` | 30-day replay-clock simulation |
| `test_twin_distributions.py` | Digital twin stochastic distribution checks |
| `test_twin_paired_comparison.py` | Paired CRN comparison correctness |
| `test_property_block_tier.py` | Hypothesis property-based decision tier tests |
| `test_phase6_orchestration.py` | Governance + audit chain orchestration |
| `test_copilot_eval.py` | AI copilot grounding & response quality |
| `acceptance/test_phase4_6_acceptance.py` | Phase 4–6 formal exit criteria (17/18 pass) |

**Benchmarked timings:**
- `run_pipeline.py` — ~15.9 s
- `run_phase2.py` — ~4 min 7 s
- `generate_output_csvs.py` — ~6.2 s
- `pytest` full run — ~1 min 14 s

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/config.yaml` | Data file paths, train/val/test split (80/10/10 chronological), LightGBM hyperparameters, feature lag/rolling windows |
| `configs/copilot.yaml` | Groq model names, guard model reference, copilot behaviour flags |
| `configs/policy.yaml` | Safety-stock multipliers, reorder policy thresholds |
| `configs/twin.yaml` | Digital twin simulation: warehouses, supplier disruption params, Monte Carlo iterations |
| `.env` | `GROQ_API_KEY` (optional — offline mode activates automatically if absent) |

---

## Known Caveats

> [!WARNING]
> **Docker Compose** (`factorymind_pharma/docker-compose.yml`) does not work as-is. The build context paths are mismatched (`context: .` inside `factorymind_pharma/` conflicts with `COPY requirements.txt` from repo root), volume mounts resolve incorrectly, and the port mapping targets 8501 instead of 8050. **Use the local `venv` as described above.**

> [!NOTE]
> **1 known failing test** — `test_phase4_6_acceptance.py::TestPhase4::test_paired_common_random_numbers`: the initial warehouse stock is high enough that a 3-day supplier delay on `S001` over a 30-day window does not trigger a stockout, so simulated profit is identical in both scenarios and the assertion `baseline_profit ≠ shocked_profit` fails. All other 97 tests pass.

> [!NOTE]
> **Groq model status**: `openai/gpt-oss-120b` returns HTTP 200. `llama-3.3-70b-versatile` returns HTTP 404 (deprecated on Groq). The copilot automatically falls back to **offline deterministic mode** (`deterministic-engine-v1`) when the Groq SDK or key is unavailable — no errors, just keyword-matched grounded responses.

---

## Phase Roadmap

| Phase | Description | Entry Point |
|-------|-------------|-------------|
| **Phase 0** | Scaffold, config, FK validation for all 15 CSVs | `run_pipeline.py` |
| **Phase 1** | Master feature table, LightGBM baseline, 2 dashboards | `run_pipeline.py` |
| **Phase 2** | Full ensemble (LightGBM + XGBoost + CatBoost + meta-learner), QC, maintenance, supplier risk | `run_phase2.py` |
| **Phase 3** | Inventory safety-stock policy gates, digital-twin re-measure | `generate_output_csvs.py` |
| **Phase 4** | Vectorized stochastic digital twin, financial P&L, Monte Carlo scenario lab | On-demand via UI / `POST /v1/twin/simulate` |
| **Phase 5** | AI Copilot — tool-use, grounding, prompt injection defense | On-demand via UI / `CopilotClient.ask()` |
| **Phase 6** | Governance — L5 approval queue, tamper-evident audit chain, closed-loop replay clock | On-demand via UI / `POST /v1/replay/step` |

For the complete step-by-step run guide, see [GETTING_STARTED.md](GETTING_STARTED.md).
