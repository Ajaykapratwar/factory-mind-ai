# FactoryMind AI (Pharma) — Getting Started & Run Guide

This guide details the exact sequence of commands to execute and verify the **FactoryMind AI (Pharma)** platform across all phases (Phase 0 to Phase 6) using the existing Python virtual environment (`venv`).

---

## 1. Prerequisites & Safety Backup

Running the pipeline and evaluation scripts retrains models and updates tracked artifacts. Before executing the pipeline, back up tracked state files:

### Tracked Files Modified by Runs
- `data/outputs/evaluation_report.csv`
- `data/outputs/evaluation_report_phase2.csv`
- `data/outputs/test_predictions.parquet`
- `factorymind_pharma/models/*.pkl` (`lgbm.pkl`, `xgb.pkl`, `catboost.pkl`, `meta_learner.pkl`, `lgbm_segments.pkl`)
- `factorymind_pharma/models/catboost_model.cbm`, `lgbm_booster.txt`, `xgb_model.json`
- `factorymind_pharma/catboost_info/*`
- `factorymind_pharma/docs/MODEL_UPGRADE_REPORT.md`
- `factorymind_pharma/docs/ACCEPTANCE_REPORT.md`
- `factorymind_pharma/docs/COPILOT_EVAL_REPORT.md`

### Recommended Backup Command (PowerShell)
```powershell
# From the project root:
New-Item -ItemType Directory -Force -Path .\backup_state | Out-Null
Copy-Item -Recurse -Force .\factorymind_pharma\models .\backup_state\models
Copy-Item -Recurse -Force .\factorymind_pharma\docs .\backup_state\docs
if (Test-Path .\data\outputs) { Copy-Item -Recurse -Force .\data\outputs .\backup_state\outputs }
```

### Linux / macOS (Bash)
```bash
mkdir -p backup_state
cp -r factorymind_pharma/models backup_state/models
cp -r factorymind_pharma/docs backup_state/docs
[ -d data/outputs ] && cp -r data/outputs backup_state/outputs
```

---

## 2. API Key Setup (Groq LLM Copilot)

The AI Copilot uses Groq to call `openai/gpt-oss-120b` (with automatic fallback to `llama-3.3-70b-versatile`). If no key is configured, the copilot automatically runs in **offline deterministic mode** (`deterministic-engine-v1`) without error.

> [!NOTE]
> Never commit or hardcode API keys. The system reads `GROQ_API_KEY` from the environment or `.env`.

### Option A: Via `.env` in Project Root (Recommended)
Copy `.env.example` to `.env` (or edit existing `.env`):
```powershell
# PowerShell
Copy-Item .env.example .env
```
```bash
# Bash
cp .env.example .env
```
Inside `.env`:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### Option B: Terminal Environment Variable

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY = "gsk_your_actual_key_here"
```

**Linux / macOS (Bash / Zsh):**
```bash
export GROQ_API_KEY="gsk_your_actual_key_here"
```

---

## 3. End-to-End Execution Sequence (Numbered)

All commands assume you start at the **project root** (`factory-mind-ai`).

### Step 1: Activate Virtual Environment
**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
*(Or on Linux/macOS: `source venv/bin/activate`)*

### Step 2: Verify Existing Dependencies
All required dependencies are pre-installed in the virtual environment. Verify imports:
```powershell
# Working Directory: project root
python -c "import pandas, numpy, pyarrow, lightgbm, xgboost, catboost, sklearn, statsmodels, scipy, yaml, dash, plotly, pytest, hypothesis, pydantic, groq, fastapi, httpx, uvicorn; print('[OK] All required packages present')"
```
*(If any package is missing, run: `pip install -r factorymind_pharma/requirements.txt`)*

### Step 3: Phase 0 & Phase 1 — Data Validation, Feature Engineering & Baseline Model
Executes schema validation, foreign-key integrity across all 15 source files, builds `master_feature_table.parquet`, and trains baseline Naive + LightGBM models.
```powershell
# Working Directory: factorymind_pharma/
cd factorymind_pharma
python -X utf8 run_pipeline.py
cd ..
```
- **Working directory**: `factorymind_pharma/`
- **Elapsed time**: ~16 seconds
- **Key outputs**: `data/processed/master_feature_table.parquet`, `data/outputs/evaluation_report.csv`, `data/outputs/test_predictions.parquet`

### Step 4: Phase 2 & Phase 3 — Residual Diagnostics, Model Ensemble & Policy Gates
Trains the full multi-model ensemble (LightGBM + XGBoost + CatBoost + Stacking Meta-Learner), runs QC Isolation Forest, machine maintenance hazard models, supplier risk scoring, and evaluates Phase 3 inventory safety-stock policies.
```powershell
# Working Directory: factorymind_pharma/
cd factorymind_pharma
python -X utf8 run_phase2.py
cd ..
```
- **Working directory**: `factorymind_pharma/`
- **Elapsed time**: ~4 minutes
- **Key outputs**: `factorymind_pharma/models/*.pkl`, `data/outputs/maintenance_risk_scores.parquet`, `data/outputs/supplier_risk_scores.parquet`, `data/outputs/qc_anomaly_scores.parquet`, `factorymind_pharma/docs/MODEL_UPGRADE_REPORT.md`

### Step 5: Generate Output CSVs & Phase 3 Re-measure
Computes the digital-twin re-measure report and multi-grain forecast evaluation tables.
```powershell
# Working Directory: project root
python -X utf8 factorymind_pharma/scripts/generate_output_csvs.py
```
- **Working directory**: project root (`factory-mind-ai`)
- **Elapsed time**: ~6 seconds
- **Key outputs**: `data/outputs/phase3_remeasure.csv`, `data/outputs/evaluation_report_phase2.csv`

### Step 6: Run Automated Test Suites
Execute the test suites across unit, property-based, simulation, and acceptance criteria:
```powershell
# Working Directory: project root
# 6a. Full Test Suite (Unit, Invariants, Closed-Loop, Property Tests)
python -X utf8 -m pytest factorymind_pharma/tests/ -v --tb=short

# 6b. Formal Acceptance Test Suite (Phases 4-6 Exit Criteria)
python -X utf8 -m pytest factorymind_pharma/tests/acceptance/ -v --tb=short
```
- **Working directory**: project root (`factory-mind-ai`)
- **Report generated**: `factorymind_pharma/docs/ACCEPTANCE_REPORT.md`

### Step 7: Launch Control Tower REST API (Service Layer)
Starts the FastAPI service exposing sensing, forecast, digital twin simulation, audit log verification, and decision approval endpoints.
```powershell
# Working Directory: project root
python -m uvicorn factorymind_pharma.services.tower_svc.api:app --host 127.0.0.1 --port 8000
```
- **Working directory**: project root (`factory-mind-ai`)
- **URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive OpenAPI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Step 8: Launch Enterprise Dashboards & AI Copilot
Starts the Plotly Dash application hosting all 10 operations dashboards and the integrated AI Copilot panel.
```powershell
# Working Directory: project root
python -X utf8 factorymind_pharma/dashboard/app.py
```
- **Working directory**: project root (`factory-mind-ai`)
- **URL**: [http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## 4. Phase-by-Phase Matrix

| Phase | Description | Key Artifacts Produced | Execution Command | Where to Inspect |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0** | Scaffold, config validation & foreign key checks | Validated in-memory schemas for 15 CSV datasets | `python -X utf8 run_pipeline.py` (in `factorymind_pharma/`) | Console log (`FK check passed`) |
| **Phase 1** | Foundation data layer & baseline forecast | `data/processed/master_feature_table.parquet`, `data/outputs/evaluation_report.csv` | `python -X utf8 run_pipeline.py` (in `factorymind_pharma/`) | [Executive Dashboard](http://127.0.0.1:8050/) & [Operational Dashboard](http://127.0.0.1:8050/operational) |
| **Phase 2** | Multi-model ensemble, QC anomalies, maintenance & supplier risk | `models/*.pkl`, `data/outputs/maintenance_risk_scores.parquet`, `supplier_risk_scores.parquet`, `qc_anomaly_scores.parquet` | `python -X utf8 run_phase2.py` (in `factorymind_pharma/`) | [Forecast Analytics](http://127.0.0.1:8050/forecast), [Machine Health](http://127.0.0.1:8050/machine-health), [Supplier & QC](http://127.0.0.1:8050/supplier-qc) |
| **Phase 3** | Inventory optimization & compliance policy gates | `data/outputs/phase3_remeasure.csv`, `evaluation_report_phase2.csv` | `python -X utf8 factorymind_pharma/scripts/generate_output_csvs.py` | [Inventory Opt](http://127.0.0.1:8050/inventory-opt), [Compliance](http://127.0.0.1:8050/compliance) |
| **Phase 4** | Vectorized stochastic digital twin & financial P&L | `data/outputs/fact_financial.parquet`, scenario Monte Carlo tensors | **On-demand**: via UI or `POST /v1/twin/simulate` | [Scenario Lab](http://127.0.0.1:8050/scenario-lab), [Financial Impact](http://127.0.0.1:8050/financial-impact) |
| **Phase 5** | AI Copilot (tool-use, grounding, prompt defense) | Grounded chat responses, provenance strips, tool traces | **On-demand**: floating button (✦) on all pages or `CopilotClient.ask()` | Floating drawer on [http://127.0.0.1:8050](http://127.0.0.1:8050/) |
| **Phase 6** | Governance, approval queue, replay clock & audit chain | Tamper-evident SHA-256 audit log, decision state transitions | **On-demand**: via UI or `POST /v1/replay/step`, `GET /v1/audit/verify` | [Control Tower](http://127.0.0.1:8050/control-tower), [API /v1/decisions](http://127.0.0.1:8000/v1/decisions) |

---

## 5. Models & Data Artifacts

### Machine Learning Models & Dashboard Binding

| Dashboard Page | URL | Artifacts Loaded | Description |
| :--- | :--- | :--- | :--- |
| **Executive Summary** | `/` | `data/processed/master_feature_table.parquet`, `data/outputs/test_predictions.parquet` | High-level executive revenue, volume, and service level KPIs |
| **Operational Telemetry** | `/operational` | `master_feature_table.parquet`, `05_inventory.csv` | Daily plant telemetry, actual vs forecast, inventory levels |
| **Forecast Analytics** | `/forecast` | `test_predictions.parquet`, `evaluation_report_phase2.csv` | Per-SKU ensemble predictions, confidence bands, WAPE & R² |
| **Machine Health** | `/machine-health` | `data/outputs/maintenance_risk_scores.parquet`, `09_machine_sensor_data.csv` | Machine failure hazard probabilities, sensor telemetry, wear heuristics |
| **Supplier & QC** | `/supplier-qc` | `data/outputs/supplier_risk_scores.parquet`, `data/outputs/qc_anomaly_scores.parquet` | Vendor risk scores, delivery unreliability, Isolation Forest QC scores |
| **Inventory Opt** | `/inventory-opt` | `05_inventory.csv`, `master_feature_table.parquet` | Reorder points, safety stock bands, stockout risk distribution |
| **Compliance** | `/compliance` | `14_compliance.csv`, `13_logistics.csv` | Audit trail flags, cold-chain temperature excursion alerts |
| **Financial Impact** | `/financial-impact`| `15_financials.csv` (or `fact_financial.parquet`) | Gross profit, COGS breakdown, loss-making SKU share |
| **Scenario Lab** | `/scenario-lab` | `configs/twin.yaml`, vectorized simulator in memory | Stochastic Monte Carlo 'what-if' simulation with CRN & mitigations |
| **Control Tower** | `/control-tower`| `sqlite/audit_chain.db`, `DecisionEngine`, `ReplayClock` | L5 Approval Queue (BLOCK, REVIEW, AUTO), tamper-evident audit badge |

### AI Copilot LLM Architecture
- **Provider**: Groq (`https://api.groq.com/openai/v1`)
- **Primary LLM**: `openai/gpt-oss-120b` (returns HTTP 200 on Groq API)
- **Automatic Fallback LLM**: `llama-3.3-70b-versatile` (configured as secondary model if primary encounters errors)
- **Guard Model Call Status**: In `copilot.yaml`, `guard_model: "meta-llama/llama-guard-4-12b"` is configured, but **no active code calls an external guard model API**. Prompt injection defense, write-intent protection, and out-of-scope refusals are executed entirely in-process via deterministic regex heuristics in `services/tower_svc/copilot/client.py` (`_check_security_and_intent()`).
- **Prompt Guard Notice**: `meta-llama/llama-prompt-guard-2-86m` is an injection classification model and is **not** used as a conversational copilot LLM.

### Identifying Copilot Operating Mode (Groq vs. Offline Keyword Mode)
Every response object includes `model_used` and a provenance strip:
- **Online Groq Mode**: `model_used` contains the active model identifier (e.g. `openai/gpt-oss-120b`). Tool execution traces list tool calls returned by the model.
- **Offline Keyword Mode**: When `GROQ_API_KEY` is not present, `model_used` is explicitly `"deterministic-engine-v1"`. The client deterministically matches intent keywords to tool functions, executes the tool, and composes a 100% numerically grounded response.

---

## 6. Smoke-Check Verification List

Run these requests while both servers are active:

### Control Tower REST API ([http://127.0.0.1:8000](http://127.0.0.1:8000))

| Method | Endpoint | Expected Result | Purpose |
| :--- | :--- | :--- | :--- |
| `GET` | `/healthz` | `HTTP 200` (`{"status": "healthy"}`) | Liveness probe |
| `GET` | `/readyz` | `HTTP 200` (`{"status": "ready"}`) | Readiness probe |
| `GET` | `/docs` | `HTTP 200` (HTML Swagger UI) | Interactive documentation |
| `GET` | `/openapi.json` | `HTTP 200` (Valid JSON schema) | OpenAPI specification |
| `GET` | `/v1/decisions` | `HTTP 200` (Array of decision records) | Control Tower approval queue |
| `GET` | `/v1/audit/verify` | `HTTP 200` (`{"is_valid": true}`) | SHA-256 cryptographic audit chain check |
| `GET` | `/v1/audit/entries` | `HTTP 200` (Array of audit log blocks) | Audit trail inspection |
| `GET` | `/v1/replay/status` | `HTTP 200` (`{"current_date": "..."}`) | Simulation replay clock status |
| `GET` | `/v1/models/registry` | `HTTP 200` (Champion/challenger metadata) | Model governance registry |
| `GET` | `/v1/sensing/machine/M001` | `HTTP 200` (Telemetry and risk) | Sensing service endpoint |
| `GET` | `/v1/forecast/P0020` | `HTTP 200` (Demand forecast & intervals) | Forecast service endpoint |
| `GET` | `/v1/policy/inventory/P0006` | `HTTP 200` (Inventory position & ROP) | Inventory policy endpoint |
| `POST` | `/v1/replay/step` | `HTTP 200` (Advances replay clock by 1 day) | Closed-loop clock step |
| `POST` | `/v1/twin/simulate` | `HTTP 200` (Scenario comparison result) | Stochastic digital twin simulation |

### Dash Enterprise Application ([http://127.0.0.1:8050](http://127.0.0.1:8050))

| Route | Expected Result | Key Elements Rendered |
| :--- | :--- | :--- |
| `/` | `HTTP 200` | Revenue KPI card, WAPE badge, Service level |
| `/operational` | `HTTP 200` | Inventory vs. actual demand chart, warehouse breakdown |
| `/forecast` | `HTTP 200` | SKU ensemble forecast curve, confidence bands |
| `/machine-health` | `HTTP 200` | Machine failure probability table, sensor trends |
| `/supplier-qc` | `HTTP 200` | Supplier risk matrix, QC anomaly scatter |
| `/inventory-opt` | `HTTP 200` | Safety stock bands, reorder point status |
| `/compliance` | `HTTP 200` | Cold-chain excursion alerts, batch audit logs |
| `/financial-impact`| `HTTP 200` | COGS breakdown, gross margin by category |
| `/scenario-lab` | `HTTP 200` | Monte Carlo shock builder form, mitigation ranking |
| `/control-tower` | `HTTP 200` | Approval queue (BLOCK, REVIEW, AUTO), SHA-256 audit badge |

### How to Check Logs for Errors & Tracebacks
- **FastAPI / Uvicorn**: Uvicorn logs request methods, paths, and status codes directly to stdout. Any unhandled exception triggers a standard Python `Traceback (most recent call last):` block with HTTP 500.
- **Dash Server**: The Dash Flask server prints HTTP requests and callback errors to stdout. For component-level callback errors, Dash renders an interactive error overlay when `debug=True`, or logs `Traceback` lines in the terminal when `debug=False`.

---

## 7. Known Caveats & Observed Results

The following observations reflect direct testing in this environment:

1. **Docker Compose Setup (`factorymind_pharma/docker-compose.yml`)**:
   - **Status**: Does not work as-is.
   - **Reason**: The Dockerfile copies `requirements.txt` from `factorymind_pharma/requirements.txt` assuming the build context is the repository root (`factory-mind-ai/`), but `docker-compose.yml` sets `context: .` pointing to `factorymind_pharma/`. Furthermore, volume mounts refer to `./factorymind_pharma/models` from inside `factorymind_pharma`, which resolves to a non-existent path `./factorymind_pharma/factorymind_pharma/models`. Port mapping also maps 8501:8501 while Dash is configured for port 8050.
   - **Supported Route**: Run directly using the local virtual environment (`venv`) as detailed in Section 3.

2. **Automated Acceptance Test Result (1 Failed / 18 Passed)**:
   - **Test**: `factorymind_pharma/tests/acceptance/test_phase4_6_acceptance.py::TestPhase4::test_paired_common_random_numbers` (Line 116).
   - **Exact Error**:
     ```text
     AssertionError: ACCEPTANCE FAIL [Phase 4] Paired common random numbers: target=baseline ≠ shocked (same seed), measured=baseline_profit=493337120, shocked_profit=493337120
     ```
   - **Root Cause**: In `services/twin_svc/simulator.py`, the initial stock across warehouses is high enough that introducing a 3-day supplier delay on supplier `S001` over a 30-day window does not cause an immediate stockout or change total units fulfilled. Consequently, simulated expected profit remains identical (`493,337,120`), causing the assertion `b_profit != s_profit` to fail. All other 18 acceptance tests pass.

3. **Overall Pytest Suite Result**:
   - **Total Tests Executed**: 98 tests.
   - **Measured Count**: **97 passed, 1 failed** (the acceptance test above).

4. **Groq Model HTTP Status**:
   - `openai/gpt-oss-120b`: Returns `HTTP 200`.
   - `llama-3.3-70b-versatile`: Returns `HTTP 404` on standard endpoint (model name was superseded or deprecated on Groq).
   - `openai` python package: Not present in `requirements.txt` (which only specifies `groq>=0.4.0`). Because `client.py` attempts `from openai import OpenAI`, the copilot automatically falls back to offline deterministic mode (`deterministic-engine-v1`) when live SDK calls are unavailable.

5. **Execution Timings (Benchmarked)**:
   - Phase 0 & Phase 1 (`run_pipeline.py`): ~15.9 seconds.
   - Phase 2 & Phase 3 (`run_phase2.py`): ~4 minutes 7 seconds (dominated by training 100 SKU Holt-Winters and ensemble models).
   - Output CSV Generation (`generate_output_csvs.py`): ~6.2 seconds.
   - Pytest Full Run (`pytest`): ~1 minute 14 seconds.
