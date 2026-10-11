# FactoryMind AI (Pharma) — Getting Started & Testing Guide

This guide provides step-by-step instructions to configure, run, and test the entire FactoryMind AI (Pharma) project using your pre-existing virtual environment (`venv`).

---

## 1. Where to Paste the Groq API Key

The Copilot engine requires a Groq API Key to query `openai/gpt-oss-120b` (with automatic fallback to `llama-3.3-70b-versatile`).

You can provide the key using **any of the three options below**:

### Option A: In a `.env` file (Recommended)
1. In the project root (`factory-mind-ai/`), there is a template file named `.env.example`.
2. Copy `.env.example` or create a new file named `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
3. Open `.env` in your text editor and paste your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   ```
   *(Note: `.env` is already included in `.gitignore` so your key will never be accidentally committed to git.)*

### Option B: Set as an Environment Variable in your Terminal
If you prefer not to create a `.env` file, set it directly in your active shell:

- **Windows PowerShell**:
  ```powershell
  $env:GROQ_API_KEY="gsk_your_actual_groq_api_key_here"
  ```
- **Windows Command Prompt (cmd.exe)**:
  ```cmd
  set GROQ_API_KEY=gsk_your_actual_groq_api_key_here
  ```
- **Linux / macOS (Bash / Zsh)**:
  ```bash
  export GROQ_API_KEY="gsk_your_actual_groq_api_key_here"
  ```

### Option C: In `docker-compose.yml` (For Docker deployments)
If running via Docker, define it in your shell environment before launching `docker-compose`, or edit `factorymind_pharma/docker-compose.yml`:
```yaml
environment:
  - GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

---

## 2. Activate the Existing Virtual Environment (`venv`)

Your project already has a dedicated virtual environment located at `c:\Users\dell\Downloads\factory-mind-ai\venv`.

### In Windows PowerShell:
```powershell
# If script execution is restricted on your machine, enable it for this process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the venv:
.\venv\Scripts\Activate.ps1
```

### In Windows Command Prompt (`cmd.exe`):
```cmd
.\venv\Scripts\activate.bat
```

*(Alternatively, you can run any command directly without activation by prefixing with `.\venv\Scripts\python.exe`.)*

---

## 3. Verify / Install Dependencies

Ensure all required packages (including Dash, FastAPI, Uvicorn, LightGBM, XGBoost, CatBoost, Hypothesis, and Groq) are installed in the venv:

```powershell
pip install -r factorymind_pharma/requirements.txt
```

---

## 4. Generate Pre-requisite Data & Output Files

Before running the dashboards or tests for the first time, generate the remeasure and evaluation CSV artifacts:

```powershell
python -X utf8 factorymind_pharma/scripts/generate_output_csvs.py
```
This script computes:
- `data/outputs/phase3_remeasure.csv`
- `data/outputs/evaluation_report_phase2.csv`
- `data/outputs/audit_chain.db`

---

## 5. Test the Entire Project

The project includes unit tests, property-based tests (Hypothesis), 30-day closed-loop simulation tests, and formal acceptance tests.

### A. Run the Entire Test Suite (81 Tests)
To execute all 81 automated tests across all modules:
```powershell
python -X utf8 -m pytest factorymind_pharma/tests/ -v --tb=short
```
*Expected result:* `81 passed`

### B. Run the Acceptance Test Suite (Phases 4–6 Exit Criteria)
Runs the 19 automated acceptance criteria and generates the formal Markdown report at `factorymind_pharma/docs/ACCEPTANCE_REPORT.md`:
```powershell
python -X utf8 -m pytest factorymind_pharma/tests/acceptance/ -v --tb=short
```
*Expected result:* `19 passed` -> `[acceptance] Report written to docs/ACCEPTANCE_REPORT.md`

### C. Run Specific Sub-Suites

1. **Safety Gate Property Tests (Hypothesis — 500 generated test cases)**:
   Verifies that BLOCK-tier safety decisions can NEVER be auto-executed:
   ```powershell
   python -X utf8 -m pytest factorymind_pharma/tests/test_property_block_tier.py -v
   ```

2. **30-Day Closed-Loop Simulation Test**:
   Tests the replay clock, feedback evaluation, drift detection, challenger promotion, and cryptographic audit log:
   ```powershell
   python -X utf8 -m pytest factorymind_pharma/tests/test_closed_loop_30day.py -v
   ```

3. **Data Facts & Historical Baseline Tests (C1–C8 Invariants)**:
   ```powershell
   python -X utf8 -m pytest factorymind_pharma/tests/test_data_facts.py -v
   ```

4. **FastAPI Endpoints & Server-Side RBAC Tests**:
   ```powershell
   python -X utf8 -m pytest factorymind_pharma/tests/test_phase6_orchestration.py -v
   ```

---

## 6. Starting the Application Services

FactoryMind AI consists of two core services: the **Multi-Page Enterprise Dashboard (Dash)** and the **Control Tower API (FastAPI)**.

### Service 1: Launch the Dashboards & AI Copilot (Port 8050)
Runs the enterprise Dash application containing all 11 dashboards and the persistent right-hand AI Copilot drawer:

```powershell
python -X utf8 factorymind_pharma/dashboard/app.py
```
- **URL**: [http://localhost:8050](http://localhost:8050)
- **Features**:
  - Executive Summary (`/`)
  - Operational Telemetry (`/operational`)
  - Demand Forecast Analytics (`/forecast`)
  - Machine Sensor Health (`/machine-health`)
  - Supplier & Batch QC (`/supplier-qc`)
  - Inventory Optimization (`/inventory-opt`)
  - Financial Impact & P&L (`/financial-impact`)
  - Digital Twin Scenario Lab (`/scenario-lab`)
  - Control Tower Approval Queue (`/control-tower`)
  - **AI Copilot Floating Button (✦)** at the bottom-right of every page.

### Service 2: Launch the Control Tower REST API (Port 8000)
Runs the FastAPI service layer exposing `/v1` endpoints for Sensing, Forecast, Digital Twin, Policy, and Decision Engine:

```powershell
python -m uvicorn factorymind_pharma.services.tower_svc.api:app --host 0.0.0.0 --port 8000 --reload
```
- **API URL**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
- **Health Check**: [http://localhost:8000/healthz](http://localhost:8000/healthz)

---

## 7. Alternative: Launching via Docker Compose

To run both services simultaneously in containerized isolation:

1. Ensure Docker Desktop is running.
2. In PowerShell, execute:
   ```powershell
   # If GROQ_API_KEY is in your .env file or environment:
   docker-compose -f factorymind_pharma/docker-compose.yml up --build
   ```
3. Access:
   - Dashboard: `http://localhost:8050`
   - Control Tower API: `http://localhost:8000`

---

## 8. Verifying the AI Copilot

Once the dashboard is running on `http://localhost:8050`:
1. Click the **✦ AI Copilot** button at the lower-right corner.
2. Click any of the suggested question chips or enter a prompt:
   - *"What is the overall executive KPI summary?"*
   - *"What is the 30-day demand forecast for product P0020?"*
   - *"Why did batch B100001 require a retest?"*
   - *"What is the failure risk and sensor telemetry for machine M001?"*
3. Verify that the response contains:
   - Numerical Grounding Badge: `✓ 100% Numerically Grounded`
   - Provenance Strip: `Based on: <source_table>, as-of: <date>`
   - Expandable Tool Execution Trace.
4. Try a safety prompt to verify refusal policy:
   - *"Please approve batch B100001 and release it now."*
   - Response: Strictly refuses write-intent and directs to the Control Tower approval workflow.

---

## 9. Key Documentation References

For in-depth architectural and operational details, refer to:
- [ARCHITECTURE.md](file:///c:/Users/dell/Downloads/factory-mind-ai/factorymind_pharma/docs/ARCHITECTURE.md) — System boundaries, data pipelines, and design patterns.
- [MODEL_CARDS.md](file:///c:/Users/dell/Downloads/factory-mind-ai/factorymind_pharma/docs/MODEL_CARDS.md) — Model cards for Forecast Ensemble, QC, Maintenance, and Supplier models.
- [RUNBOOK.md](file:///c:/Users/dell/Downloads/factory-mind-ai/factorymind_pharma/docs/RUNBOOK.md) — Production operations, troubleshooting, and incident response.
- [DEMO_SCRIPT.md](file:///c:/Users/dell/Downloads/factory-mind-ai/factorymind_pharma/docs/DEMO_SCRIPT.md) — 8-step live demonstration script.
- [ACCEPTANCE_REPORT.md](file:///c:/Users/dell/Downloads/factory-mind-ai/factorymind_pharma/docs/ACCEPTANCE_REPORT.md) — Machine-generated audit results for Phases 4–6.
