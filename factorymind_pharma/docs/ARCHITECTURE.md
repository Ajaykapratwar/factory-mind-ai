# FactoryMind AI (Pharma) — Architecture

> Phase 6 handover artifact · §13.2

---

## 1. System Overview

FactoryMind AI is a five-layer decision-support platform for pharmaceutical manufacturing. It ingests raw operational data, produces forecasts and risk scores, simulates alternative futures with a digital twin, enforces policy gates, and surfaces every insight through dashboards and a conversational AI copilot. **No component makes clinical, regulatory, or release decisions;** all outputs are labelled decision support.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  L5 CONTROL TOWER  (tower_svc)                                         │
│  DecisionEngine · AuditLog · ReplayClock · Copilot · EventBus          │
├───────────────────────────┬─────────────────────────────────────────────┤
│  L4 POLICY (policy_svc)   │  L3 FORECAST (forecast_svc)                 │
│  InventoryOptimizer        │  Stacked Ensemble (LGBM+XGB+CB+Meta)       │
│  ComplianceGates           │  FeedbackWriter · DriftDetector             │
├───────────────────────────┴─────────────────────────────────────────────┤
│  L2 DIGITAL TWIN (twin_svc)                                             │
│  DigitalTwinSimulator · FinancialIntelligence · TwinValidator           │
├─────────────────────────────────────────────────────────────────────────┤
│  L1 SENSING (sensing_svc)                                               │
│  QC Anomaly · Maintenance Classifier · Supplier Risk Scorer             │
├─────────────────────────────────────────────────────────────────────────┤
│  DATA LAKE  (15 CSV sources → master_feature_table.parquet)            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer Detail

### L1 — Sensing (`services/sensing_svc`, `qc_svc`, `maintenance_svc`, `supplier_svc`)

| Component | Model | Output |
|---|---|---|
| QC Anomaly Detector | Isolation Forest (scikit-learn) | `anomaly_score ∈ [-1,+1]`, empirical z-scores per QC field |
| Maintenance Classifier | Random Forest on sensor telemetry | `failure_prob ∈ [0,1]`, risk tier LOW/MEDIUM/HIGH/CRITICAL |
| Supplier Risk Scorer | Rule-based composite (delivery reliability, rejection rate) | `risk_score ∈ [0,1]`, tier |

All models are serialised to `models/` as `.pkl` files; versions tracked in `models/registry.json`.

### L2 — Digital Twin (`services/twin_svc`)

A **stochastic, vectorised NumPy simulator** with state tensor `[n_paths × n_skus × n_warehouses]`.

| File | Purpose |
|---|---|
| `simulator.py` | Monte Carlo daily-step engine; runs 500 paths × 90 days in ≤ 20 s |
| `distributions.py` | Empirical fits for demand residuals, yield, machine availability, QC rates, cold-chain |
| `financial.py` | Prices simulation outcomes; builds `fact_financial`; computes modelled cost estimates |
| `validator.py` | Distributional back-test: P10–P90 interval coverage, yield drift, downtime |

Topology is **config-based** (see `configs/twin.yaml`); never inferred from `production_line` / `machine_id` columns (C4). Every parameter is tagged `source: assumed | fitted`.

### L3 — Forecast (`services/forecast_svc`)

Stacked ensemble:
- **Base learners:** LightGBM, XGBoost, CatBoost (trained with lag-1/7/14/28 features + disease index + seasonality)
- **Meta-learner:** Ridge regression on out-of-fold base predictions
- **Evaluation:** WAPE (primary), RMSE, Bias %, R² — reported at row, product-day, and product-week grain (C7)
- **Segments:** Per-product lag-autocorrelation bucketing for residual bootstrap in the twin

### L4 — Policy (`services/policy_svc`)

| Component | Logic |
|---|---|
| `InventoryOptimizer` | Reorder-point and safety-stock policy; simulated vs. baseline on twin |
| `ComplianceGates` | Phase 3 release gates for cold-chain, compliance audit status |

Rule tiers (deterministic, no ML) are in `configs/policy.yaml`.

### L5 — Control Tower (`services/tower_svc`)

| Component | Responsibility |
|---|---|
| `DecisionEngine` | Creates `DecisionRecord`s, enforces BLOCK/REVIEW/AUTO-ELIGIBLE tiers, RBAC |
| `AuditLog` | SHA-256 hash-chained append-only log; `/v1/audit/verify` detects tampering |
| `ReplayClock` | Advances simulated date over held-out window; drives feedback + drift detection |
| `FeedbackWriter` | Joins forecasts to actuals; maintains rolling WAPE, bias % per SKU |
| `ModelLifecycleManager` | Champion/challenger promotion (≥ 2 % WAPE gain + bias gate) |
| `EventBus` | In-process `asyncio` pub/sub; handlers are idempotent |
| `Copilot (CopilotClient)` | Tool-calling LLM interface (Groq); zero ungrounded numbers enforced in code |

---

## 3. Data Flow

```
Raw CSVs (15 tables)
      │
      ▼
  master_feature_table.parquet   ◄── Phase 0/1 pipeline
      │
  ┌───┴──────────────────────────────────────────────────┐
  │  L1 Sensing: anomaly scores, risk scores, failure prob│
  └───────────────────────────────┬──────────────────────┘
                                  │
  ┌───────────────────────────────▼──────────────────────┐
  │  L3 Forecast: demand point + interval per SKU/day    │
  └───────────────────────────────┬──────────────────────┘
                                  │
  ┌───────────────────────────────▼──────────────────────┐
  │  L2 Twin: stochastic simulation of outcomes + P&L    │
  └───────────────────────────────┬──────────────────────┘
                                  │
  ┌───────────────────────────────▼──────────────────────┐
  │  L4 Policy: reorder decisions, compliance gates      │
  └───────────────────────────────┬──────────────────────┘
                                  │
  ┌───────────────────────────────▼──────────────────────┐
  │  L5 Tower: decisions → queue → resolve → audit log   │
  │  Copilot: tool call → grounding check → answer       │
  └──────────────────────────────────────────────────────┘
                                  │
                        Dashboards 1–11
```

---

## 4. Event Flow (`EventBus`)

```
ForecastUpdated  ──► Tower: create reorder DecisionRecord
QCAnomalyFlagged ──► Tower: create QC review DecisionRecord (BLOCK if FAIL)
MachineRiskHigh  ──► Tower: create maintenance DecisionRecord
SupplierRiskChanged ► Tower: create supplier review DecisionRecord
ScenarioCompleted ──► Dashboard 10: refresh results
DecisionCreated  ──► AuditLog: append entry
DecisionResolved ──► AuditLog: append entry; EventBus: downstream handlers
DriftDetected    ──► Tower: trigger challenger train; AuditLog entry
ModelPromoted    ──► AuditLog entry; registry.json updated
```

---

## 5. Service Contracts

Each service exposes a FastAPI app under `/v1`. All request/response models live in `contracts/`:

| Contract file | Models |
|---|---|
| `contracts/decision.py` | `DecisionRecord`, `ResolveDecisionRequest`, `RuleTier`, `Role` |
| `contracts/scenario.py` | `Scenario`, `ScenarioComparisonResult`, all shock types |

`contracts/openapi.json` — committed OpenAPI 3.1 spec, generated from the running app.

Services communicate **only** through these contracts or the event bus. No service reads another service's storage directly.

---

## 6. Dashboard Map

| # | Page file | Layer |
|---|---|---|
| 1 | `executive.py` | L3/L4 |
| 2 | `forecast_analytics.py` | L3 |
| 3 | `inventory_optimization.py` | L4 |
| 4 | `operational.py` | L1/L4 |
| 5 | `machine_health.py` | L1 |
| 6 | `financial_impact.py` | L2/L4 |
| 7 | `supplier_qc.py` | L1 |
| 8–9 | (Phase 1–3 pages) | L1 |
| 10 | `scenario_lab.py` | L2 |
| 11 | `control_tower.py` | L5 |

Copilot panel: right-hand collapsible panel on every page. Single token file: `dashboard/assets/tokens.css`. No hardcoded hex.

---

## 7. Security

- **Auth:** Local accounts, hashed passwords, signed session tokens. Role checks enforced server-side.
- **RBAC:** `planner`, `qa_reviewer`, `admin`. Only `qa_reviewer` can resolve BLOCK/QC-driven decisions.
- **Audit:** SHA-256 hash-chained log; tampering detected by `/v1/audit/verify`.
- **Copilot:** Prompt injection detection, write-intent refusal, numeric grounding enforcement in code.

> This is an engineering integrity control. It is **not** described as regulatory compliance (e.g., 21 CFR Part 11).

---

## 8. Infrastructure

- `docker-compose.yml` brings up all services + dashboard with one command.
- SQLite for decisions and audit log (Postgres swap is a documented option, not built).
- Structured JSON logging with correlation IDs.
- Config via environment variables; secrets never in source.
- Pinned `requirements.txt`; CI runs lint, tests, metric gate, data-facts tests, copilot eval (cached).
