"""
FactoryMind AI (Pharma) - FastAPI Service Endpoints (§12.1)
Exposes /v1 APIs for sensing_svc, forecast_svc, twin_svc, policy_svc, and tower_svc
with shared Pydantic contracts, RBAC enforcement, and audit verification.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from factorymind_pharma.contracts.decision import (
    DecisionRecord,
    ResolveDecisionRequest,
    RuleTier,
    DecisionStatus,
    Role,
)
from factorymind_pharma.contracts.scenario import Scenario, ScenarioComparisonResult
from factorymind_pharma.services.tower_svc.decision_engine import DecisionEngine
from factorymind_pharma.services.tower_svc.audit_log import AuditLog, AuditVerificationResult
from factorymind_pharma.services.tower_svc.replay_clock import ReplayClock, FeedbackWriter, ModelLifecycleManager
from factorymind_pharma.services.tower_svc.copilot.tools import (
    get_kpi_summary, get_forecast, get_inventory_position,
    get_machine_risk, get_supplier_risk, get_financials
)

# App instance
app = FastAPI(
    title="FactoryMind Pharma - Service Layer API",
    version="1.0.0",
    description="Operational Service Boundaries for Sensing, Forecast, Digital Twin, Policy, and Control Tower.",
)

# In-memory singletons
engine = DecisionEngine()
audit_log = engine.audit
replay_clock = ReplayClock()
feedback_writer = FeedbackWriter()
lifecycle_mgr = ModelLifecycleManager(audit_log=audit_log)


# -------------------------------------------------------------
# Health and Readiness Probes (§12.1)
# -------------------------------------------------------------
@app.get("/healthz", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "factorymind-fullstack", "timestamp": "2026-10-11T15:00:00"}

@app.get("/readyz", tags=["Health"])
def ready_check():
    return {"status": "ready", "audit_chain": "connected", "data_loaded": True}


# -------------------------------------------------------------
# Tower Service: Decisions & Approval Workflow (§10, §12.1)
# -------------------------------------------------------------
@app.get("/v1/decisions", response_model=List[DecisionRecord], tags=["Control Tower"])
def list_decisions(status: Optional[str] = Query(None), tier: Optional[str] = Query(None)):
    """List operational decisions in the approval queue."""
    return engine.list_decisions(status=status, tier=tier)

@app.get("/v1/decisions/{decision_id}", response_model=DecisionRecord, tags=["Control Tower"])
def get_decision(decision_id: str):
    dec = engine.get_decision(decision_id)
    if not dec:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
    return dec

@app.post("/v1/decisions/{decision_id}/resolve", response_model=DecisionRecord, tags=["Control Tower"])
def resolve_decision(decision_id: str, req: ResolveDecisionRequest):
    """
    Resolve a decision with role-based access control.
    Planner attempting to resolve a QA-gated or BLOCK decision returns HTTP 403 Forbidden (§10.4).
    """
    try:
        updated = engine.resolve_decision(
            decision_id=decision_id,
            action=req.action,
            resolution_note=req.resolution_note,
            role=req.role,
            username=req.username,
        )
        return updated
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


# -------------------------------------------------------------
# Tower Service: Tamper-Evident Audit Chain (§10.5)
# -------------------------------------------------------------
@app.get("/v1/audit/verify", response_model=AuditVerificationResult, tags=["Audit Chain"])
def verify_audit_chain():
    """Recomputes full SHA-256 hash chain from genesis and reports any tampering."""
    return audit_log.verify_chain()

@app.get("/v1/audit/entries", tags=["Audit Chain"])
def get_audit_entries(limit: int = 50):
    return [e.model_dump() for e in audit_log.get_entries(limit=limit)]


# -------------------------------------------------------------
# Tower Service: Closed Loop Replay & Model Lifecycle (§11)
# -------------------------------------------------------------
@app.get("/v1/replay/status", tags=["Replay Clock"])
def get_replay_status():
    return replay_clock.get_status()

@app.post("/v1/replay/step", tags=["Replay Clock"])
def step_replay_clock():
    new_date = replay_clock.step_day()
    eval_metrics = feedback_writer.evaluate_day(new_date)
    return {"current_date": new_date, "daily_eval": eval_metrics}

@app.get("/v1/models/registry", tags=["Model Lifecycle"])
def get_model_registry():
    return lifecycle_mgr.load_registry()

@app.post("/v1/models/promote", tags=["Model Lifecycle"])
def promote_challenger(challenger_id: str = "ensemble-challenger-v2"):
    promoted, msg = lifecycle_mgr.evaluate_and_promote_challenger(challenger_id=challenger_id)
    if not promoted:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


# -------------------------------------------------------------
# Sensing, Forecast, Twin, Policy Read APIs (§12.1)
# -------------------------------------------------------------
@app.get("/v1/sensing/machine/{machine_id}", tags=["Sensing Service"])
def get_sensing_telemetry(machine_id: str):
    res = get_machine_risk(machine_id)
    return res.model_dump()

@app.get("/v1/forecast/{product_id}", tags=["Forecast Service"])
def get_forecast_api(product_id: str, horizon: int = 30, grain: str = "daily"):
    res = get_forecast(product_id, horizon=horizon, grain=grain)
    return res.model_dump()

@app.get("/v1/policy/inventory/{product_id}", tags=["Policy Service"])
def get_inventory_api(product_id: str, warehouse_id: Optional[str] = None):
    res = get_inventory_position(product_id, warehouse_id=warehouse_id)
    return res.model_dump()

@app.post("/v1/twin/simulate", response_model=ScenarioComparisonResult, tags=["Twin Service"])
def simulate_twin_scenario(scenario: Scenario):
    from factorymind_pharma.services.twin_svc.simulator import DigitalTwinSimulator
    sim = DigitalTwinSimulator()
    res = sim.run_scenario(scenario)
    return res
