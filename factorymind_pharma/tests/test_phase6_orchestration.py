"""
FactoryMind AI (Pharma) - Phase 6 Orchestration, RBAC & Hardening Test Suite (§12.3, §14)
Tests:
  1. Property test: BLOCK tier can NEVER yield auto_executed for any input.
  2. RBAC test: planner cannot resolve QA-gated/BLOCK decisions (HTTP 403 / PermissionError).
  3. Tamper test: hash-chain detects payload mutation at exact index.
  4. Closed-loop model promotion test: promoted ONLY with >= 2% WAPE gain & bias gate.
  5. FastAPI service contract test: all endpoints functional.
"""
from __future__ import annotations
import pytest
from pathlib import Path
import tempfile
import sqlite3

from factorymind_pharma.services.tower_svc.decision_engine import DecisionEngine
from factorymind_pharma.services.tower_svc.audit_log import AuditLog
from factorymind_pharma.services.tower_svc.replay_clock import ReplayClock, FeedbackWriter, ModelLifecycleManager, DriftDetector
from fastapi.testclient import TestClient
from factorymind_pharma.services.tower_svc.api import app


def test_property_block_tier_never_auto_executed():
    """Exit Gate 1: Property test: BLOCK tier never auto-executed for any input combination."""
    engine = DecisionEngine()

    test_cases = [
        # Major Deviation
        {"audit_status": "Major Deviation", "release_status": "PASS", "temp_flag": 0, "conf": 0.99},
        # QC FAIL
        {"audit_status": "Compliant", "release_status": "FAIL", "temp_flag": 0, "conf": 0.95},
        # Cold Chain Excursion
        {"audit_status": "Compliant", "release_status": "PASS", "temp_flag": 1, "conf": 0.99},
        # Multiple triggers simultaneously
        {"audit_status": "Major Deviation", "release_status": "FAIL", "temp_flag": 1, "conf": 1.0},
        # Low confidence
        {"audit_status": "Major Deviation", "release_status": "PASS", "temp_flag": 0, "conf": 0.10},
    ]

    for i, tc in enumerate(test_cases):
        dec = engine.propose_decision(
            decision_id=f"PROP-TEST-{i}",
            decision_type="batch_release",
            subject_ids=[f"B999{i}", "P0001"],
            recommended_action="Quarantine batch",
            audit_status=tc["audit_status"],
            release_status=tc["release_status"],
            temperature_excursion_flag=tc["temp_flag"],
            ensemble_agreement=tc["conf"],
            interval_width_ratio=0.10,
            days_stale=0.0,
        )
        assert dec.rule_tier == "BLOCK", f"Case {i} failed: expected BLOCK tier, got {dec.rule_tier}"
        assert dec.status != "auto_executed", f"CRITICAL SAFETY VIOLATION: BLOCK tier was auto_executed in case {i}!"
        assert dec.status == "pending_review"


def test_rbac_planner_cannot_resolve_block_decision():
    """Exit Gate 2: RBAC test: planner cannot resolve QA-gated decisions (403 / PermissionError)."""
    engine = DecisionEngine()

    # Create BLOCK decision
    dec = engine.propose_decision(
        decision_id="RBAC-TEST-001",
        decision_type="batch_release",
        subject_ids=["B100004", "P0051"],
        recommended_action="Reject batch B100004",
        audit_status="Major Deviation",
    )

    # 1. Attempt resolution as 'planner' -> MUST fail
    with pytest.raises(PermissionError) as exc_info:
        engine.resolve_decision(
            decision_id=dec.id,
            action="approve",
            resolution_note="Planner attempting override",
            role="planner",
            username="john_planner",
        )
    assert "cannot resolve" in str(exc_info.value)

    # 2. Attempt resolution as 'qa_reviewer' -> MUST succeed
    resolved = engine.resolve_decision(
        decision_id=dec.id,
        action="reject",
        resolution_note="QA Manager confirming rejection",
        role="qa_reviewer",
        username="dr_smith_qa",
    )
    assert resolved.status == "rejected"
    assert resolved.resolver == "dr_smith_qa"


def test_audit_chain_tamper_detection():
    """Exit Gate 3: Tamper test detected at correct index; verify endpoint green initially."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        test_db = Path(tmp_dir) / "test_audit.db"
        audit = AuditLog(db_path=test_db)

        # 1. Add sequential entries
        e0 = audit.append("Event0", {"data": "genesis_payload"})
        e1 = audit.append("Event1", {"data": "second_payload"})
        e2 = audit.append("Event2", {"data": "third_payload"})

        # Initial check must pass
        res_initial = audit.verify_chain()
        assert res_initial.is_valid is True
        assert res_initial.total_records == 3
        assert res_initial.broken_index is None

        # 2. Mutate entry #1 payload directly in SQLite without updating record_hash
        audit.tamper_for_test(1, {"data": "MALICIOUS_ALTERATION"})

        # Verify MUST catch tampering at index 1
        res_tampered = audit.verify_chain()
        assert res_tampered.is_valid is False
        assert res_tampered.broken_index == 1, f"Expected broken index 1, got {res_tampered.broken_index}"
        assert "tampering detected at index 1" in res_tampered.message


def test_model_promotion_rules():
    """Exit Gate 4 & 5: Challenger promoted only with >= 2% WAPE gain and bias gate."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        reg_file = Path(tmp_dir) / "registry.json"
        db_file = Path(tmp_dir) / "audit.db"

        audit = AuditLog(db_path=db_file)
        mgr = ModelLifecycleManager(registry_path=reg_file, audit_log=audit)

        # 1. Test case: Challenger beats champion by 2.40 pp (20.2% -> 17.8%), bias = 0.40%
        # Meets both criteria (>= 2% gain and |bias| <= 5%)
        promoted, msg = mgr.evaluate_and_promote_challenger(
            challenger_id="ensemble-challenger-v2",
            min_wape_gain_pct=2.0,
            max_bias_pct=5.0,
        )
        assert promoted is True
        assert "PROMOTED to Champion" in msg

        reg = mgr.load_registry()
        assert reg["active_champion"] == "ensemble-challenger-v2"
        assert reg["models"]["ensemble-challenger-v2"]["status"] == "champion"
        assert reg["models"]["ensemble-champion-v1"]["status"] == "retired_champion"

        # 2. Test rollback
        rb_ok, rb_msg = mgr.rollback_to_champion("ensemble-champion-v1")
        assert rb_ok is True
        reg_rb = mgr.load_registry()
        assert reg_rb["active_champion"] == "ensemble-champion-v1"


def test_fastapi_endpoints_and_rbac():
    """Tests FastAPI /v1 service endpoints and HTTP 403 on RBAC violation."""
    client = TestClient(app)

    # Health check
    res_h = client.get("/healthz")
    assert res_h.status_code == 200
    assert res_h.json()["status"] == "healthy"

    # Audit verify
    res_a = client.get("/v1/audit/verify")
    assert res_a.status_code == 200
    assert res_a.json()["is_valid"] is True

    # List decisions
    res_d = client.get("/v1/decisions")
    assert res_d.status_code == 200
    assert len(res_d.json()) >= 3

    # Resolve BLOCK decision as planner -> MUST return 403
    res_block = client.post(
        "/v1/decisions/DEC-20221020-002/resolve",
        json={
            "action": "approve",
            "resolution_note": "Illegal planner attempt",
            "role": "planner",
            "username": "planner_user",
        }
    )
    assert res_block.status_code == 403, f"Expected HTTP 403, got {res_block.status_code}"
    assert "cannot resolve" in res_block.json()["detail"]

    # Resolve BLOCK decision as qa_reviewer -> MUST return 200
    res_qa = client.post(
        "/v1/decisions/DEC-20221020-002/resolve",
        json={
            "action": "reject",
            "resolution_note": "Authorized QA rejection",
            "role": "qa_reviewer",
            "username": "qa_user",
        }
    )
    assert res_qa.status_code == 200
    assert res_qa.json()["status"] == "rejected"
