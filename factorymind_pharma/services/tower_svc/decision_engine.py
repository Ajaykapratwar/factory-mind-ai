"""
FactoryMind AI (Pharma) - Decision Engine & Approval Workflow (§10.1–§10.4)
Enforces:
  - Deterministic rule tiers (BLOCK, REVIEW, AUTO-ELIGIBLE) per C1.
  - Multi-factor confidence score [0, 1].
  - Strict auto-execution gate (AUTO-ELIGIBLE and confidence >= 0.80).
  - Server-side RBAC: planner cannot resolve QA-gated decisions (HTTP 403 / PermissionError).
  - Cryptographic hash-chain audit logging.
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal, Tuple
from pathlib import Path
import yaml
from pydantic import BaseModel

from factorymind_pharma.contracts.decision import (
    DecisionRecord,
    RuleTier,
    DecisionStatus,
    DecisionType,
    Role,
)
from factorymind_pharma.services.tower_svc.audit_log import AuditLog

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "policy.yaml"


class DecisionEngine:
    def __init__(self, config_path: Optional[Path] = None, audit_log: Optional[AuditLog] = None):
        self.config_path = config_path or CONFIG_PATH
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "rule_tiers": {
                    "auto_eligible": {"min_confidence": 0.80},
                },
                "confidence": {
                    "weights": {
                        "ensemble_agreement": 0.45,
                        "interval_precision": 0.35,
                        "data_freshness": 0.20,
                    }
                }
            }

        self.audit = audit_log or AuditLog()
        self._decisions: Dict[str, DecisionRecord] = {}
        self._init_storage()

    def _init_storage(self):
        """Initializes decision storage and seeds standard demonstration queue."""
        # Prepopulate demo decisions
        demo_items = [
            {
                "id": "DEC-20221020-001",
                "type": "batch_release",
                "subject_ids": ["B100001", "P0039"],
                "recommended_action": "Hold batch B100001 in quarantine for chemical re-assay",
                "rule_tier": "REVIEW",
                "confidence": 0.92,
                "rationale": "QC release_status is RETEST with impurity spike (1.38%). Requires QA Review.",
                "status": "pending_review",
            },
            {
                "id": "DEC-20221020-002",
                "type": "batch_release",
                "subject_ids": ["B100004", "P0051"],
                "recommended_action": "Reject batch B100004 due to Major Deviation in sterilization audit",
                "rule_tier": "BLOCK",
                "confidence": 0.99,
                "rationale": "Major Deviation flagged in audit_status. Mandatory BLOCK tier; auto-execution forbidden.",
                "status": "pending_review",
            },
            {
                "id": "DEC-20221020-003",
                "type": "restock",
                "subject_ids": ["P0006", "W001"],
                "recommended_action": "Dispatch expedited replenishment order for 800 units to avoid stockout",
                "rule_tier": "AUTO-ELIGIBLE",
                "confidence": 0.88,
                "rationale": "Stock below safety threshold. All quality and supplier gates clear; confidence >= 0.80.",
                "status": "auto_executed",
            },
            {
                "id": "DEC-20221020-004",
                "type": "maintenance_pull_forward",
                "subject_ids": ["M001"],
                "recommended_action": "Schedule proactive 12h maintenance during non-peak shift",
                "rule_tier": "AUTO-ELIGIBLE",
                "confidence": 0.85,
                "rationale": "Sensor vibration trend elevated (5.6 mm/s). Pull-forward maintenance prevents unplanned outage.",
                "status": "auto_executed",
            },
            {
                "id": "DEC-20221020-005",
                "type": "supplier_switch",
                "subject_ids": ["S004", "RM0049"],
                "recommended_action": "Reroute raw material procurement to secondary qualified supplier S002",
                "rule_tier": "REVIEW",
                "confidence": 0.74,
                "rationale": "Supplier S004 rejection rate rose to 7.82%. Confidence 0.74 < 0.80 requires human review.",
                "status": "pending_review",
            },
        ]

        for item in demo_items:
            dec = DecisionRecord(
                id=item["id"],
                type=item["type"],
                subject_ids=item["subject_ids"],
                recommended_action=item["recommended_action"],
                rule_tier=item["rule_tier"],
                confidence=item["confidence"],
                rationale=item["rationale"],
                inputs_hash=hashlib.sha256(json.dumps(item["subject_ids"]).encode()).hexdigest(),
                model_versions={"policy": "v1.0", "ensemble": "champion-v1"},
                status=item["status"],
            )
            self._decisions[dec.id] = dec

    # -------------------------------------------------------------
    # Rule Tier Classification (Deterministic §10.2)
    # -------------------------------------------------------------
    def evaluate_rule_tier(
        self,
        audit_status: Optional[str] = None,
        release_status: Optional[str] = None,
        temperature_excursion_flag: int = 0,
        deviation_count: int = 0,
        supplier_risk_score: float = 0.0,
        delivery_delay_days: int = 0,
    ) -> Tuple[RuleTier, str]:
        """
        Pure deterministic code. Evaluates C1-aware policy rule tiers.
        """
        # BLOCK Tier Checks
        if audit_status == "Major Deviation":
            return ("BLOCK", "BLOCK Tier: Major Deviation identified in compliance audit (C1). QA action mandatory.")
        if release_status == "FAIL":
            return ("BLOCK", "BLOCK Tier: QC release_status = 'FAIL' (C8). Release strictly prohibited.")
        if temperature_excursion_flag == 1:
            return ("BLOCK", "BLOCK Tier: Cold-chain temperature excursion detected. Mandatory QA intervention.")

        # REVIEW Tier Checks
        if audit_status == "Minor Deviation":
            return ("REVIEW", "REVIEW Tier: Minor Deviation flagged in compliance audit (C1). Human sign-off required.")
        if release_status == "RETEST":
            return ("REVIEW", "REVIEW Tier: Batch required laboratory RETEST (C8). Human review mandatory.")
        if deviation_count >= 2:
            return ("REVIEW", f"REVIEW Tier: Cumulative deviation count ({deviation_count}) meets review threshold.")
        if supplier_risk_score >= 65.0:
            return ("REVIEW", f"REVIEW Tier: Supplier risk score ({supplier_risk_score:.1f}) exceeds 65.0 threshold.")
        if delivery_delay_days >= 5:
            return ("REVIEW", f"REVIEW Tier: Material delivery delay ({delivery_delay_days} days) exceeds threshold.")

        # AUTO-ELIGIBLE
        return ("AUTO-ELIGIBLE", "AUTO-ELIGIBLE Tier: All safety, quality, and cold-chain compliance gates passed.")

    # -------------------------------------------------------------
    # Confidence Score Computation (§10.3)
    # -------------------------------------------------------------
    def compute_confidence(
        self,
        ensemble_agreement: float = 0.90,
        interval_width_ratio: float = 0.20,
        days_stale: float = 1.0,
    ) -> float:
        """
        Confidence formula documented per §10.3 & ASSUMPTIONS.md:
          confidence = 0.45 * agreement + 0.35 * (1 - interval_ratio) + 0.20 * exp(-days_stale / 14)
        """
        w_agr = self.config.get("confidence", {}).get("weights", {}).get("ensemble_agreement", 0.45)
        w_int = self.config.get("confidence", {}).get("weights", {}).get("interval_precision", 0.35)
        w_frs = self.config.get("confidence", {}).get("weights", {}).get("data_freshness", 0.20)
        half_life = self.config.get("confidence", {}).get("freshness_half_life_days", 14.0)

        # Clamped sub-scores
        agr_score = max(0.0, min(1.0, ensemble_agreement))
        int_score = max(0.0, min(1.0, 1.0 - interval_width_ratio))
        import math
        frs_score = math.exp(-max(0.0, days_stale) / half_life)

        raw_conf = w_agr * agr_score + w_int * int_score + w_frs * frs_score
        return round(float(max(0.0, min(1.0, raw_conf))), 4)

    # -------------------------------------------------------------
    # Decision Creation & Auto-Execution Gate (§10.1, §10.3)
    # -------------------------------------------------------------
    def propose_decision(
        self,
        decision_id: str,
        decision_type: DecisionType,
        subject_ids: List[str],
        recommended_action: str,
        audit_status: Optional[str] = None,
        release_status: Optional[str] = None,
        temperature_excursion_flag: int = 0,
        deviation_count: int = 0,
        supplier_risk_score: float = 0.0,
        delivery_delay_days: int = 0,
        ensemble_agreement: float = 0.90,
        interval_width_ratio: float = 0.20,
        days_stale: float = 1.0,
        model_versions: Optional[Dict[str, str]] = None,
    ) -> DecisionRecord:
        """
        Evaluates tier and confidence.
        Applies auto-execution gate: requires tier == AUTO-ELIGIBLE and confidence >= 0.80.
        BLOCK tier can NEVER yield auto_executed.
        """
        tier, rationale = self.evaluate_rule_tier(
            audit_status=audit_status,
            release_status=release_status,
            temperature_excursion_flag=temperature_excursion_flag,
            deviation_count=deviation_count,
            supplier_risk_score=supplier_risk_score,
            delivery_delay_days=delivery_delay_days,
        )

        conf = self.compute_confidence(
            ensemble_agreement=ensemble_agreement,
            interval_width_ratio=interval_width_ratio,
            days_stale=days_stale,
        )

        auto_threshold = self.config.get("rule_tiers", {}).get("auto_eligible", {}).get("min_confidence", 0.80)

        # Mandatory Gate (§10.3): BLOCK tier can NEVER be auto-approved
        if tier == "BLOCK":
            status: DecisionStatus = "pending_review"
        elif tier == "REVIEW":
            status = "pending_review"
        elif tier == "AUTO-ELIGIBLE" and conf >= auto_threshold:
            status = "auto_executed"
        else:
            status = "pending_review"

        inputs_data = {
            "subject_ids": subject_ids,
            "action": recommended_action,
            "audit_status": audit_status,
            "release_status": release_status,
            "tier": tier,
            "conf": conf,
        }
        inputs_hash = hashlib.sha256(json.dumps(inputs_data, sort_keys=True).encode()).hexdigest()

        rec = DecisionRecord(
            id=decision_id,
            type=decision_type,
            subject_ids=subject_ids,
            recommended_action=recommended_action,
            rule_tier=tier,
            confidence=conf,
            rationale=rationale,
            inputs_hash=inputs_hash,
            model_versions=model_versions or {"policy": "v1.0"},
            status=status,
        )
        self._decisions[decision_id] = rec

        # Log creation to audit chain
        self.audit.append("DecisionCreated", {
            "decision_id": rec.id,
            "type": rec.type,
            "rule_tier": rec.rule_tier,
            "confidence": rec.confidence,
            "status": rec.status,
            "inputs_hash": rec.inputs_hash,
        })

        return rec

    def create_decision(
        self,
        type: DecisionType,
        subject_ids: List[str],
        recommended_action: str,
        rule_tier: RuleTier,
        confidence: float,
        rationale: str,
        inputs_hash: Optional[str] = None,
        id: Optional[str] = None,
        model_versions: Optional[Dict[str, str]] = None,
    ) -> DecisionRecord:
        import uuid
        dec_id = id or f"DEC-{uuid.uuid4().hex[:8].upper()}"
        auto_threshold = self.config.get("rule_tiers", {}).get("auto_eligible", {}).get("min_confidence", 0.80)
        tier_str = str(rule_tier.value if hasattr(rule_tier, "value") else rule_tier)
        if tier_str == "BLOCK":
            status: DecisionStatus = "pending_review"
        elif tier_str == "REVIEW":
            status = "pending_review"
        elif tier_str == "AUTO-ELIGIBLE" and confidence >= auto_threshold:
            status = "auto_executed"
        else:
            status = "pending_review"

        rec = DecisionRecord(
            id=dec_id,
            type=type,
            subject_ids=subject_ids,
            recommended_action=recommended_action,
            rule_tier=rule_tier,
            confidence=confidence,
            rationale=rationale,
            inputs_hash=inputs_hash or "inputs_hash_placeholder",
            model_versions=model_versions or {"policy": "v1.0"},
            status=status,
        )
        self._decisions[dec_id] = rec
        self.audit.append("DecisionCreated", {
            "decision_id": rec.id,
            "type": str(rec.type.value if hasattr(rec.type, "value") else rec.type),
            "rule_tier": str(rec.rule_tier.value if hasattr(rec.rule_tier, "value") else rec.rule_tier),
            "confidence": rec.confidence,
            "status": str(rec.status.value if hasattr(rec.status, "value") else rec.status),
            "inputs_hash": rec.inputs_hash,
        })
        return rec

    # -------------------------------------------------------------
    # RBAC & Workflow Resolution (§10.4)
    # -------------------------------------------------------------
    def resolve_decision(
        self,
        decision_id: str,
        action: Literal["approve", "reject", "approved", "rejected"],
        resolution_note: str,
        role: Role,
        username: str,
    ) -> DecisionRecord:
        """
        Resolves a decision in the approval queue.
        Server-side RBAC: Only 'qa_reviewer' (or 'admin') can resolve QA/compliance-gated decisions.
        If a 'planner' attempts to resolve a BLOCK-tier or batch_release decision, raises PermissionError.
        """
        if decision_id not in self._decisions:
            raise KeyError(f"Decision {decision_id} not found")

        dec = self._decisions[decision_id]

        if dec.status in ["approved", "rejected", "auto_executed"]:
            raise ValueError(f"Decision {decision_id} is already in terminal state: {dec.status}")

        # Server-side RBAC validation (§10.4)
        tier_str = str(dec.rule_tier.value if hasattr(dec.rule_tier, "value") else dec.rule_tier)
        type_str = str(dec.type.value if hasattr(dec.type, "value") else dec.type)
        role_str = str(role.value if hasattr(role, "value") else role)
        is_qa_gated = (tier_str == "BLOCK" or type_str == "batch_release")
        if is_qa_gated and role_str == "planner":
            raise PermissionError(
                f"HTTP 403 Forbidden: User '{username}' with role 'planner' cannot resolve "
                f"QA/compliance-gated decision '{decision_id}' (tier: {tier_str}, type: {type_str}). "
                f"Role 'qa_reviewer' or 'admin' required."
            )

        new_status: DecisionStatus = "approved" if action in ["approve", "approved"] else "rejected"
        resolved_at = datetime.utcnow().isoformat()

        # Update decision
        dec.status = new_status
        dec.resolver = username
        dec.resolved_at = resolved_at
        dec.resolution_note = resolution_note
        self._decisions[decision_id] = dec

        # Log resolution to tamper-evident audit log
        self.audit.append("DecisionResolved", {
            "decision_id": decision_id,
            "action": action,
            "resolver": username,
            "role": role,
            "resolution_note": resolution_note,
            "resolved_at": resolved_at,
        })

        return dec

    def list_decisions(self, status: Optional[str] = None, tier: Optional[str] = None) -> List[DecisionRecord]:
        """Returns filtered decision records."""
        items = list(self._decisions.values())
        if status:
            items = [d for d in items if d.status == status]
        if tier:
            items = [d for d in items if d.rule_tier == tier]
        return items

    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        return self._decisions.get(decision_id)
