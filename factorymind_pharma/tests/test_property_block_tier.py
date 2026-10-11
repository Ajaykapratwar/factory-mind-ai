"""
tests/test_property_block_tier.py
§14.1 / §12.3 — Property test (Hypothesis): BLOCK tier can NEVER yield auto_executed
for any generated input combination.
Run: pytest tests/test_property_block_tier.py -v
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from factorymind_pharma.contracts.decision import RuleTier, DecisionStatus, DecisionType
from factorymind_pharma.services.tower_svc.decision_engine import DecisionEngine


# Shared engine instance (decisions are in-memory; safe to reuse across examples)
_engine = DecisionEngine()


@given(
    decision_type=st.sampled_from(["batch_release", "restock", "maintenance", "supplier_switch"]),
    subject_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    rationale=st.text(min_size=0, max_size=200),
    inputs_hash=st.text(min_size=8, max_size=64, alphabet="0123456789abcdef"),
)
@settings(
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_block_tier_never_auto_executed(
    decision_type: str,
    subject_id: str,
    confidence: float,
    rationale: str,
    inputs_hash: str,
):
    """
    Property: For any combination of inputs, a BLOCK-tier DecisionRecord must NEVER
    receive status=AUTO_EXECUTED regardless of confidence value.

    This is the core safety invariant of §10.2 / §14.1.
    """
    dec = _engine.create_decision(
        type=decision_type,
        subject_ids=[subject_id or "X"],
        recommended_action="Auto-generated test action",
        rule_tier=RuleTier.BLOCK,
        confidence=confidence,
        rationale=rationale or "Property test",
        inputs_hash=inputs_hash,
    )
    assert dec.status != DecisionStatus.AUTO_EXECUTED, (
        f"SAFETY VIOLATION: BLOCK-tier decision {dec.id} was auto_executed! "
        f"confidence={confidence:.4f}, type={decision_type}"
    )


@given(
    confidence=st.floats(min_value=0.80, max_value=1.0, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_review_tier_never_auto_executed(confidence: float):
    """
    Property: REVIEW-tier decisions must also never be auto_executed,
    even at confidence >= 0.80.
    """
    dec = _engine.create_decision(
        type="batch_release",
        subject_ids=["PROP-REVIEW"],
        recommended_action="Review required",
        rule_tier=RuleTier.REVIEW,
        confidence=confidence,
        rationale="RETEST or Minor Deviation",
        inputs_hash="cafebabe",
    )
    assert dec.status != DecisionStatus.AUTO_EXECUTED, (
        f"SAFETY VIOLATION: REVIEW-tier decision {dec.id} was auto_executed! "
        f"confidence={confidence:.4f}"
    )


@given(
    confidence=st.floats(min_value=0.80, max_value=1.0, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_auto_eligible_with_high_confidence_can_auto_execute(confidence: float):
    """
    Positive property: AUTO-ELIGIBLE + confidence >= 0.80 MUST be auto_executed.
    This verifies the gate works in both directions.
    """
    dec = _engine.create_decision(
        type="restock",
        subject_ids=["PROP-AUTO"],
        recommended_action="Auto restock",
        rule_tier=RuleTier.AUTO_ELIGIBLE,
        confidence=confidence,
        rationale="All clear, high confidence",
        inputs_hash="00112233",
    )
    assert dec.status == DecisionStatus.AUTO_EXECUTED, (
        f"AUTO-ELIGIBLE + confidence={confidence:.4f} did not auto_execute: status={dec.status}"
    )
