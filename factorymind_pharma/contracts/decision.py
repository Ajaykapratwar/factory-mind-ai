from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict

from enum import Enum

class RuleTier(str, Enum):
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"
    AUTO_ELIGIBLE = "AUTO-ELIGIBLE"
    AUTO_ELIGIBLE_UNDERSCORE = "AUTO_ELIGIBLE"

class DecisionStatus(str, Enum):
    PROPOSED = "proposed"
    AUTO_EXECUTED = "auto_executed"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

class DecisionType(str, Enum):
    RESTOCK = "restock"
    BATCH_RELEASE = "batch_release"
    MAINTENANCE = "maintenance"
    MAINTENANCE_PULL_FORWARD = "maintenance_pull_forward"
    SUPPLIER_SWITCH = "supplier_switch"

class Role(str, Enum):
    PLANNER = "planner"
    QA_REVIEWER = "qa_reviewer"
    ADMIN = "admin"

class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique decision ID, e.g. DEC-20221020-001")
    type: DecisionType = Field(..., description="Decision action category")
    subject_ids: List[str] = Field(..., description="Subject entities affected (SKU, batch, machine, supplier)")
    recommended_action: str = Field(..., description="Specific recommended operational action")
    rule_tier: RuleTier = Field(..., description="Deterministic rule tier (BLOCK, REVIEW, AUTO-ELIGIBLE)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Computed confidence score in [0, 1]")
    rationale: str = Field(..., description="Detailed rationale explaining trigger and tier logic")
    inputs_hash: str = Field(..., description="SHA-256 hash of canonical decision inputs")
    model_versions: Dict[str, str] = Field(default_factory=dict, description="Versions of models that contributed")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO UTC timestamp")
    status: DecisionStatus = Field(default="proposed", description="Lifecycle state")
    resolver: Optional[str] = Field(None, description="Username of resolver if acted upon")
    resolved_at: Optional[str] = Field(None, description="ISO UTC timestamp of resolution")
    resolution_note: Optional[str] = Field(None, description="Note provided during resolution")

class ResolveDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "reject"] = Field(..., description="Resolution action")
    resolution_note: str = Field(..., description="Mandatory rationale or audit note")
    role: Role = Field(..., description="User role executing the resolution")
    username: str = Field(..., description="User ID or username")
