"""
Scenario contract (Pydantic models) per Spec §4.4.
Enforces typed shocks with discriminator; unknown shock types fail validation.
"""
from typing import Annotated, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator, ConfigDict


class BaseShock(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MachineOutage(BaseShock):
    type: Literal["machine_outage"] = "machine_outage"
    machine_id: Optional[str] = None
    line_id: Optional[str] = None
    start_day: int = Field(ge=0, description="Start day relative to simulation start")
    duration_h: float = Field(gt=0, description="Duration in hours")

    @model_validator(mode="after")
    def validate_target(self):
        if not self.machine_id and not self.line_id:
            raise ValueError("MachineOutage must specify machine_id or line_id")
        return self


class SupplierDelay(BaseShock):
    type: Literal["supplier_delay"] = "supplier_delay"
    supplier_id: str
    extra_days: int = Field(gt=0, description="Extra days of supplier delivery delay")
    start_day: int = Field(ge=0, description="Start day relative to simulation start")


class DemandShock(BaseShock):
    type: Literal["demand_shock"] = "demand_shock"
    category: Optional[str] = None
    product_id: Optional[str] = None
    pct_change: float = Field(description="Percentage change in demand (e.g. +0.25 for +25%, -0.30 for -30%)")
    start_day: int = Field(ge=0, description="Start day relative to simulation start")
    duration_days: int = Field(gt=0, description="Duration of shock in days")

    @model_validator(mode="after")
    def validate_target(self):
        if not self.category and not self.product_id:
            raise ValueError("DemandShock must specify category or product_id")
        return self


class ColdChainEvent(BaseShock):
    type: Literal["cold_chain_event"] = "cold_chain_event"
    origin: Optional[str] = None
    destination: Optional[str] = None
    start_day: int = Field(ge=0, description="Start day relative to simulation start")
    duration_days: int = Field(gt=0, description="Duration of cold-chain event in days")


class QualityShift(BaseShock):
    type: Literal["quality_shift"] = "quality_shift"
    product_id: Optional[str] = None
    line_id: Optional[str] = None
    retest_rate_multiplier: float = Field(gt=0, description="Multiplier on retest rate (e.g. 2.0 = double retests)")


Shock = Annotated[
    Union[MachineOutage, SupplierDelay, DemandShock, ColdChainEvent, QualityShift],
    Field(discriminator="type")
]


class Scenario(BaseModel):
    id: str
    name: str
    horizon_days: Literal[30, 60, 90] = 30
    n_paths: int = Field(default=500, ge=10, le=5000)
    seed: int = 42
    shocks: List[Shock] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class MitigationAction(BaseModel):
    id: str
    action_type: Literal[
        "expedite_raw_material",
        "inter_warehouse_transfer",
        "reprioritize_skus",
        "pull_forward_maintenance"
    ]
    name: str
    description: str
    assumed_cost: float
    expected_profit_saved: float
    p_stockout_delta: float
    target_id: Optional[str] = None


class ScenarioMetricSummary(BaseModel):
    p_stockout_mean: float
    p_stockout_p95: float
    expected_lost_units: float
    fill_rate_p5: float
    fill_rate_p50: float
    fill_rate_p95: float
    days_of_supply_mean: float
    expected_profit: float
    profit_p5: float
    profit_p95: float


class ScenarioComparisonResult(BaseModel):
    scenario_id: str
    scenario_name: str
    horizon_days: int
    n_paths: int
    baseline_metrics: ScenarioMetricSummary
    shocked_metrics: ScenarioMetricSummary
    profit_delta: float
    profit_at_risk_p5: float
    p_stockout_delta: float
    fill_rate_delta: float
    ranked_mitigations: List[MitigationAction] = Field(default_factory=list)

    @property
    def baseline_summary(self) -> dict:
        return self.baseline_metrics.model_dump()

    @property
    def shocked_summary(self) -> dict:
        return self.shocked_metrics.model_dump()
