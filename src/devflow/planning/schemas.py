from typing import Literal

from pydantic import BaseModel, Field


class ProposedChange(BaseModel):
    path: str
    symbols: list[str] = Field(default_factory=list)
    change: str
    reason: str
    evidence: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class OutstandingItem(BaseModel):
    kind: Literal["repository_context", "user_decision", "external_information"]
    question: str
    impact: str
    suggested_action: str


class PlanRisk(BaseModel):
    description: str
    likelihood: Literal["low", "medium", "high"]
    impact: Literal["low", "medium", "high"]
    mitigation: str
    related_files: list[str] = Field(default_factory=list)


class PlanDecision(BaseModel):
    question: str
    status: Literal["resolved", "unresolved"]
    decision: str = ""
    rationale: str
    source: Literal["repository", "user", "planner"]


class PlanRevision(BaseModel):
    based_on: str | None = None
    changes: list[str] = Field(default_factory=list)


class DevelopmentPlan(BaseModel):
    status: Literal[
        "ready",
        "needs_repository_context",
        "needs_user_decision",
        "not_feasible",
    ]
    objective: str
    design_summary: str
    assumptions: list[str] = Field(default_factory=list)
    proposed_changes: list[ProposedChange] = Field(default_factory=list)
    outstanding_items: list[OutstandingItem] = Field(default_factory=list)
    decisions: list[PlanDecision] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    risks: list[PlanRisk] = Field(default_factory=list)
    revision: PlanRevision = Field(default_factory=PlanRevision)
