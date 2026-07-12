from typing import Annotated, Literal

from pydantic import BaseModel, Field

ShortText = Annotated[str, Field(max_length=500)]
DetailText = Annotated[str, Field(max_length=1500)]
PathText = Annotated[str, Field(max_length=500)]


class ProposedChange(BaseModel):
    path: PathText
    symbols: list[ShortText] = Field(default_factory=list, max_length=20)
    change: DetailText
    reason: DetailText
    evidence: list[ShortText] = Field(default_factory=list, max_length=20)
    dependencies: list[ShortText] = Field(default_factory=list, max_length=20)


class OutstandingItem(BaseModel):
    kind: Literal["repository_context", "user_decision", "external_information"]
    question: DetailText
    impact: DetailText
    suggested_action: DetailText


class PlanRisk(BaseModel):
    description: DetailText
    likelihood: Literal["low", "medium", "high"]
    impact: Literal["low", "medium", "high"]
    mitigation: DetailText
    related_files: list[PathText] = Field(default_factory=list, max_length=20)


class PlanDecision(BaseModel):
    question: DetailText
    status: Literal["resolved", "unresolved"]
    decision: DetailText = ""
    rationale: DetailText
    source: Literal["repository", "user", "planner"]


class PlanRevision(BaseModel):
    based_on: str | None = None
    changes: list[DetailText] = Field(default_factory=list, max_length=20)


class DevelopmentPlan(BaseModel):
    status: Literal[
        "ready",
        "needs_repository_context",
        "needs_user_decision",
        "blocked",
        "not_feasible",
    ]
    objective: DetailText
    design_summary: str = Field(max_length=3000)
    assumptions: list[DetailText] = Field(default_factory=list, max_length=20)
    proposed_changes: list[ProposedChange] = Field(default_factory=list, max_length=30)
    outstanding_items: list[OutstandingItem] = Field(default_factory=list, max_length=15)
    decisions: list[PlanDecision] = Field(default_factory=list, max_length=20)
    acceptance_criteria: list[DetailText] = Field(default_factory=list, max_length=30)
    verification: list[DetailText] = Field(default_factory=list, max_length=30)
    risks: list[PlanRisk] = Field(default_factory=list, max_length=15)
    revision: PlanRevision = Field(default_factory=PlanRevision)
