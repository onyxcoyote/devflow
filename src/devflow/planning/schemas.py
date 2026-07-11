from typing import Literal

from pydantic import BaseModel, Field


class ProposedChange(BaseModel):
    area: str
    description: str
    likely_files: list[str] = Field(default_factory=list)
    reason: str


class DevelopmentPlan(BaseModel):
    status: Literal[
        "ready",
        "needs_context",
        "needs_user_decision",
        "not_feasible",
    ]
    objective: str
    understanding: str
    assumptions: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    proposed_changes: list[ProposedChange] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
