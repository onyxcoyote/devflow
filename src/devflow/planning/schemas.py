from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

ShortText = Annotated[
    str,
    Field(description="Concise text; keep to 500 characters or fewer."),
]
DetailText = Annotated[
    str,
    Field(description="Detailed text; keep to 1,500 characters or fewer."),
]
PathText = Annotated[
    str,
    Field(description="Repository-relative path; keep to 500 characters or fewer."),
]


class PlanSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProposedChange(PlanSchema):
    path: PathText
    symbols: list[ShortText] = Field(
        description="Affected symbols. Return [] if empty; maximum 20 items."
    )
    change: DetailText
    reason: DetailText
    evidence: list[ShortText] = Field(
        description="Supporting repository evidence. Return [] if empty; maximum 20 items."
    )
    dependencies: list[ShortText] = Field(
        description="Dependencies on other changes. Return [] if empty; maximum 20 items."
    )


class OutstandingItem(PlanSchema):
    kind: Literal["repository_context", "user_decision", "external_information"]
    question: DetailText
    impact: DetailText
    suggested_action: DetailText


class PlanRisk(PlanSchema):
    description: DetailText
    likelihood: Literal["low", "medium", "high"]
    impact: Literal["low", "medium", "high"]
    mitigation: DetailText
    related_files: list[PathText] = Field(
        description="Related repository paths. Return [] if empty; maximum 20 items."
    )


class PlanDecision(PlanSchema):
    question: DetailText
    status: Literal["resolved", "unresolved"]
    decision: DetailText = Field(
        description="The decision, or an empty string when unresolved."
    )
    rationale: DetailText
    source: Literal["repository", "user", "planner"]


class PlanRevision(PlanSchema):
    based_on: str | None = Field(
        description="Previous plan path, or null for an initial plan."
    )
    changes: list[DetailText] = Field(
        description="Changes from the previous plan. Return [] for an initial plan; maximum 20 items."
    )


class DevelopmentPlan(PlanSchema):
    status: Literal[
        "ready",
        "needs_repository_context",
        "needs_user_decision",
        "blocked",
        "not_feasible",
    ]
    objective: DetailText
    design_summary: str = Field(
        description="Concise design summary; keep to 3,000 characters or fewer."
    )
    assumptions: list[DetailText] = Field(
        description="Material assumptions. Return [] if empty; maximum 20 items."
    )
    proposed_changes: list[ProposedChange] = Field(
        description="File-specific proposed changes. Return [] if empty; maximum 30 items."
    )
    outstanding_items: list[OutstandingItem] = Field(
        description="Unresolved blockers or questions. Return [] if empty; maximum 15 items."
    )
    decisions: list[PlanDecision] = Field(
        description="Resolved and unresolved decisions. Return [] if empty; maximum 20 items."
    )
    acceptance_criteria: list[DetailText] = Field(
        description="Acceptance criteria. Return [] if empty; maximum 30 items."
    )
    verification: list[DetailText] = Field(
        description="Verification steps. Return [] if empty; maximum 30 items."
    )
    risks: list[PlanRisk] = Field(
        description="Material implementation risks. Return [] if empty; maximum 15 items."
    )
    revision: PlanRevision = Field(
        description="Revision metadata. Use null based_on and [] changes for an initial plan."
    )
