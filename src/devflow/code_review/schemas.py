from typing import Literal

from pydantic import BaseModel, Field


class CodeFinding(BaseModel):
    severity: Literal["high", "medium", "low"]
    confidence: Literal["high", "medium", "low"]
    category: Literal[
        "correctness",
        "test_coverage",
        "maintainability",
        "performance",
        "security",
    ]
    file: str | None = None
    line: int | None = None
    summary: str
    reasoning: str
    suggested_action: str


class CheckAssessment(BaseModel):
    command_index: int = Field(ge=0)
    status: Literal[
        "passed",
        "change_failure",
        "unrelated_failure",
        "environment_failure",
        "uncertain",
    ]
    reasoning: str


class CodeReview(BaseModel):
    verdict: Literal[
        "approve",
        "changes_recommended",
        "blocked_by_failed_checks",
        "insufficient_context",
    ]
    confidence: Literal["high", "medium", "low"]
    summary: str
    findings: list[CodeFinding] = Field(default_factory=list, max_length=8)
    check_assessments: list[CheckAssessment] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
