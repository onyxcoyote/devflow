# src/devflow/text_review/schemas.py

from typing import Literal

from pydantic import BaseModel, Field


class TextIssue(BaseModel):
    description: str = Field(
        description="A specific clarity or wording issue."
    )
    severity: Literal["low", "medium", "high"]


class AIReview(BaseModel):
    verdict: Literal["clear", "needs_improvement"]
    summary: str = Field(
        description="One concise assessment of the text."
    )
    issues: list[TextIssue]
