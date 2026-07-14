from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ImplementationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileReplacement(ImplementationSchema):
    path: str = Field(description="Repository-relative path named by the approved plan.")
    old_text: str = Field(description="Exact existing text to replace; empty only for a new file.")
    new_text: str = Field(description="Complete replacement text.")
    reason: str = Field(description="Short connection to the approved plan.")


class ImplementationProposal(ImplementationSchema):
    status: Literal["ready", "needs_user_decision", "blocked"]
    summary: str
    replacements: list[FileReplacement] = Field(
        description="Ordered exact replacements; return [] when blocked."
    )
    deviations: list[str] = Field(
        description="Differences from the plan; return [] if none."
    )
    questions: list[str] = Field(
        description="Material human decisions; return [] if none."
    )
