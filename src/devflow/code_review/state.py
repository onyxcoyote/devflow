from typing import Any, TypedDict


class CodeReviewState(TypedDict):
    base_ref: str
    changed_files: list[str]
    diff: str
    diff_truncated: bool
    command_results: list[dict[str, Any]]
    review_context: str
    review: dict[str, Any]
    assessment: dict[str, Any]
    report: str
    model_info: dict[str, Any]
