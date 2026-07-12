from typing import Any, TypedDict


class PlanningState(TypedDict):
    request: str
    repo_path: str
    repository_context: dict[str, Any]
    context_text: str
    context_source: dict[str, Any]
    previous_plan: dict[str, Any] | None
    model_info: dict[str, Any]
    model_result: dict[str, Any]
    save_model_exchange: bool
    model_exchange: dict[str, Any]
    plan: dict[str, Any]
    report: str
