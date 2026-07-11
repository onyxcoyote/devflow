from typing import Any, TypedDict


class PlanningState(TypedDict):
    request: str
    repo_path: str
    repository_context: dict[str, Any]
    tracked_files: list[str]
    context_request: dict[str, Any]
    context_text: str
    model_info: dict[str, Any]
    model_result: dict[str, Any]
    max_context_chars: int
    max_requested_files: int
    max_searches: int
    max_search_results_chars: int
    save_model_exchange: bool
    model_exchange: dict[str, Any]
    plan: dict[str, Any]
    report: str
