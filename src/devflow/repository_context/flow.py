from prefect import flow

from .config import SerenaContextConfig
from .serena import run_serena_context


@flow(name="serena-context")
def serena_context_flow(
    request: str,
    config: SerenaContextConfig,
    initial_report: dict | None = None,
    active_questions: list[str] | None = None,
    gate_between_rounds: bool = False,
    auto_approve: bool = False,
) -> dict:
    return run_serena_context(
        request,
        config,
        initial_report=initial_report,
        active_questions=active_questions,
        gate_between_rounds=gate_between_rounds,
        auto_approve=auto_approve,
    )
