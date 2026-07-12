from prefect import flow

from .config import SerenaContextConfig
from .serena import run_serena_context


@flow(name="serena-context")
def serena_context_flow(request: str, config: SerenaContextConfig) -> dict:
    return run_serena_context(request, config)
