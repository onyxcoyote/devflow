from prefect import flow

from .config import CodeReviewConfig
from .tasks import collect_change, run_code_review_graph, run_configured_commands, save_review_outputs, validate_repository


@flow(name="code-review")
def code_review_flow(config: CodeReviewConfig) -> dict:
    validate_repository(config)
    change = collect_change(config)
    command_results = run_configured_commands(config)
    final_state = run_code_review_graph({
        "base_ref": config.base_ref,
        **change,
        "command_results": command_results,
        "model_info": {
            "provider": config.model.provider,
            "model": config.model.model,
            "base_url": config.model.base_url,
            "temperature": config.model.temperature,
        },
        "model_result": {},
        "save_model_exchange": config.save_model_exchange,
        "model_exchange": {},
        "review_context": "",
        "review": {},
        "assessment": {},
        "report": "",
    }, config)
    paths = save_review_outputs(final_state, config.output_dir)
    return {"assessment": final_state["assessment"], "paths": paths}
