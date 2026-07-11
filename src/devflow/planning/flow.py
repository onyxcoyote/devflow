from prefect import flow

from .config import PlanningConfig
from .tasks import collect_repository_context, run_planning_graph, save_plan_outputs


@flow(name="development-plan")
def planning_flow(request: str, config: PlanningConfig) -> dict:
    collected_context = collect_repository_context(config)
    final_state = run_planning_graph({
        "request": request,
        "repo_path": config.repo_path,
        "repository_context": collected_context["repository_context"],
        "tracked_files": collected_context["tracked_files"],
        "context_request": {},
        "context_text": "",
        "model_info": {
            "provider": config.model.provider,
            "model": config.model.model,
            "base_url": config.model.base_url,
            "temperature": config.model.temperature,
        },
        "model_result": {},
        "max_context_chars": config.max_context_chars,
        "max_requested_files": config.max_requested_files,
        "max_searches": config.max_searches,
        "max_search_results_chars": config.max_search_results_chars,
        "save_model_exchange": config.save_model_exchange,
        "model_exchange": {},
        "plan": {},
        "report": "",
    }, config)
    paths = save_plan_outputs(final_state, config.output_dir)
    return {"plan": final_state["plan"], "paths": paths}
