from prefect import flow

from .config import PlanningConfig
from .tasks import collect_repository_context, run_planning_graph, save_plan_outputs


@flow(name="development-plan")
def planning_flow(request: str, config: PlanningConfig) -> dict:
    repository_context = collect_repository_context(config)
    final_state = run_planning_graph({
        "request": request,
        "repo_path": config.repo_path,
        "repository_context": repository_context,
        "context_text": "",
        "model_info": {
            "provider": config.model.provider,
            "model": config.model.model,
            "base_url": config.model.base_url,
            "temperature": config.model.temperature,
        },
        "model_result": {},
        "save_model_exchange": config.save_model_exchange,
        "model_exchange": {},
        "plan": {},
        "report": "",
    }, config)
    paths = save_plan_outputs(final_state, config.output_dir)
    return {"plan": final_state["plan"], "paths": paths}
