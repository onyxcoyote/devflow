import json
import time

from prefect import flow

from devflow.repository_context.config import SerenaContextConfig
from devflow.repository_context.flow import serena_context_flow

from .config import PlanningConfig
from .artifacts import load_context_artifact, load_previous_plan
from .tasks import (
    run_planning_graph,
    save_plan_outputs,
)


@flow(name="development-plan")
def planning_flow(
    request: str,
    config: PlanningConfig,
    serena_config: SerenaContextConfig,
    context_path: str | None = None,
    previous_plan_path: str | None = None,
) -> dict:
    previous_plan, resolved_previous_plan_path = load_previous_plan(
        previous_plan_path
    )
    if context_path:
        repository_context, context_source = load_context_artifact(
            context_path,
            config.repo_path,
        )
    else:
        discovery_request = request
        if previous_plan is not None:
            discovery_request += (
                "\n\nReview this previous implementation plan as a draft. Investigate "
                "repository facts needed to confirm, correct, or complete it:\n"
                + json.dumps(previous_plan, ensure_ascii=False)
            )
        context_result = serena_context_flow(discovery_request, serena_config)
        repository_context = context_result["report"]
        context_source = {
            "mode": "generated",
            **context_result["paths"],
        }
        time.sleep(serena_config.model_request_min_interval_seconds)
    if resolved_previous_plan_path:
        context_source["previous_plan_path"] = resolved_previous_plan_path
    final_state = run_planning_graph({
        "request": request,
        "repo_path": config.repo_path,
        "repository_context": repository_context,
        "context_source": context_source,
        "previous_plan": previous_plan,
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
