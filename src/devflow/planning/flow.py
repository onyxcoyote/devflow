import json
import time

from prefect import flow, get_run_logger

from devflow.repository_context.config import SerenaContextConfig
from devflow.repository_context.flow import serena_context_flow

from .config import PlanningConfig
from .artifacts import load_context_artifact, load_previous_plan
from .research import (
    MAX_SUPPLEMENTAL_CONTEXT_ROUNDS,
    question_key,
    repository_context_questions,
    supplemental_context_request,
)
from .tasks import (
    run_planning_graph,
    save_plan_outputs,
)


def _planning_state(
    request: str,
    config: PlanningConfig,
    repository_context: dict,
    context_source: dict,
    previous_plan: dict | None,
) -> dict:
    return {
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
    }


@flow(name="development-plan")
def planning_flow(
    request: str,
    config: PlanningConfig,
    serena_config: SerenaContextConfig,
    context_path: str | None = None,
    previous_plan_path: str | None = None,
) -> dict:
    logger = get_run_logger()
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
    if not isinstance(repository_context, dict):
        repository_context = {"initial": repository_context}
    repository_context.setdefault("supplemental_rounds", [])
    context_source.setdefault("supplemental_rounds", [])

    final_state = run_planning_graph(
        _planning_state(
            request,
            config,
            repository_context,
            context_source,
            previous_plan,
        ),
        config,
    )
    seen_questions = set()
    seen_supplemental_reports = set()
    planning_rounds = []
    supplemental_rounds_completed = 0

    for round_number in range(1, MAX_SUPPLEMENTAL_CONTEXT_ROUNDS + 1):
        questions = repository_context_questions(final_state["plan"])
        planning_rounds.append({
            "round": round_number,
            "status": final_state["plan"]["status"],
            "repository_questions": questions,
            "plan_attempts": final_state["model_result"].get("plan_attempts", []),
        })
        new_questions = [
            item for item in questions
            if question_key(item["question"]) not in seen_questions
        ]
        if not new_questions:
            if questions:
                logger.warning(
                    "Stopping supplemental context: planner repeated previously investigated questions"
                )
            elif final_state["plan"]["status"] == "needs_repository_context":
                logger.warning(
                    "Stopping supplemental context: planner supplied no repository questions"
                )
            break

        logger.info(
            "Planning requested additional repository context (round %d/%d)",
            round_number,
            MAX_SUPPLEMENTAL_CONTEXT_ROUNDS,
        )
        for index, item in enumerate(new_questions, start=1):
            logger.info("Context question %d: %s", index, item["question"])
            seen_questions.add(question_key(item["question"]))

        supplemental_request = supplemental_context_request(
            request,
            new_questions,
            round_number,
        )
        logger.info("Sending %d targeted question(s) to Serena context", len(new_questions))
        supplemental_result = serena_context_flow(supplemental_request, serena_config)
        supplemental_report = supplemental_result["report"]
        report_key = json.dumps(
            supplemental_report,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        if report_key in seen_supplemental_reports:
            logger.warning("Stopping supplemental context: Serena returned no new evidence")
            break
        seen_supplemental_reports.add(report_key)

        repository_context["supplemental_rounds"].append({
            "round": round_number,
            "questions": new_questions,
            "report": supplemental_report,
        })
        context_source["supplemental_rounds"].append({
            "round": round_number,
            "questions": new_questions,
            "paths": supplemental_result.get("paths", {}),
        })
        supplemental_rounds_completed += 1
        logger.info(
            "Supplemental context round %d completed; retrying planning with added evidence",
            round_number,
        )
        time.sleep(serena_config.model_request_min_interval_seconds)
        final_state = run_planning_graph(
            _planning_state(
                request,
                config,
                repository_context,
                context_source,
                previous_plan,
            ),
            config,
        )
    else:
        planning_rounds.append({
            "round": MAX_SUPPLEMENTAL_CONTEXT_ROUNDS + 1,
            "status": final_state["plan"]["status"],
            "repository_questions": repository_context_questions(final_state["plan"]),
            "plan_attempts": final_state["model_result"].get("plan_attempts", []),
        })
        remaining_questions = repository_context_questions(final_state["plan"])
        if remaining_questions:
            logger.warning(
                "Stopping supplemental context: research budget exhausted with %d question(s) remaining",
                len(remaining_questions),
            )

    final_state["model_result"]["planning_rounds"] = planning_rounds
    logger.info(
        "Planning research completed after %d supplemental context round(s); final status=%s",
        supplemental_rounds_completed,
        final_state["plan"]["status"],
    )
    paths = save_plan_outputs(final_state, config.output_dir)
    return {"plan": final_state["plan"], "paths": paths}
