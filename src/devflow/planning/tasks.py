from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect import get_run_logger, task

from devflow.code_review.models import get_code_review_model

from .config import PlanningConfig
from .graph import build_planning_graph
from .schemas import PLAN_SCHEMA_VERSION, PLAN_STRUCTURED_OUTPUT_METHOD


@task
def run_planning_graph(initial_state: dict[str, Any], config: PlanningConfig) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting development plan with provider=%s model=%s endpoint=%s",
        config.model.provider,
        config.model.model,
        config.model.base_url,
    )
    logger.info(
        "Creating planning models with output limits initial=%d compact_retry=%d",
        config.max_output_tokens,
        config.compact_retry_output_tokens,
    )
    logger.info(
        "Planning structured output: method=%s schema=%s size_limits=guidance-only",
        PLAN_STRUCTURED_OUTPUT_METHOD,
        PLAN_SCHEMA_VERSION,
    )
    model = get_code_review_model(
        config.model,
        max_output_tokens=config.max_output_tokens,
    )
    compact_retry_model = get_code_review_model(
        config.model,
        max_output_tokens=config.compact_retry_output_tokens,
    )
    logger.info(
        "Planning client parameters: initial max_tokens=%s num_predict=%s; "
        "compact retry max_tokens=%s num_predict=%s",
        getattr(model, "max_tokens", None),
        getattr(model, "num_predict", None),
        getattr(compact_retry_model, "max_tokens", None),
        getattr(compact_retry_model, "num_predict", None),
    )
    logger.info("Building planning graph")
    graph = build_planning_graph(model, compact_retry_model, logger)
    logger.info("Invoking planning graph")
    result = graph.invoke(initial_state)
    logger.info("Planning graph completed with status=%s", result["plan"]["status"])
    return result


@task
def save_plan_outputs(final_state: dict[str, Any], output_dir: str) -> dict[str, str]:
    root = Path(output_dir)
    run_dir = root / "runs" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    markdown_path = run_dir / "plan.md"
    json_path = run_dir / "plan.json"
    evidence_path = run_dir / "evidence.json"
    exchange_path = run_dir / "model-exchange.json"

    markdown_path.write_text(final_state["report"], encoding="utf-8")
    json_path.write_text(
        json.dumps(final_state["plan"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps({
            "request": final_state["request"],
            "repo_path": final_state["repo_path"],
            "context_source": final_state["context_source"],
            "model_info": final_state["model_info"],
            "model_result": final_state["model_result"],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if final_state["save_model_exchange"]:
        exchange_path.write_text(
            json.dumps(
                final_state["model_exchange"],
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(Path("runs") / run_dir.name, target_is_directory=True)
    paths = {
        "markdown": str(markdown_path.resolve()),
        "json": str(json_path.resolve()),
        "evidence": str(evidence_path.resolve()),
    }
    if final_state["save_model_exchange"]:
        paths["model_exchange"] = str(exchange_path.resolve())
    return paths
