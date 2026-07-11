from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect import get_run_logger, task

from devflow.code_review.models import get_code_review_model

from .config import PlanningConfig
from .context import directory_summary
from .graph import build_planning_graph

INITIAL_CONTEXT_FILES = (
    "AGENTS.md",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
)


@task
def collect_repository_context(config: PlanningConfig) -> dict[str, Any]:
    repo = Path(config.repo_path)
    if not repo.is_dir():
        raise ValueError(f"Repository path does not exist: {repo}")

    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if inside.returncode != 0:
        raise ValueError(f"Not a Git work tree: {repo}")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked_files_text = subprocess.run(
        ["git", "ls-files"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked_files = [line for line in tracked_files_text.splitlines() if line]

    context: dict[str, Any] = {
        "head_commit": head,
        "git_status": status or "(clean)",
        "directory_summary": directory_summary(tracked_files),
        "key_files": {},
    }
    for name in INITIAL_CONTEXT_FILES:
        path = repo / name
        if path.is_file():
            context["key_files"][name] = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

    initial_budget = max(1, config.max_context_chars // 3)
    serialized = json.dumps(context, ensure_ascii=False)
    context["initial_context_truncated"] = len(serialized) > initial_budget
    if context["initial_context_truncated"]:
        key_files = context["key_files"]
        per_file_budget = initial_budget // max(1, len(key_files))
        context["key_files"] = {
            name: contents[:per_file_budget]
            for name, contents in key_files.items()
        }
    return {"repository_context": context, "tracked_files": tracked_files}


@task
def run_planning_graph(initial_state: dict[str, Any], config: PlanningConfig) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting development plan with provider=%s model=%s endpoint=%s",
        config.model.provider,
        config.model.model,
        config.model.base_url,
    )
    graph = build_planning_graph(get_code_review_model(config.model))
    return graph.invoke(initial_state)


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
        json.dumps(
            {
                "request": final_state["request"],
                "repo_path": final_state["repo_path"],
                "repository_context": final_state["repository_context"],
                "context_request": final_state["context_request"],
                "model_info": final_state["model_info"],
                "model_result": final_state["model_result"],
            },
            indent=2,
            ensure_ascii=False,
        ),
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
