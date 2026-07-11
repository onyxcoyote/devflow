from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect import get_run_logger, task

from .config import CodeReviewConfig
from .graph import build_code_review_graph
from .models import get_code_review_model


def _run_command(
    command: tuple[str, ...],
    repo_path: str,
    max_output_chars: int,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "command": list(command),
            "return_code": result.returncode,
            "passed": result.returncode == 0,
            "timed_out": False,
            "stdout": result.stdout[-max_output_chars:],
            "stderr": result.stderr[-max_output_chars:],
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": list(command),
            "return_code": None,
            "passed": False,
            "timed_out": True,
            "stdout": (error.stdout or "")[-max_output_chars:],
            "stderr": (error.stderr or "")[-max_output_chars:],
        }
    except OSError as error:
        return {
            "command": list(command),
            "return_code": None,
            "passed": False,
            "timed_out": False,
            "stdout": "",
            "stderr": str(error),
        }


@task
def validate_repository(config: CodeReviewConfig) -> None:
    repo = Path(config.repo_path)
    if not repo.is_dir():
        raise ValueError(f"Repository path does not exist: {repo}")

    checks = [
        (
            ["git", "rev-parse", "--is-inside-work-tree"],
            f"Not a Git work tree: {repo}",
        ),
        (
            ["git", "rev-parse", "--verify", config.base_ref],
            f"Base ref does not exist: {config.base_ref}",
        ),
    ]
    for command, message in checks:
        result = subprocess.run(
            command,
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(message)


@task
def collect_change(config: CodeReviewConfig) -> dict[str, Any]:
    files = subprocess.run(
        ["git", "diff", "--name-only", config.base_ref, "--"],
        cwd=config.repo_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    full_diff = subprocess.run(
        [
            "git",
            "diff",
            "--find-renames",
            "--no-ext-diff",
            config.base_ref,
            "--",
        ],
        cwd=config.repo_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        "changed_files": [line for line in files.splitlines() if line],
        "diff": full_diff[: config.max_diff_chars],
        "diff_truncated": len(full_diff) > config.max_diff_chars,
    }


@task
def run_configured_commands(config: CodeReviewConfig) -> list[dict[str, Any]]:
    return [
        _run_command(
            command,
            config.repo_path,
            config.max_command_output_chars,
        )
        for command in (*config.check_commands, *config.test_commands)
    ]


@task
def run_code_review_graph(
    initial_state: dict[str, Any],
    config: CodeReviewConfig,
) -> dict[str, Any]:
    logger = get_run_logger()
    logger.info(
        "Starting AI review with provider=%s model=%s endpoint=%s",
        config.model.provider,
        config.model.model,
        config.model.base_url,
    )
    graph = build_code_review_graph(get_code_review_model(config.model))
    return graph.invoke(initial_state)


@task
def save_review_outputs(
    final_state: dict[str, Any],
    output_dir: str,
) -> dict[str, str]:
    root = Path(output_dir)
    run_dir = root / "runs" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    markdown_path = run_dir / "review.md"
    json_path = run_dir / "review.json"
    evidence_path = run_dir / "evidence.json"

    markdown_path.write_text(final_state["report"], encoding="utf-8")
    json_path.write_text(
        json.dumps(final_state["assessment"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            {
                key: final_state[key]
                for key in (
                    "base_ref",
                    "changed_files",
                    "diff_truncated",
                    "command_results",
                    "model_info",
                )
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(Path("runs") / run_dir.name, target_is_directory=True)

    return {
        "markdown": str(markdown_path.resolve()),
        "json": str(json_path.resolve()),
        "evidence": str(evidence_path.resolve()),
    }
