import json
import subprocess
from pathlib import Path
from typing import Any

from prefect import task

from .config import CodeReviewConfig
from .graph import code_review_graph


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
        stdout = result.stdout[-max_output_chars:]
        stderr = result.stderr[-max_output_chars:]
        return {
            "command": list(command),
            "return_code": result.returncode,
            "passed": result.returncode == 0,
            "timed_out": False,
            "stdout": stdout,
            "stderr": stderr,
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
    repo = Path(config.repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise ValueError(f"Repository path does not exist: {repo}")

    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise ValueError(f"Not a Git work tree: {repo}")

    base_result = subprocess.run(
        ["git", "rev-parse", "--verify", config.base_ref],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if base_result.returncode != 0:
        raise ValueError(
            f"Base ref does not exist in repository: {config.base_ref}"
        )


@task
def collect_change(config: CodeReviewConfig) -> dict[str, Any]:
    repo = str(Path(config.repo_path).expanduser().resolve())

    files_result = subprocess.run(
        ["git", "diff", "--name-only", config.base_ref, "--"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    diff_result = subprocess.run(
        [
            "git",
            "diff",
            "--find-renames",
            "--no-ext-diff",
            config.base_ref,
            "--",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )

    full_diff = diff_result.stdout
    return {
        "changed_files": [
            line for line in files_result.stdout.splitlines() if line
        ],
        "diff": full_diff[: config.max_diff_chars],
        "diff_truncated": len(full_diff) > config.max_diff_chars,
    }


@task
def run_configured_commands(
    config: CodeReviewConfig,
) -> list[dict[str, Any]]:
    commands = (*config.check_commands, *config.test_commands)
    return [
        _run_command(
            command,
            config.repo_path,
            config.max_command_output_chars,
        )
        for command in commands
    ]


@task
def run_code_review_graph(initial_state: dict[str, Any]) -> dict[str, Any]:
    return code_review_graph.invoke(initial_state)


@task
def save_review_outputs(
    final_state: dict[str, Any],
    output_dir: str,
) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    markdown_path = directory / "review.md"
    json_path = directory / "review.json"
    evidence_path = directory / "evidence.json"

    markdown_path.write_text(final_state["report"], encoding="utf-8")
    json_path.write_text(
        json.dumps(final_state["review"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            {
                "base_ref": final_state["base_ref"],
                "changed_files": final_state["changed_files"],
                "diff_truncated": final_state["diff_truncated"],
                "command_results": final_state["command_results"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "markdown": str(markdown_path),
        "json": str(json_path),
        "evidence": str(evidence_path),
    }
