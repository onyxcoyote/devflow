from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path


def create_plan_run_dir(output_dir: str) -> Path:
    root = Path(output_dir)
    run_dir = root / "runs" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve()


def repository_head(repo_path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Not a Git work tree: {repo_path}")
    return result.stdout.strip()


def load_context_artifact(path: str, repo_path: str) -> tuple[dict, dict]:
    context_path = Path(path).expanduser().resolve()
    if not context_path.is_file():
        raise FileNotFoundError(f"Context file not found: {context_path}")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    evidence_path = context_path.with_name("evidence.json")
    if not evidence_path.is_file():
        raise ValueError(
            f"Context evidence file not found beside context: {evidence_path}"
        )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected_repo = str(Path(repo_path).resolve())
    if evidence.get("repo_path") != expected_repo:
        raise ValueError(
            "Context repository does not match the planning repository: "
            f"{evidence.get('repo_path')} != {expected_repo}"
        )
    context_head = evidence.get("head_commit")
    current_head = repository_head(repo_path)
    if not context_head:
        raise ValueError("Context evidence does not record a repository commit")
    if context_head != current_head:
        raise ValueError(
            f"Context is stale: it describes {context_head}, current HEAD is {current_head}"
        )
    return context, {
        "mode": "reused",
        "context_path": str(context_path),
        "evidence_path": str(evidence_path),
        "head_commit": context_head,
    }


def load_previous_plan(path: str | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, None
    plan_path = Path(path).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Previous plan not found: {plan_path}")
    return json.loads(plan_path.read_text(encoding="utf-8")), str(plan_path)
