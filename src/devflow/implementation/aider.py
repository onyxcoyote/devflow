from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _aider_model(config) -> str:
    if config.aider_model:
        return config.aider_model
    if config.model.provider == "openrouter":
        return f"openrouter/{config.model.model}"
    if config.model.provider == "ollama":
        return f"ollama_chat/{config.model.model}"
    return config.model.model


def _repository_changes(repo_path: str) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_path, capture_output=True, text=True, check=True,
    )
    changes = []
    for line in result.stdout.splitlines():
        path = line[3:].split(" -> ")[-1]
        if path == ".devflow" or path.startswith(".devflow/"):
            continue
        changes.append(path)
    return changes


def _changed_paths(repo_path: str) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_path, capture_output=True, text=True, check=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        value = line[3:].split(" -> ")[-1]
        if value and value not in paths:
            paths.append(value)
    return paths


def build_aider_prompt(plan: dict, impact_context: dict, planned_paths: list[str]) -> str:
    investigations = impact_context.get("implementation_investigations", [])
    return (
        "Implement the approved development plan in this repository. Inspect the repository "
        "directly before editing. Follow the plan, preserve unrelated behavior, and do not "
        "invent fields or APIs. Trace every new value through its actual types and data flow. "
        "You may edit an additional file only when repository inspection proves it is needed; "
        "keep that edit minimal because Devflow will require separate approval. Do not commit.\n\n"
        f"APPROVED FILE SCOPE\n{json.dumps(planned_paths, indent=2)}\n\n"
        f"IMPLEMENTATION INVESTIGATIONS\n{json.dumps(investigations, indent=2)}\n\n"
        f"APPROVED PLAN\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n\n"
        f"IMPACT CONTEXT\n{json.dumps(impact_context, indent=2, ensure_ascii=False)}"
    )


def create_aider_proposal(
    *, plan: dict, impact_context: dict, planned_paths: list[str],
    repo_path: str, run_dir: Path, config,
) -> dict:
    executable = shutil.which("aider")
    if executable is None:
        raise RuntimeError(
            "Aider backend requested but the 'aider' executable was not found on PATH"
        )
    dirty = _repository_changes(repo_path)
    if dirty:
        raise RuntimeError(
            "Aider isolated mode requires a clean repository outside .devflow; "
            f"found changes: {', '.join(dirty[:10])}"
        )
    prompt_path = run_dir / "aider-prompt.txt"
    prompt_path.write_text(
        build_aider_prompt(plan, impact_context, planned_paths), encoding="utf-8"
    )
    transcript_path = run_dir / "aider-transcript.log"
    patch_path = run_dir / "aider.patch"
    worktree_parent = Path(tempfile.mkdtemp(prefix="devflow-aider-"))
    worktree = worktree_parent / "repo"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=repo_path, capture_output=True, text=True, check=True,
    )
    try:
        command = [
            executable,
            "--model", _aider_model(config),
            "--message-file", str(prompt_path),
            "--no-auto-commits",
            "--yes-always",
            *[path for path in planned_paths if (worktree / path).is_file()],
        ]
        env = os.environ.copy()
        if config.model.provider == "ollama":
            env.setdefault("OLLAMA_API_BASE", config.model.base_url)
        result = subprocess.run(
            command, cwd=worktree, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, check=False,
        )
        transcript_path.write_text(
            f"COMMAND: {json.dumps(command)}\nEXIT CODE: {result.returncode}\n\n"
            f"STDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "-N", "."], cwd=worktree, check=True)
        changed_paths = _changed_paths(str(worktree))
        patch_result = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "--"], cwd=worktree,
            capture_output=True, check=True,
        )
        patch_path.write_bytes(patch_result.stdout)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=repo_path, capture_output=True, check=False,
        )
        shutil.rmtree(worktree_parent, ignore_errors=True)
    extra_paths = [path for path in changed_paths if path not in planned_paths]
    missing_paths = [path for path in planned_paths if path not in changed_paths]
    status = "ready" if result.returncode == 0 and changed_paths else "blocked"
    return {
        "proposal": {
            "status": status,
            "summary": "Aider produced an isolated repository diff." if status == "ready" else (
                "Aider did not produce an applicable repository diff."
            ),
            "replacements": [],
            "deviations": [f"Additional changed file: {path}" for path in extra_paths],
            "questions": [],
            "changed_paths": changed_paths,
            "extra_paths": extra_paths,
            "missing_planned_paths": missing_paths,
            "exit_code": result.returncode,
        },
        "patch_path": str(patch_path.resolve()),
        "prompt_path": str(prompt_path.resolve()),
        "transcript_path": str(transcript_path.resolve()),
    }


def apply_aider_patch(patch_path: str, repo_path: str) -> None:
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", patch_path],
        cwd=repo_path, check=True,
    )
