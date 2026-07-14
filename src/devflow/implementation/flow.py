from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from prefect import flow, get_run_logger

from devflow.code_review.config import CodeReviewConfig
from devflow.code_review.models import get_code_review_model
from devflow.code_review.tasks import _run_command

from .schemas import ImplementationProposal


MAX_SOURCE_CHARS = 80_000


def _confirm(prompt: str, auto_approve: bool) -> bool:
    if auto_approve:
        return True
    if not sys.stdin.isatty():
        return False
    return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}


def _load_plan(plan_path: str, repo_path: str) -> tuple[dict, Path]:
    path = Path(plan_path).expanduser().resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("status") != "ready":
        raise ValueError("Implementation requires an approved plan with status=ready")
    evidence_path = path.with_name("evidence.json")
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if Path(evidence.get("repo_path", "")).resolve() != Path(repo_path).resolve():
            raise ValueError("Plan repository does not match implementation repository")
    return plan, path


def _planned_sources(plan: dict, repo_path: str) -> tuple[list[str], dict[str, str]]:
    root = Path(repo_path).resolve()
    paths = []
    sources = {}
    total = 0
    for change in plan.get("proposed_changes", []):
        relative = change.get("path", "")
        if not relative or relative in paths:
            continue
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError(f"Plan path escapes repository: {relative}")
        paths.append(relative)
        if candidate.is_file() and total < MAX_SOURCE_CHARS:
            try:
                content = candidate.read_text(encoding="utf-8")
            except UnicodeError as error:
                raise ValueError(f"Planned file is not UTF-8 text: {relative}") from error
            remaining = MAX_SOURCE_CHARS - total
            sources[relative] = content[:remaining]
            total += len(sources[relative])
    return paths, sources


def _validate_replacements(proposal: dict, planned_paths: list[str], repo_path: str) -> None:
    root = Path(repo_path).resolve()
    simulated: dict[str, str | None] = {}
    for edit in proposal["replacements"]:
        path = edit["path"]
        if path not in planned_paths:
            raise ValueError(f"Model proposed an unplanned file: {path}")
        candidate = (root / path).resolve()
        candidate.relative_to(root)
        old = edit["old_text"]
        if path not in simulated:
            simulated[path] = (
                candidate.read_text(encoding="utf-8")
                if candidate.exists()
                else None
            )
        content = simulated[path]
        if content is not None:
            if not old:
                raise ValueError(f"Existing file requires non-empty old_text: {path}")
            if content.count(old) != 1:
                raise ValueError(
                    f"old_text must match exactly once at this edit step in {path}"
                )
            simulated[path] = content.replace(old, edit["new_text"], 1)
        elif old:
            raise ValueError(f"New file must use empty old_text: {path}")
        else:
            simulated[path] = edit["new_text"]


def _apply_replacements(proposal: dict, repo_path: str) -> None:
    root = Path(repo_path).resolve()
    for edit in proposal["replacements"]:
        candidate = (root / edit["path"]).resolve()
        if candidate.exists():
            content = candidate.read_text(encoding="utf-8")
            content = content.replace(edit["old_text"], edit["new_text"], 1)
        else:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            content = edit["new_text"]
        candidate.write_text(content, encoding="utf-8")


@flow(name="implementation")
def implementation_flow(
    plan_path: str,
    config: CodeReviewConfig,
    *,
    auto_approve: bool = False,
) -> dict:
    logger = get_run_logger()
    plan, resolved_plan_path = _load_plan(plan_path, config.repo_path)
    planned_paths, sources = _planned_sources(plan, config.repo_path)
    prompt = (
        "Implement the approved plan using exact text replacements. Stay within planned files. "
        "Inspect the supplied bounded source before editing. Preserve existing behavior unless "
        "the plan changes it. If implementation reveals a material product, compatibility, "
        "security, data, or architecture decision, return needs_user_decision with no edits. "
        "Record non-material departures in deviations. Do not use placeholders.\n\n"
        f"PLAN\n{json.dumps(plan, indent=2)}\n\n"
        f"PLANNED SOURCES\n{json.dumps(sources, indent=2)}"
    )
    model = get_code_review_model(config.model)
    structured = model.with_structured_output(
        ImplementationProposal,
        method="function_calling",
    )
    logger.info("Requesting implementation proposal for %d planned files", len(planned_paths))
    proposal = structured.invoke(prompt).model_dump()

    run_dir = Path(config.repo_path) / ".devflow" / "implementations" / "runs" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    proposal_path = run_dir / "proposal.json"
    proposal_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    print(f"Implementation proposal: {proposal_path.resolve()}")
    print(f"Status: {proposal['status']}; edits: {len(proposal['replacements'])}")
    for deviation in proposal["deviations"]:
        print(f"Deviation: {deviation}")
    for question in proposal["questions"]:
        print(f"Decision needed: {question}")

    applied = False
    command_results = []
    if proposal["status"] == "ready" and proposal["replacements"]:
        _validate_replacements(proposal, planned_paths, config.repo_path)
        if _confirm("Apply this implementation proposal?", auto_approve):
            _apply_replacements(proposal, config.repo_path)
            applied = True
            command_results = [
                _run_command(command, config.repo_path, config.max_command_output_chars)
                for command in (*config.check_commands, *config.test_commands)
            ]
            for result in command_results:
                print(f"Validation {'passed' if result['passed'] else 'FAILED'}: {' '.join(result['command'])}")
    evidence = {
        "plan_path": str(resolved_plan_path),
        "planned_paths": planned_paths,
        "proposal_status": proposal["status"],
        "applied": applied,
        "command_results": command_results,
        "git_diff": subprocess.run(
            ["git", "diff", "--no-ext-diff", "--"], cwd=config.repo_path,
            capture_output=True, text=True, check=True,
        ).stdout,
    }
    evidence_path = run_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return {"proposal": proposal, "applied": applied, "paths": {
        "proposal": str(proposal_path.resolve()),
        "evidence": str(evidence_path.resolve()),
    }}
