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


class ReplacementValidationError(ValueError):
    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


def _text_preview(value: str, limit: int = 240) -> str:
    escaped = value.replace("\r", "\\r").replace("\n", "\\n")
    return escaped[:limit] + ("..." if len(escaped) > limit else "")


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


def _load_impact_context(plan_path: Path) -> dict:
    path = plan_path.with_name("impact-context.json")
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _impact_paths(context: dict) -> list[str]:
    paths = []
    for chain in context.get("impact_chains", []):
        values = chain.get("affected_definitions", [])
        values += [stage.get("path", "") for stage in chain.get("stages", [])]
        for value in values:
            path = value.split(":", 1)[0]
            if path and path not in paths:
                paths.append(path)
    for decision in context.get("architecture_decisions", []):
        for path in decision.get("affected_files", []):
            if path and path not in paths:
                paths.append(path)
    return paths


def _validate_replacements(proposal: dict, planned_paths: list[str], repo_path: str) -> None:
    root = Path(repo_path).resolve()
    simulated: dict[str, str | None] = {}
    edits_applied_by_path: dict[str, int] = {}
    for edit_index, edit in enumerate(proposal["replacements"], start=1):
        path = edit["path"]
        if path not in planned_paths:
            raise ReplacementValidationError(
                f"Edit {edit_index} proposed an unplanned file: {path}",
                {"edit_index": edit_index, "path": path, "reason": "unplanned_path"},
            )
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
                raise ReplacementValidationError(
                    f"Edit {edit_index} has empty old_text for existing file {path}",
                    {"edit_index": edit_index, "path": path, "reason": "empty_old_text"},
                )
            match_count = content.count(old)
            if match_count != 1:
                details = {
                    "edit_index": edit_index,
                    "path": path,
                    "reason": "old_text_match_count",
                    "match_count": match_count,
                    "prior_edits_in_file": edits_applied_by_path.get(path, 0),
                    "old_text": old,
                    "old_text_preview": _text_preview(old),
                }
                raise ReplacementValidationError(
                    f"Edit {edit_index} old_text matched {match_count} times in {path} "
                    f"after {details['prior_edits_in_file']} prior edit(s); expected exactly 1. "
                    f"old_text={details['old_text_preview']!r}",
                    details,
                )
            simulated[path] = content.replace(old, edit["new_text"], 1)
        elif old:
            raise ReplacementValidationError(
                f"Edit {edit_index} uses old_text for new file {path}",
                {"edit_index": edit_index, "path": path, "reason": "new_file_old_text"},
            )
        else:
            simulated[path] = edit["new_text"]
        edits_applied_by_path[path] = edits_applied_by_path.get(path, 0) + 1


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
    run_dir = Path(config.repo_path) / ".devflow" / "implementations" / "runs" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    impact_context = _load_impact_context(resolved_plan_path)
    planned_paths, _ = _planned_sources(plan, config.repo_path)
    additional_paths = [
        path for path in _impact_paths(impact_context) if path not in planned_paths
    ]
    preflight_path = run_dir / "preflight.json"
    preflight_path.write_text(json.dumps({
        "planned_paths": planned_paths,
        "impact_paths_not_in_plan": additional_paths,
    }, indent=2), encoding="utf-8")
    if additional_paths:
        print("Implementation preflight found impact files not named by the plan:")
        for path in additional_paths:
            print(f"  - {path}")
        if not _confirm("Add these impact-reviewed files to implementation scope?", auto_approve):
            proposal = {
                "status": "blocked",
                "summary": "Implementation stopped at impact-scope preflight.",
                "replacements": [],
                "deviations": [],
                "questions": ["Review impact files omitted from the approved plan."],
            }
            proposal_path = run_dir / "proposal.json"
            proposal_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
            evidence_path = run_dir / "evidence.json"
            evidence_path.write_text(json.dumps({
                "plan_path": str(resolved_plan_path),
                "applied": False,
                "preflight_stopped": True,
                "impact_paths_not_in_plan": additional_paths,
            }, indent=2), encoding="utf-8")
            return {"proposal": proposal, "applied": False, "paths": {
                "proposal": str(proposal_path.resolve()),
                "preflight": str(preflight_path.resolve()),
                "evidence": str(evidence_path.resolve()),
            }}
    expanded_plan = {
        **plan,
        "proposed_changes": [
            *plan.get("proposed_changes", []),
            *[{"path": path} for path in additional_paths],
        ],
    }
    planned_paths, sources = _planned_sources(expanded_plan, config.repo_path)
    prompt = (
        "Implement the approved plan using exact text replacements. Stay within planned files. "
        "Inspect the supplied bounded source before editing. Preserve existing behavior unless "
        "the plan changes it. If implementation reveals a material product, compatibility, "
        "security, data, or architecture decision, return needs_user_decision with no edits. "
        "Record non-material departures in deviations. Do not use placeholders. Copy old_text "
        "verbatim from the supplied source, include enough surrounding text to make it unique, "
        "and order same-file edits so each old_text exists after earlier edits.\n\n"
        f"PLAN\n{json.dumps(plan, indent=2)}\n\n"
        f"IMPACT CONTEXT AND APPROVED ARCHITECTURE DECISIONS\n"
        f"{json.dumps(impact_context, indent=2)}\n\n"
        f"PLANNED SOURCES\n{json.dumps(sources, indent=2)}"
    )
    model = get_code_review_model(config.model)
    structured = model.with_structured_output(
        ImplementationProposal,
        method="function_calling",
    )
    logger.info("Requesting implementation proposal for %d planned files", len(planned_paths))
    proposal = None
    validation_error = None
    attempt_prompt = prompt
    for attempt in (1, 2):
        proposal = structured.invoke(attempt_prompt).model_dump()
        attempt_path = run_dir / f"proposal-attempt-{attempt}.json"
        attempt_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
        if proposal["status"] != "ready" or not proposal["replacements"]:
            validation_error = None
            break
        try:
            _validate_replacements(proposal, planned_paths, config.repo_path)
            validation_error = None
            break
        except ReplacementValidationError as error:
            validation_error = error
            logger.warning("Implementation proposal attempt %d is not applicable: %s", attempt, error)
            print(f"Implementation proposal attempt {attempt} failed validation: {error}")
            if attempt == 1:
                attempt_prompt = (
                    "Repair the implementation proposal below. The repository has not been "
                    "modified. Return the complete proposal again. Correct the exact replacement "
                    "identified by the deterministic validator; do not broaden scope.\n\n"
                    f"VALIDATION ERROR\n{json.dumps(error.details, indent=2)}\n\n"
                    f"FAILED PROPOSAL\n{json.dumps(proposal, indent=2)}\n\n"
                    + prompt
                )

    assert proposal is not None
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
    diagnostic_path = None
    if validation_error is not None:
        diagnostic_path = run_dir / "validation-error.json"
        diagnostic_path.write_text(json.dumps({
            "message": str(validation_error),
            "details": validation_error.details,
            "proposal": str(proposal_path.resolve()),
            "attempts": 2,
        }, indent=2), encoding="utf-8")
        print("Implementation stopped: repaired proposal still cannot be applied safely.")
        print(f"Validation diagnostic: {diagnostic_path.resolve()}")
    elif proposal["status"] == "ready" and proposal["replacements"]:
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
        "validation_error": (
            {"message": str(validation_error), "details": validation_error.details}
            if validation_error is not None else None
        ),
        "git_diff": subprocess.run(
            ["git", "diff", "--no-ext-diff", "--"], cwd=config.repo_path,
            capture_output=True, text=True, check=True,
        ).stdout,
    }
    evidence_path = run_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    paths = {
        "proposal": str(proposal_path.resolve()),
        "evidence": str(evidence_path.resolve()),
        "preflight": str(preflight_path.resolve()),
    }
    if diagnostic_path is not None:
        paths["validation_error"] = str(diagnostic_path.resolve())
    return {"proposal": proposal, "applied": applied, "paths": paths}
