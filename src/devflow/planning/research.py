from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from devflow.repository_context.config import SerenaContextConfig


MAX_SUPPLEMENTAL_CONTEXT_ROUNDS = 3
MAX_SUPPLEMENTAL_TOOL_CALLS = 8
MAX_PLANNER_FILES = 4
MAX_PLANNER_FILE_CHARS = 12_000
MAX_PLANNER_SOURCE_CHARS = 36_000


def repository_context_questions(plan: dict) -> list[dict[str, str]]:
    if plan.get("status") != "needs_repository_context":
        return []
    return [
        item
        for item in plan.get("outstanding_items", [])
        if item.get("kind") == "repository_context" and item.get("question")
    ]


def user_decision_questions(plan: dict) -> list[dict[str, str]]:
    if plan.get("status") != "needs_user_decision":
        return []
    return [
        item for item in plan.get("outstanding_items", [])
        if item.get("kind") == "user_decision" and item.get("question")
    ]


def question_key(question: str) -> str:
    return " ".join(question.lower().split()).rstrip("?.!")


def supplemental_context_request(
    request: str,
    questions: list[dict[str, str]],
    round_number: int,
) -> str:
    lines = [
        "Perform a targeted supplemental repository investigation for implementation planning.",
        (
            "The prior report is authoritative starting context: do not rediscover its known "
            "files, symbols, or relationships. Investigate only the numbered questions below. "
            "For every question, copy its text exactly into question_resolutions when answered. "
            "If it asks for an exact schema, signature, configuration, or field list, enumerate "
            "the complete requested structure in the resolution and cite the defining source; "
            "do not merely state where it is defined. Keep only that exact question unresolved "
            "when it cannot be answered. Do not revisit prior missing-context items, make product "
            "decisions, or include unrelated discoveries."
        ),
        "",
        f"ORIGINAL DEVELOPMENT REQUEST\n{request}",
        "",
        f"PLANNER REPOSITORY QUESTIONS — ROUND {round_number}",
    ]
    for index, item in enumerate(questions, start=1):
        lines.extend([
            f"{index}. {item['question']}",
            f"   Impact: {item.get('impact', '')}",
            f"   Suggested investigation: {item.get('suggested_action', '')}",
        ])
    return "\n".join(lines)


def supplemental_prior_report(repository_context: dict) -> dict:
    """Retain established evidence without carrying old research questions forward."""
    return {
        key: value
        for key, value in repository_context.items()
        if key not in {"missing_context", "question_resolutions", "supplemental_rounds"}
    } | {
        "missing_context": [],
        "question_resolutions": [],
    }


def supplemental_serena_config(config: SerenaContextConfig) -> SerenaContextConfig:
    return replace(
        config,
        max_rounds=1,
        max_tool_calls_per_round=min(
            config.max_tool_calls_per_round,
            MAX_SUPPLEMENTAL_TOOL_CALLS,
        ),
        max_total_tool_calls=min(
            config.max_total_tool_calls,
            MAX_SUPPLEMENTAL_TOOL_CALLS,
        ),
    )


def normalize_supplemental_report(
    report: dict,
    questions: list[dict[str, str]],
) -> dict:
    """Restrict the supplemental result to the planner's active questions."""
    requested = {question_key(item["question"]): item for item in questions}
    resolutions = []
    resolved_keys = set()
    for resolution in report.get("question_resolutions", []):
        key = question_key(resolution.get("question", ""))
        if key in requested and key not in resolved_keys:
            resolutions.append({
                **resolution,
                "question": requested[key]["question"],
            })
            resolved_keys.add(key)

    unresolved = [
        {
            "kind": "repository",
            "description": item["question"],
            "suggested_action": item.get("suggested_action", "Inspect the defining source."),
            "related_files": [],
            "related_symbols": [],
        }
        for key, item in requested.items()
        if key not in resolved_keys
    ]
    return {
        **report,
        "status": "sufficient" if not unresolved else "needs_repository_context",
        "question_resolutions": resolutions,
        "missing_context": unresolved,
    }


def context_approved_paths(repository_context: dict) -> list[str]:
    ranked: list[tuple[int, str]] = []
    role_rank = {
        "probable_change_target": 0,
        "candidate_change_target": 1,
        "supporting_context": 2,
    }

    def collect(report: dict) -> None:
        for item in report.get("relevant_files", []):
            path = item.get("path")
            if path:
                ranked.append((role_rank.get(item.get("role"), 3), path))
        for resolution in report.get("question_resolutions", []):
            source = resolution.get("source", "").split(":", 1)[0]
            if source:
                ranked.append((0, source))
        for supplemental in report.get("supplemental_rounds", []):
            nested = supplemental.get("report")
            if isinstance(nested, dict):
                collect(nested)

    collect(repository_context)
    unique = []
    seen = set()
    for _, path in sorted(ranked, key=lambda item: item[0]):
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def read_context_approved_files(repo_path: str, repository_context: dict) -> dict[str, str]:
    root = Path(repo_path).resolve()
    excerpts: dict[str, str] = {}
    total_chars = 0
    for relative_path in context_approved_paths(repository_context):
        if len(excerpts) >= MAX_PLANNER_FILES or total_chars >= MAX_PLANNER_SOURCE_CHARS:
            break
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if not candidate.is_file():
            continue
        try:
            content = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        remaining = MAX_PLANNER_SOURCE_CHARS - total_chars
        limit = min(MAX_PLANNER_FILE_CHARS, remaining)
        excerpt = content[:limit]
        if len(content) > limit:
            excerpt += "\n[Devflow truncated this context-approved file]"
        excerpts[relative_path] = excerpt
        total_chars += len(excerpt)
    return excerpts
