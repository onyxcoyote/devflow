from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from devflow.repository_context.config import SerenaContextConfig


MAX_SUPPLEMENTAL_CONTEXT_ROUNDS = 3
MAX_INITIAL_CONTEXT_REFINEMENT_ROUNDS = 3
MAX_SUPPLEMENTAL_TOOL_CALLS = 8
MAX_PLANNER_FILES = 8
MAX_PLANNER_FILE_CHARS = 20_000
MAX_PLANNER_SOURCE_CHARS = 100_000


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


def context_user_questions(report: dict) -> list[dict[str, str]]:
    return [
        {
            "question": item.get("description", ""),
            "impact": item.get("suggested_action", ""),
        }
        for item in report.get("missing_context", [])
        if item.get("kind") == "user_decision" and item.get("description")
    ]


def apply_user_answers_to_context(
    report: dict,
    answers: list[dict[str, str]],
) -> None:
    answer_by_key = {
        question_key(item["question"]): item["answer"] for item in answers
    }
    unresolved = []
    for item in report.get("missing_context", []):
        key = question_key(item.get("description", ""))
        answer = answer_by_key.get(key)
        if item.get("kind") == "user_decision" and answer:
            report.setdefault("question_resolutions", []).append({
                "question": item["description"],
                "resolution": answer,
                "source": "user input",
            })
        else:
            unresolved.append(item)
    report["missing_context"] = unresolved
    report.setdefault("user_answers", []).extend(answers)
    if report.get("status") == "needs_user_decision" and not any(
        item.get("kind") == "user_decision" for item in unresolved
    ):
        report["status"] = (
            "needs_repository_context"
            if any(item.get("kind") == "repository" for item in unresolved)
            else "sufficient"
        )


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


def impact_context_request(request: str) -> tuple[str, list[dict[str, str]]]:
    question = (
        "IMPACT CLOSURE: Trace every changed concept required by this request from its origin or "
        "calculation through concrete object types, construction, callers, mappings, consumers, "
        "serialization/API boundaries, and persistence. Identify all affected definitions and "
        "material architecture decisions."
    )
    questions = [{
        "question": question,
        "impact": "An incomplete impact chain can omit required files or change the wrong type.",
        "suggested_action": "Follow qualified symbols and data/control flow until each chain closes.",
    }]
    return supplemental_context_request(request, questions, 1), questions


def supplemental_prior_report(repository_context: dict) -> dict:
    """Retain established evidence without carrying old research questions forward."""
    prior = {
        key: value
        for key, value in repository_context.items()
        if key not in {"missing_context", "question_resolutions", "supplemental_rounds"}
    } | {
        "missing_context": [],
        "question_resolutions": [],
    }
    prior.setdefault("research_checkpoints", [])
    for supplemental in repository_context.get("supplemental_rounds", []):
        report = supplemental.get("report", {})
        for key in ("relevant_files", "relevant_symbols", "evidence", "research_checkpoints"):
            prior.setdefault(key, []).extend(report.get(key, []))
    return prior


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

    checkpoints = [
        checkpoint for checkpoint in report.get("research_checkpoints", [])
        if question_key(checkpoint.get("original_question", "")) in requested
    ]
    unresolved_checkpoints = [
        checkpoint for checkpoint in checkpoints
        if checkpoint.get("status") == "unresolved"
    ]
    unresolved = [
        {
            "kind": "repository",
            "description": checkpoint["subquestion"],
            "suggested_action": (
                checkpoint.get("next_investigation")
                or "Continue the targeted repository investigation."
            ),
            "related_files": [],
            "related_symbols": checkpoint.get("sources_inspected", []),
        }
        for checkpoint in unresolved_checkpoints
    ]
    checkpointed_questions = {
        question_key(item.get("original_question", "")) for item in checkpoints
    }
    unresolved.extend(
        {
            "kind": "repository",
            "description": item["question"],
            "suggested_action": item.get("suggested_action", "Inspect the defining source."),
            "related_files": [],
            "related_symbols": [],
        }
        for key, item in requested.items()
        if key not in resolved_keys and key not in checkpointed_questions
    )
    return {
        **report,
        "status": "sufficient" if not unresolved else "needs_repository_context",
        "question_resolutions": resolutions,
        "missing_context": unresolved,
        "research_checkpoints": checkpoints,
    }


def supplemental_progress_signature(report: dict) -> tuple:
    return tuple(sorted(
        (
            question_key(item.get("original_question", "")),
            question_key(item.get("subquestion", "")),
            item.get("status", ""),
            item.get("answer", ""),
            item.get("partial_findings", ""),
            tuple(item.get("sources_inspected", [])),
        )
        for item in report.get("research_checkpoints", [])
    ))


def merge_context_refinement(context: dict, report: dict) -> None:
    """Merge a targeted report while making its unresolved set authoritative."""
    for field in (
        "relevant_files", "relevant_symbols", "evidence", "question_resolutions",
        "impact_chains", "architecture_decisions",
    ):
        existing = context.setdefault(field, [])
        seen = {json.dumps(item, sort_keys=True, default=str) for item in existing}
        for item in report.get(field, []):
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in seen:
                existing.append(item)
                seen.add(key)

    checkpoints = {
        (
            question_key(item.get("original_question", "")),
            question_key(item.get("subquestion", "")),
        ): item
        for item in context.get("research_checkpoints", [])
    }
    for item in report.get("research_checkpoints", []):
        checkpoints[(
            question_key(item.get("original_question", "")),
            question_key(item.get("subquestion", "")),
        )] = item
    context["research_checkpoints"] = list(checkpoints.values())
    retained = [
        item for item in context.get("missing_context", [])
        if item.get("kind") != "repository"
    ]
    context["missing_context"] = [*retained, *report.get("missing_context", [])]
    if any(item.get("kind") == "repository" for item in context["missing_context"]):
        context["status"] = "needs_repository_context"
    elif any(item.get("kind") == "user_decision" for item in context["missing_context"]):
        context["status"] = "needs_user_decision"
    else:
        context["status"] = report.get("status", context.get("status"))


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
