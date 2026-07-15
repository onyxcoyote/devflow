from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import SerenaContextConfig

READ_ONLY_SERENA_TOOLS = {
    "find_declaration",
    "find_file",
    "find_implementations",
    "find_referencing_symbols",
    "find_symbol",
    "get_symbols_overview",
    "list_dir",
    "read_file",
    "search_for_pattern",
}

GENERATED_ARTIFACT_PATH = re.compile(r"(^|[\\/])\.devflow([\\/]|$)")
GENERATED_ARTIFACT_EXCLUSION_ARGUMENTS = frozenset({
    "exclude",
    "exclude_glob",
    "excluded_paths",
    "ignored_paths",
    "paths_exclude_glob",
})
ROUND_EXTENSION_CALLS = 6


SERENA_SCHEMA_VERSION = "portable-v1"
SERENA_STRUCTURED_OUTPUT_METHOD = "function_calling"


class SerenaSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchBrief(SerenaSchema):
    interpretation: str = Field(description="Concise interpretation of the requested outcome.")
    terms_and_definitions: list[str] = Field(
        description="Important request terms and tentative definitions; return [] if none."
    )
    assumptions: list[str] = Field(description="Explicit working assumptions; return [] if none.")
    clarification_questions: list[str] = Field(
        description=(
            "Only material questions whose answers could change files, architecture, persistence, "
            "public behavior, or research direction; return [] if none."
        )
    )
    expected_flow: list[str] = Field(
        description="Expected functional or data-flow stages to verify; return [] if none."
    )
    likely_areas: list[str] = Field(description="Likely repository areas and why.")
    excluded_areas: list[str] = Field(
        description="Areas currently believed irrelevant, with reasons; return [] if none."
    )
    research_questions: list[str] = Field(
        description="Three to eight independently answerable questions required for completion."
    )
    completion_evidence: list[str] = Field(
        description="Evidence that would demonstrate the research is complete."
    )


class RelevantFile(SerenaSchema):
    path: str = Field(description="Repository-relative path; maximum 500 characters.")
    role: Literal[
        "probable_change_target",
        "candidate_change_target",
        "supporting_context",
    ]
    reason: str = Field(description="Grounded reason; maximum 800 characters.")
    symbols: list[str] = Field(
        description="Maximum 20 symbols, each at most 300 characters; return [] if empty."
    )


class ContextEvidence(SerenaSchema):
    claim: str = Field(description="Short factual claim; maximum 500 characters.")
    source: str = Field(description="Supporting path or symbol; maximum 500 characters.")


class QuestionResolution(SerenaSchema):
    question: str = Field(description="Resolved question; maximum 1,000 characters.")
    resolution: str = Field(description="Grounded answer; maximum 1,000 characters.")
    source: str = Field(description="Supporting path or symbol; maximum 500 characters.")


class MissingContextItem(SerenaSchema):
    kind: Literal[
        "repository",
        "user_decision",
        "tool_failure",
        "external_information",
    ]
    description: str = Field(description="Missing information; maximum 1,000 characters.")
    suggested_action: str = Field(description="Next action; maximum 1,000 characters.")
    related_files: list[str] = Field(
        description="Maximum 10 repository paths, each at most 500 characters; return [] if empty."
    )
    related_symbols: list[str] = Field(
        description="Maximum 20 symbols, each at most 300 characters; return [] if empty."
    )


class ResearchCheckpoint(SerenaSchema):
    original_question: str = Field(description="The exact active planner question.")
    subquestion: str = Field(description="One independently answerable part of the question.")
    status: Literal["resolved", "unresolved"]
    answer: str = Field(description="Grounded answer, or an empty string when unresolved.")
    partial_findings: str = Field(
        description="Useful grounded progress, or an empty string when none."
    )
    sources_inspected: list[str] = Field(
        description="Inspected repository paths or symbols; return [] if none."
    )
    next_investigation: str = Field(
        description="Specific next retrieval action, or an empty string when resolved."
    )


class ImpactStage(SerenaSchema):
    stage: str = Field(description="Flow stage such as calculation, mapping, API, or persistence.")
    path: str = Field(description="Repository-relative source path, or empty when unresolved.")
    symbol: str = Field(description="Qualified symbol at this stage, or empty when unresolved.")
    object_type: str = Field(description="Concrete qualified type at this stage, or empty.")
    relationship: str = Field(description="How data or control reaches the next stage.")
    status: Literal["resolved", "unresolved", "not_applicable"]


class ImpactChain(SerenaSchema):
    concept: str = Field(description="Changed field, behavior, type, or method being traced.")
    stages: list[ImpactStage] = Field(description="Ordered upstream-to-downstream flow stages.")
    affected_definitions: list[str] = Field(
        description="Qualified path:symbol definitions affected; return [] if none."
    )
    callers: list[str] = Field(description="Affected callers as path:symbol; return [] if none.")
    consumers: list[str] = Field(description="Affected consumers as path:symbol; return [] if none.")
    persistence_effects: list[str] = Field(
        description="Persistence and serialization consequences; return [] if none."
    )
    closure_gaps: list[str] = Field(
        description="Material untraced portions of the flow; return [] only when impact is closed."
    )
    potential_side_effects: list[str] = Field(
        description="Behavioral, state, persistence, or compatibility side effects; return [] if none."
    )


class ArchitectureDecisionCandidate(SerenaSchema):
    kind: Literal[
        "persistence", "schema", "source_of_truth", "new_component", "public_api",
        "compatibility", "security", "state_lifecycle",
    ]
    question: str
    recommendation: str
    alternatives: list[str] = Field(description="Credible alternatives; return [] if none.")
    consequences: list[str] = Field(description="Important tradeoffs and effects.")
    evidence: list[str] = Field(description="Supporting path:symbol evidence.")
    affected_files: list[str] = Field(description="Repository-relative affected files.")


class SerenaContextReport(SerenaSchema):
    status: Literal[
        "sufficient",
        "needs_repository_context",
        "needs_user_decision",
        "blocked",
    ]
    architecture_summary: str = Field(
        description="Concise grounded architecture summary; maximum 3,000 characters."
    )
    relevant_files: list[RelevantFile] = Field(
        description="Maximum 30 grounded files; return [] if empty."
    )
    relevant_symbols: list[str] = Field(
        description="Maximum 50 symbols, each at most 300 characters; return [] if empty."
    )
    evidence: list[ContextEvidence] = Field(
        description="Maximum 30 short evidence entries; return [] if empty."
    )
    question_resolutions: list[QuestionResolution] = Field(
        description="Maximum 10 grounded question resolutions; return [] if empty."
    )
    missing_context: list[MissingContextItem] = Field(
        description="Maximum 10 missing-context items; return [] if empty."
    )
    research_checkpoints: list[ResearchCheckpoint] = Field(
        description=(
            "Targeted supplemental research progress by subquestion; return [] for ordinary "
            "context discovery."
        )
    )
    impact_chains: list[ImpactChain] = Field(
        description="Request-specific impact closure traces; return [] during broad discovery."
    )
    architecture_decisions: list[ArchitectureDecisionCandidate] = Field(
        description="Material architecture changes requiring human confirmation; return [] if none."
    )


class ContextReconciliation(SerenaSchema):
    summary: str = Field(description="Concise explanation of how discoveries change context scope.")
    assumptions_changed: list[str] = Field(
        description="Prior assumptions confirmed, replaced, or invalidated; return [] if none."
    )
    relevant_files: list[RelevantFile] = Field(
        description="Files added or reclassified because of the research answers."
    )
    relevant_symbols: list[str] = Field(description="Newly implicated symbols; return [] if none.")
    evidence: list[ContextEvidence] = Field(description="New short grounded evidence entries.")
    impact_chains: list[ImpactChain] = Field(
        description="New or revised impact chains; return [] if impact is not yet established."
    )
    missing_context: list[MissingContextItem] = Field(
        description="New gaps exposed by reconciling the answers; return [] if none."
    )


class SerenaContextRunError(RuntimeError):
    def __init__(self, message: str, diagnostic_path: str):
        super().__init__(message)
        self.diagnostic_path = diagnostic_path


class _SerenaReportGenerationError(RuntimeError):
    def __init__(self, errors: list[dict[str, Any]]):
        super().__init__("Serena context report generation failed after two attempts")
        self.details = errors


class _DegenerateExplorerOutput(RuntimeError):
    def __init__(self, details: dict[str, Any]):
        super().__init__(
            "Explorer returned repetitive output without Serena tool calls"
        )
        self.details = details


class _ModelRequestLimiter:
    def __init__(self, minimum_interval_seconds: float, input_target_tokens: int | None = None):
        self.minimum_interval_seconds = minimum_interval_seconds
        self.last_started_at: float | None = None
        self.request_count = 0
        self.wait_count = 0
        self.total_wait_seconds = 0.0
        self.input_target_tokens = input_target_tokens
        self.estimated_input_tokens = 0
        self.actual_input_tokens = 0
        self.actual_output_tokens = 0

    @staticmethod
    def _estimated_tokens(value: Any) -> int:
        return max(1, int(len(str(value)) / 3.5))

    async def invoke(self, model, messages, purpose: str):
        if not 10 <= len(purpose) <= 50:
            raise ValueError("Model-call purpose must be 10-50 characters")
        if self.last_started_at is not None:
            wait_seconds = max(
                0.0,
                self.minimum_interval_seconds - (perf_counter() - self.last_started_at),
            )
            if wait_seconds > 0:
                self.wait_count += 1
                self.total_wait_seconds += wait_seconds
                await asyncio.sleep(wait_seconds)
        self.last_started_at = perf_counter()
        self.request_count += 1
        print(f"LLM call {self.request_count}: {purpose}")
        estimated = self._estimated_tokens(messages)
        self.estimated_input_tokens += estimated
        if self.input_target_tokens is not None:
            print(
                f"  Estimated context: {estimated:,}/{self.input_target_tokens:,} input tokens"
            )
        try:
            response = await model.ainvoke(messages)
            usage = getattr(response, "usage_metadata", None) or {}
            self.actual_input_tokens += int(
                usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            )
            self.actual_output_tokens += int(
                usage.get("output_tokens") or usage.get("completion_tokens") or 0
            )
            return response
        except Exception as error:
            print(f"LLM ERROR ({purpose}): {type(error).__name__}: {error}")
            raise


def _confirm_additional_context_round(
    report: dict[str, Any],
    *,
    auto_approve: bool,
    review_path: Path | None = None,
) -> str:
    gaps = [
        item for item in report.get("missing_context", [])
        if item.get("kind") == "repository"
    ]
    print("Repository gaps requesting another context round:")
    for index, item in enumerate(gaps, start=1):
        print(f"  {index}. {item.get('description', '')}")
    if auto_approve:
        print("Additional context round auto-approved")
        return "continue"
    if not sys.stdin.isatty():
        print("Additional context round declined: stdin is not interactive")
        return "stop"
    while True:
        answer = input(
            "[C]ontinue context research, add a research [H]int, [O]pen context for "
            "human review, [P]roceed anyway to planning, or [S]top? [S]: "
        ).strip().lower()
        choice = {
            "c": "continue", "continue": "continue",
            "h": "hint", "hint": "hint",
            "o": "open", "open": "open",
            "p": "proceed", "proceed": "proceed",
            "s": "stop", "stop": "stop", "": "stop",
        }.get(answer)
        if choice == "open":
            if review_path is not None:
                review_path.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"Context review: {review_path.resolve()}")
                try:
                    subprocess.Popen(["xdg-open", str(review_path.resolve())])
                except OSError:
                    webbrowser.open(review_path.resolve().as_uri())
            continue
        if choice == "hint":
            for item in gaps:
                hint = input(
                    f"Optional search hint for: {item.get('description', '')}\n> "
                ).strip()
                if hint:
                    item["suggested_action"] = (
                        item.get("suggested_action", "")
                        + f" User search hint (not evidence): {hint}"
                    )
            return "continue"
        if choice:
            return choice


def _print_context_progress(report: dict[str, Any]) -> None:
    def concise(value: Any) -> str:
        text = str(value)
        return text[:500] + ("..." if len(text) > 500 else "")

    resolutions = report.get("question_resolutions", [])
    if resolutions:
        print("Resolved context:")
    for item in resolutions:
        print(f"  - {concise(item.get('question', ''))}: {concise(item.get('resolution', ''))}")
        print(f"    Source: {item.get('source', '')}")
    for checkpoint in report.get("research_checkpoints", []):
        label = "Resolved" if checkpoint.get("status") == "resolved" else "Partial"
        print(f"{label}: {checkpoint.get('subquestion', '')}")
        detail = checkpoint.get("answer") or checkpoint.get("partial_findings")
        if detail:
            print(f"  {concise(detail)}")
        if checkpoint.get("sources_inspected"):
            print(f"  Inspected: {', '.join(checkpoint['sources_inspected'])}")
        if checkpoint.get("next_investigation"):
            print(f"  Next: {checkpoint['next_investigation']}")


def _round_extension_choice(
    events: list[dict[str, Any]],
    transcript_path: Path,
    *,
    auto_approve: bool,
) -> str:
    if auto_approve:
        return "extend"
    if not sys.stdin.isatty():
        return "report"
    while True:
        answer = input(
            "Soft round limit reached while Serena is still researching. "
            "[E]xtend by 6 calls, create [R]eport now, [O]pen transcript, or [S]top? [R]: "
        ).strip().lower()
        choice = {
            "e": "extend", "extend": "extend",
            "r": "report", "report": "report", "": "report",
            "o": "open", "open": "open",
            "s": "stop", "stop": "stop",
        }.get(answer)
        if choice == "open":
            transcript_path.write_text(
                json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Live transcript: {transcript_path.resolve()}")
            try:
                subprocess.Popen(["xdg-open", str(transcript_path.resolve())])
            except OSError:
                webbrowser.open(transcript_path.resolve().as_uri())
            continue
        if choice:
            return choice


def _langchain_tools(mcp_tools) -> list[dict[str, Any]]:
    tools = []
    for tool in mcp_tools:
        if tool.name not in READ_ONLY_SERENA_TOOLS:
            continue
        schema = (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or {"type": "object"}
        )
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "Serena read-only retrieval tool",
                "parameters": schema,
            },
        })
    return tools


def _references_generated_artifacts(value: Any) -> bool:
    if isinstance(value, str):
        return GENERATED_ARTIFACT_PATH.search(value) is not None
    if isinstance(value, dict):
        return any(
            key not in GENERATED_ARTIFACT_EXCLUSION_ARGUMENTS
            and _references_generated_artifacts(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_references_generated_artifacts(item) for item in value)
    return False


def _is_generated_artifact_string(value: Any) -> bool:
    return isinstance(value, str) and _references_generated_artifacts(value)


def _without_generated_artifacts(value: Any) -> Any:
    if isinstance(value, str):
        kept_lines = [
            line for line in value.splitlines()
            if not _references_generated_artifacts(line)
        ]
        return "\n".join(kept_lines)
    if isinstance(value, list):
        return [
            _without_generated_artifacts(item)
            for item in value
            if not _is_generated_artifact_string(item)
        ]
    if isinstance(value, dict):
        return {
            key: _without_generated_artifacts(item)
            for key, item in value.items()
            if not _is_generated_artifact_string(key)
            and not _is_generated_artifact_string(item)
        }
    return value


def _tool_result_text(result, max_chars: int) -> str:
    structured = (
        getattr(result, "structuredContent", None)
        or getattr(result, "structured_content", None)
    )
    if structured is not None:
        text = json.dumps(
            _without_generated_artifacts(structured),
            ensure_ascii=False,
            default=str,
        )
    else:
        blocks = []
        for block in getattr(result, "content", []):
            block_text = getattr(block, "text", None)
            blocks.append(block_text if block_text is not None else str(block))
        text = _without_generated_artifacts("\n".join(blocks))
    return text[:max_chars]


def _call_signature(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps([name, arguments], sort_keys=True, ensure_ascii=False, default=str)


def _repetition_score(value: str) -> float:
    words = re.findall(r"\w+", value.lower())
    if len(words) < 40:
        return 0.0
    grams = [tuple(words[index:index + 4]) for index in range(len(words) - 3)]
    return 1.0 - (len(set(grams)) / len(grams))


def _is_degenerate_explorer_output(value: str) -> bool:
    if len(value) < 500:
        return False
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    repeated_lines = len(lines) >= 8 and len(set(lines)) / len(lines) < 0.35
    return repeated_lines or _repetition_score(value) >= 0.72


def _is_unscoped_pattern_search(name: str, arguments: dict[str, Any]) -> bool:
    if name != "search_for_pattern":
        return False
    scope_keys = (
        "relative_path", "path", "paths_include_glob", "include_glob", "file_mask"
    )
    return not any(arguments.get(key) not in (None, "", ".", "**/*") for key in scope_keys)


def _question_key(value: str) -> str:
    return " ".join(value.lower().split()).rstrip("?.!")


_FILE_ROLE_RANK = {
    "supporting_context": 0,
    "candidate_change_target": 1,
    "probable_change_target": 2,
}


def _merge_relevant_files(target: list[dict[str, Any]], additions: list[dict[str, Any]]) -> dict[str, int]:
    by_path = {item.get("path"): item for item in target if item.get("path")}
    added = 0
    upgraded = 0
    for addition in additions:
        path = addition.get("path")
        if not path:
            continue
        current = by_path.get(path)
        if current is None:
            copied = dict(addition)
            target.append(copied)
            by_path[path] = copied
            added += 1
            continue
        if _FILE_ROLE_RANK.get(addition.get("role"), -1) > _FILE_ROLE_RANK.get(current.get("role"), -1):
            current["role"] = addition["role"]
            upgraded += 1
        current["symbols"] = list(dict.fromkeys([
            *current.get("symbols", []), *addition.get("symbols", [])
        ]))
        new_reason = addition.get("reason", "")
        if new_reason and new_reason not in current.get("reason", ""):
            current["reason"] = (current.get("reason", "") + " | " + new_reason).strip(" |")[:800]
    return {"added": added, "upgraded": upgraded}


def _merge_unique(target: list[Any], additions: list[Any]) -> int:
    seen = {json.dumps(item, sort_keys=True, default=str) for item in target}
    added = 0
    for item in additions:
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            target.append(item)
            seen.add(key)
            added += 1
    return added


def _apply_reconciliation(report: dict[str, Any], delta: dict[str, Any]) -> dict[str, int]:
    file_delta = _merge_relevant_files(
        report.setdefault("relevant_files", []), delta.get("relevant_files", [])
    )
    symbols_added = _merge_unique(
        report.setdefault("relevant_symbols", []), delta.get("relevant_symbols", [])
    )
    _merge_unique(report.setdefault("evidence", []), delta.get("evidence", []))
    _merge_unique(report.setdefault("impact_chains", []), delta.get("impact_chains", []))
    gaps_added = _merge_unique(
        report.setdefault("missing_context", []), delta.get("missing_context", [])
    )
    report["reconciliation"] = {
        "summary": delta.get("summary", ""),
        "assumptions_changed": delta.get("assumptions_changed", []),
        "files_added": file_delta["added"],
        "roles_upgraded": file_delta["upgraded"],
        "symbols_added": symbols_added,
        "new_gaps": gaps_added,
    }
    if gaps_added and report.get("status") == "sufficient":
        report["status"] = "needs_repository_context"
    return report["reconciliation"]


def _apply_brief_coverage(report: dict[str, Any], brief: dict[str, Any]) -> list[dict[str, Any]]:
    resolutions = {
        _question_key(item.get("question", "")): item
        for item in report.get("question_resolutions", [])
    }
    ledger = [
        {
            "question": item["question"],
            "status": item["status"],
            "answer": item.get("answer", ""),
            "source": "human" if item["status"] == "human_answered" else "research brief",
        }
        for item in brief.get("clarification_answers", [])
    ]
    for question in brief.get("research_questions", []):
        resolution = resolutions.get(_question_key(question))
        if resolution:
            source = resolution.get("source", "")
            source_path = source.split(":", 1)[0]
            path_like_source = "/" in source_path or "." in Path(source_path).name
            source_reconciled = not path_like_source or any(
                item.get("path") == source_path for item in report.get("relevant_files", [])
            )
            ledger.append({
                "question": question,
                "status": "self_answered" if source_reconciled else "answered_unreconciled",
                "answer": resolution.get("resolution", ""),
                "source": source,
            })
            if not source_reconciled:
                report.setdefault("missing_context", []).append({
                    "kind": "repository",
                    "description": f"Reconcile the implications of the answer to: {question}",
                    "suggested_action": (
                        f"Add or classify {source_path} and trace affected definitions, callers, "
                        "consumers, types, mappings, and persistence effects."
                    ),
                    "related_files": [source_path],
                    "related_symbols": [],
                })
            continue
        ledger.append({"question": question, "status": "unresolved", "answer": "", "source": ""})
        if not any(
            _question_key(item.get("description", "")) == _question_key(question)
            for item in report.get("missing_context", [])
        ):
            report.setdefault("missing_context", []).append({
                "kind": "repository",
                "description": question,
                "suggested_action": "Answer this research-brief coverage question with evidence.",
                "related_files": [],
                "related_symbols": [],
            })
    if any(item["status"] in {"unresolved", "answered_unreconciled"} for item in ledger) and report.get("status") == "sufficient":
        report["status"] = "needs_repository_context"
    report["research_brief"] = brief
    report["inquiry_ledger"] = ledger
    return ledger


def _clarify_research_brief(brief: dict[str, Any], *, auto_approve: bool) -> None:
    questions = brief.get("clarification_questions", [])
    answers = []
    if not questions:
        return
    print(f"Research brief found {len(questions)} material clarification question(s).")
    for question in questions:
        print(f"  Question: {question}")
        if auto_approve or not sys.stdin.isatty():
            answers.append({"question": question, "status": "assumed", "answer": ""})
            continue
        while True:
            action = input("[A]nswer, [S]kip with stated assumption, [D]ocumentation hint, or [R]esearch? [R]: ").strip().lower()
            if action in {"a", "answer"}:
                answer = input("Answer:\n> ").strip()
                if answer:
                    answers.append({"question": question, "status": "human_answered", "answer": answer})
                    break
            elif action in {"s", "skip"}:
                assumption = input("Assumption to use (blank keeps the brief assumption):\n> ").strip()
                answers.append({"question": question, "status": "assumed", "answer": assumption})
                break
            elif action in {"d", "documentation"}:
                hint = input("Documentation path or guidance:\n> ").strip()
                if hint:
                    answers.append({"question": question, "status": "human_answered", "answer": hint})
                    break
            else:
                answers.append({"question": question, "status": "researching", "answer": ""})
                break
    brief["clarification_answers"] = answers


async def _create_research_brief(
    request: str,
    model,
    limiter: _ModelRequestLimiter,
    *,
    auto_approve: bool,
) -> dict[str, Any]:
    print("Context stage: request interpretation")
    brief_model = model.with_structured_output(
        ResearchBrief, method=SERENA_STRUCTURED_OUTPUT_METHOD
    )
    brief = await limiter.invoke(
        brief_model,
        (
            "Prepare a concise research brief before using repository tools. Interpret the "
            "development request, expose assumptions, identify only material clarification "
            "questions, and define three to eight independently answerable research questions. "
            "Do not claim repository facts and do not propose implementation details as facts. "
            "Prefer asking for clarification when different meanings would send research into "
            "different application areas.\n\nDEVELOPMENT REQUEST\n" + request
        ),
        "Create repository research brief",
    )
    result = brief.model_dump()
    _clarify_research_brief(result, auto_approve=auto_approve)
    print(
        f"Research brief: questions={len(result['research_questions'])} "
        f"clarifications={len(result['clarification_questions'])}"
    )
    return result


def _should_continue(
    report: dict[str, Any],
    *,
    round_number: int,
    total_tool_calls: int,
    config: SerenaContextConfig,
) -> bool:
    kinds = {item["kind"] for item in report["missing_context"]}
    return (
        "repository" in kinds
        and "tool_failure" not in kinds
        and round_number < config.max_rounds
        and total_tool_calls < config.max_total_tool_calls
    )


def _round_focus_instruction(is_final_round: bool) -> str:
    if not is_final_round:
        return (
            "Focus this round on repository-answerable gaps from the prior report. Preserve "
            "genuine requirement ambiguities as user decisions rather than inventing an answer."
        )
    return (
        "This is the final available context round. Resolve specific repository gaps from the "
        "prior report before exploring anything new. If a relevant file path is already known, "
        "inspect that file first. If a named symbol is missing, locate and inspect it. Follow "
        "schema and type references discovered in those files. Do not investigate user-decision "
        "items; preserve them for the human. Do not broaden the search unless resolving a listed "
        "gap requires it."
    )


def _bounded_transcript(
    events: list[dict[str, Any]],
    max_chars: int,
) -> tuple[str, int]:
    included: list[dict[str, Any]] = []
    for event in events:
        candidate = json.dumps([*included, event], ensure_ascii=False, default=str)
        if len(candidate) > max_chars:
            break
        included.append(event)
    text = json.dumps(included, ensure_ascii=False, default=str)
    if len(included) < len(events):
        text += (
            f"\n[Devflow included {len(included)} of {len(events)} complete events "
            "because of the transcript limit]"
        )
    return text, len(included)


async def _explore_round(
    request: str,
    config: SerenaContextConfig,
    session,
    model,
    report_model,
    request_limiter: _ModelRequestLimiter,
    tools: list[dict[str, Any]],
    *,
    round_number: int,
    prior_report: dict[str, Any] | None,
    executed_signatures: set[str],
    tool_call_budget: int,
    max_tool_call_budget: int,
    is_final_round: bool,
    active_questions: list[str] | None = None,
    research_brief: dict[str, Any] | None = None,
    auto_approve: bool = False,
    live_transcript_path: Path | None = None,
):
    try:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    except ImportError as error:
        raise ValueError("LangChain is required; reinstall devflow") from error
    explorer = model.bind_tools(tools)
    prior_text = json.dumps(prior_report, ensure_ascii=False) if prior_report else "None"
    messages = [
        SystemMessage(content=(
            "You are exploring a code repository to identify context for an implementation plan. "
            "Use Serena's semantic retrieval tools. Do not propose edits and do not claim a path, "
            "symbol, or relationship unless a tool result supports it. Start broad, then follow "
            "definitions and references. Before stopping with repository context missing, attempt "
            "to locate each named symbol and inspect known relevant files. Distinguish information "
            "that is unavailable from information that has not yet been investigated. Avoid "
            "repeating completed calls from earlier rounds."
            " Treat .devflow as generated workflow output: never search, inspect, cite, or use "
            "files beneath it as repository evidence."
        )),
        HumanMessage(content=(
            f"Development request:\n{request}\n\n"
            f"Approved research brief:\n{json.dumps(research_brief or {}, ensure_ascii=False)}\n\n"
            f"Exploration round: {round_number}\n"
            f"Prior grounded report:\n{prior_text}\n\n"
            f"{_round_focus_instruction(is_final_round)}"
        )),
    ]
    events: list[dict[str, Any]] = []
    calls_attempted = 0
    tool_result_chars = 0
    active_budget = tool_call_budget
    stop_round = False
    consecutive_tool_timeouts = 0
    live_transcript_path = live_transcript_path or Path("serena-live-transcript.json")

    while calls_attempted < max_tool_call_budget and not stop_round:
        purpose = (
            f"Choose repository searches (round {round_number})"
            if not events
            else "Review tool results and continue"
        )
        response = await request_limiter.invoke(explorer, messages, purpose)
        messages.append(response)
        tool_calls = response.tool_calls or []
        if not tool_calls:
            summary = str(response.content)
            if _is_degenerate_explorer_output(summary):
                details = {
                    "round": round_number,
                    "response_chars": len(summary),
                    "repetition_score": round(_repetition_score(summary), 3),
                    "response_preview": summary[:500],
                    "live_transcript": str(live_transcript_path.resolve()),
                }
                events.append({"degenerate_explorer_output": details})
                live_transcript_path.write_text(
                    json.dumps(events, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                raise _DegenerateExplorerOutput(details)
            events.append({"assistant_summary": summary[:1_000]})
            break
        for call_index, call in enumerate(tool_calls):
            while calls_attempted >= active_budget and active_budget < max_tool_call_budget:
                choice = _round_extension_choice(
                    events, live_transcript_path, auto_approve=auto_approve
                )
                if choice == "extend":
                    active_budget = min(
                        active_budget + ROUND_EXTENSION_CALLS,
                        max_tool_call_budget,
                    )
                    print(f"Extended current context round to {active_budget} tool calls")
                    break
                stop_round = True
                break
            if stop_round or calls_attempted >= active_budget:
                events.append({
                    "round": round_number,
                    "pending_tool_calls": [
                        {"tool": pending.get("name"), "arguments": pending.get("args", {})}
                        for pending in tool_calls[call_index:]
                    ],
                    "reason": "round_limit_not_extended",
                })
                break
            name = call["name"]
            if name not in READ_ONLY_SERENA_TOOLS:
                continue
            arguments = call.get("args", {})
            argument_hint = next(
                (str(arguments[key]) for key in (
                    "relative_path", "path", "name_path", "name", "substring_pattern"
                ) if arguments.get(key)),
                "",
            )
            print(
                f"Serena tool {calls_attempted + 1}/{max_tool_call_budget}: "
                f"{name}{' ' + argument_hint[:120] if argument_hint else ''}"
            )
            if _references_generated_artifacts(arguments):
                result_text = (
                    "Devflow did not execute this call because .devflow contains generated "
                    "workflow artifacts, not repository source."
                )
                events.append({
                    "round": round_number,
                    "tool": name,
                    "arguments": arguments,
                    "blocked_generated_artifact": True,
                    "result": result_text,
                })
                messages.append(ToolMessage(
                    content=result_text,
                    tool_call_id=call["id"],
                ))
                calls_attempted += 1
                continue
            if _is_unscoped_pattern_search(name, arguments):
                result_text = (
                    "Devflow did not execute this repository-wide pattern search. Narrow it with "
                    "relative_path or a bounded inclusion glob."
                )
                events.append({
                    "round": round_number,
                    "tool": name,
                    "arguments": arguments,
                    "blocked_unscoped_search": True,
                    "result": result_text,
                })
                messages.append(ToolMessage(content=result_text, tool_call_id=call["id"]))
                calls_attempted += 1
                continue
            signature = _call_signature(name, arguments)
            if signature in executed_signatures:
                result_text = (
                    "Devflow did not execute this exact duplicate call. Use the prior result or "
                    "refine the arguments."
                )
                events.append({
                    "round": round_number,
                    "tool": name,
                    "arguments": arguments,
                    "duplicate": True,
                    "result": result_text,
                })
            else:
                executed_signatures.add(signature)
                started_at = perf_counter()
                try:
                    result = await session.call_tool(name, arguments=arguments)
                    call_timeout = False
                except (TimeoutError, asyncio.TimeoutError):
                    result = None
                    call_timeout = True
                elapsed = round(perf_counter() - started_at, 3)
                remaining_chars = max(
                    0, config.max_round_tool_result_chars - tool_result_chars
                )
                if remaining_chars == 0:
                    events.append({
                        "round": round_number,
                        "tool_result_budget_reached": True,
                        "tool_result_chars": tool_result_chars,
                    })
                    stop_round = True
                    break
                result_text = (
                    "Serena tool call timed out. Narrow the next retrieval operation."
                    if call_timeout else _tool_result_text(
                        result, min(config.max_tool_result_chars, remaining_chars)
                    )
                )
                is_error = call_timeout or bool(
                    getattr(result, "isError", False)
                    or getattr(result, "is_error", False)
                )
                timed_out = call_timeout or (is_error and "timeout" in result_text.lower())
                consecutive_tool_timeouts = (
                    consecutive_tool_timeouts + 1 if timed_out else 0
                )
                tool_result_chars += len(result_text)
                events.append({
                    "round": round_number,
                    "tool": name,
                    "arguments": arguments,
                    "duplicate": False,
                    "result": result_text,
                    "elapsed_seconds": elapsed,
                    "is_error": is_error,
                    "timed_out": timed_out,
                })
            messages.append(ToolMessage(
                content=result_text,
                tool_call_id=call["id"],
            ))
            calls_attempted += 1
            if consecutive_tool_timeouts >= 2:
                events.append({
                    "round": round_number,
                    "reason": "consecutive_tool_timeouts",
                    "consecutive_timeouts": consecutive_tool_timeouts,
                })
                stop_round = True
                break
        if stop_round:
            break
        if calls_attempted >= active_budget and active_budget < max_tool_call_budget:
            choice = _round_extension_choice(
                events, live_transcript_path, auto_approve=auto_approve
            )
            if choice == "extend":
                active_budget = min(
                    active_budget + ROUND_EXTENSION_CALLS,
                    max_tool_call_budget,
                )
                print(f"Extended current context round to {active_budget} tool calls")
            else:
                break

    live_transcript_path.write_text(
        json.dumps(events, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    structured_report_model = report_model.with_structured_output(
        SerenaContextReport,
        method=SERENA_STRUCTURED_OUTPUT_METHOD,
    )
    reconciliation_model = report_model.with_structured_output(
        ContextReconciliation,
        method=SERENA_STRUCTURED_OUTPUT_METHOD,
    )
    transcript_text, included_events = _bounded_transcript(
        events,
        config.max_transcript_chars,
    )
    supplemental_instruction = ""
    if active_questions:
        supplemental_instruction = (
            "\nThis is targeted supplemental research. Treat PRIOR REPORT as established "
            "evidence and report only answers to ACTIVE QUESTIONS. Copy each question exactly "
            "into question_resolutions. Decompose compound or conditional questions into "
            "independently answerable research_checkpoints. Preserve established checkpoint "
            "answers from PRIOR REPORT and investigate only unresolved subquestions. Record "
            "partial_findings, sources_inspected, and one specific next_investigation when a "
            "subquestion remains unresolved. For requests for an entire schema, signature, field "
            "list, or configuration, provide the complete requested structure in resolution; "
            "a file location alone is not an answer. Do not retain or investigate unrelated "
            "missing context from prior work.\nACTIVE QUESTIONS\n"
            + json.dumps(active_questions, ensure_ascii=False)
            + "\n"
        )
        if any("IMPACT CLOSURE" in question for question in active_questions):
            supplemental_instruction += (
                "\nPerform explicit impact closure. Trace each changed concept from origin or "
                "calculation through concrete qualified object types, construction, mappings, "
                "callers, consumers, API/serialization, and persistence. Populate impact_chains. "
                "Do not mark a chain closed while a material stage is unknown. Put every closure "
                "gap in missing_context. Populate architecture_decisions for persistence, schema, "
                "source-of-truth, new-component, public-contract, compatibility, security, or "
                "state-lifecycle choices.\n"
            )
    report_prompt = (
        "Create an accumulated grounded repository-context report for the development request. "
        "Merge the prior report with this round, retaining useful earlier evidence. Include only "
        "paths and symbols supported by tool results. Classify every missing item as repository, "
        "user_decision, tool_failure, or external_information. Use needs_repository_context only "
        "for gaps that another Serena round could answer; use needs_user_decision for genuine "
        "requirement ambiguity; use blocked for tool failures; otherwise use sufficient. "
        "Classify every relevant file as probable_change_target, candidate_change_target, "
        "or supporting_context. A probable change target requires repository evidence tracing "
        "the requested behavior or data through that file; do not mark a file probable merely "
        "because its name or contents look plausible. Use candidate_change_target for plausible "
        "files whose ownership has not been established, and supporting_context for files needed to "
        "understand the implementation but not currently expected to change. Relevant files are "
        "research inputs, not automatic modification instructions. "
        "When a question or ambiguity from the prior report is no longer missing, include a short "
        "question_resolutions entry explaining the question, its grounded resolution, and the file "
        "path or symbol that supports it. If there is no repository source for the resolution, keep "
        "the question in missing_context rather than guessing or silently dropping it. "
        "Be concise. Do not copy source code, tool results, or the transcript into the report. "
        "Represent evidence only as short factual claims paired with file paths or symbol names. "
        "Keep the complete report comfortably below the output-token limit.\n"
        "For every research question in the approved research brief, copy the question exactly "
        "into question_resolutions with grounded evidence when answered, or keep it in "
        "missing_context. Do not declare sufficient while a research question is uncovered.\n"
        f"{supplemental_instruction}\n"
        f"DEVELOPMENT REQUEST\n{request}\n\n"
        f"PRIOR REPORT\n{prior_text}\n\n"
        f"CURRENT ROUND TRANSCRIPT\n{transcript_text}"
    )
    attempt_errors: list[dict[str, Any]] = []
    prompts = [
        report_prompt,
        (
            "The previous report-formatting attempt failed or exceeded its output limit. "
            "Retry once with an especially compact report. Do not reproduce code or transcript "
            "content. Include only the most important grounded files, symbols, short evidence, "
            "and unresolved items.\n\n" + report_prompt
        ),
    ]
    for attempt, prompt in enumerate(prompts, start=1):
        try:
            purpose = (
                "Create grounded context report"
                if attempt == 1
                else "Retry compact context report"
            )
            report = await request_limiter.invoke(structured_report_model, prompt, purpose)
            report_dict = report.model_dump()
            print("Context stage: reconcile research impact")
            reconciliation = await request_limiter.invoke(
                reconciliation_model,
                (
                    "Reconcile the grounded research answers into implementation scope. Identify "
                    "what the discoveries change about relevant files, concrete types, callers, "
                    "consumers, mappings, serialization, persistence, and state lifecycle. Return "
                    "only deltas supported by the report or current transcript. Reclassify existing "
                    "files when evidence strengthens their role. Expose new repository gaps instead "
                    "of guessing. Do not duplicate files merely to change their role or reason.\n\n"
                    f"DEVELOPMENT REQUEST\n{request}\n\n"
                    f"RESEARCH BRIEF\n{json.dumps(research_brief or {}, ensure_ascii=False)}\n\n"
                    f"CURRENT REPORT\n{json.dumps(report_dict, ensure_ascii=False)}\n\n"
                    f"CURRENT ROUND TRANSCRIPT\n{transcript_text}"
                ),
                "Reconcile answers into context scope",
            )
            reconciliation_stats = _apply_reconciliation(
                report_dict, reconciliation.model_dump()
            )
            _apply_brief_coverage(report_dict, research_brief or {})
            print(
                "Answer reconciled: "
                f"files added={reconciliation_stats['files_added']}, "
                f"roles upgraded={reconciliation_stats['roles_upgraded']}, "
                f"symbols added={reconciliation_stats['symbols_added']}, "
                f"new gaps={reconciliation_stats['new_gaps']}"
            )
            return report_dict, events, calls_attempted, attempt_errors
        except Exception as error:
            attempt_errors.append({
                "attempt": attempt,
                "type": type(error).__name__,
                "message": str(error),
                "round": round_number,
                "transcript_events_included": included_events,
                "transcript_events_total": len(events),
            })

    raise _SerenaReportGenerationError(attempt_errors)


async def _explore_with_session(
    request: str,
    config: SerenaContextConfig,
    session,
    *,
    initial_report: dict[str, Any] | None = None,
    active_questions: list[str] | None = None,
    gate_between_rounds: bool = False,
    auto_approve: bool = False,
    live_transcript_path: Path | None = None,
):
    try:
        from devflow.code_review.models import get_code_review_model
    except ImportError as error:
        raise ValueError("LangChain is required; reinstall devflow") from error

    await session.initialize()
    listed = await session.list_tools()
    tools = _langchain_tools(listed.tools)
    if not tools:
        raise ValueError("Serena exposed no permitted read-only retrieval tools")

    model = get_code_review_model(
        config.model,
        max_output_tokens=config.max_explorer_output_tokens,
    )
    report_model = get_code_review_model(
        config.model,
        max_output_tokens=config.max_report_output_tokens,
    )
    executed_signatures: set[str] = set()
    all_events: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    total_tool_calls = 0
    report: dict[str, Any] | None = initial_report
    report_errors: list[dict[str, Any]] = []
    request_limiter = _ModelRequestLimiter(
        config.model_request_min_interval_seconds,
        config.context_input_target_tokens,
    )
    if active_questions:
        research_brief = {
            "interpretation": "Targeted supplemental repository research.",
            "terms_and_definitions": [],
            "assumptions": [],
            "clarification_questions": [],
            "expected_flow": [],
            "likely_areas": [],
            "excluded_areas": [],
            "research_questions": list(active_questions),
            "completion_evidence": ["A grounded resolution and source for every active question."],
        }
    elif initial_report and initial_report.get("research_brief"):
        research_brief = initial_report["research_brief"]
    else:
        research_brief = await _create_research_brief(
            request, model, request_limiter, auto_approve=auto_approve
        )

    for round_number in range(1, config.max_rounds + 1):
        remaining = config.max_total_tool_calls - total_tool_calls
        if remaining <= 0:
            break
        round_budget = min(config.max_tool_calls_per_round, remaining)
        is_final_round = (
            round_number == config.max_rounds
            or remaining <= round_budget
        )
        report, events, calls, round_report_errors = await _explore_round(
            request,
            config,
            session,
            model,
            report_model,
            request_limiter,
            tools,
            round_number=round_number,
            prior_report=report,
            executed_signatures=executed_signatures,
            tool_call_budget=round_budget,
            max_tool_call_budget=remaining,
            is_final_round=is_final_round,
            active_questions=active_questions,
            research_brief=research_brief,
            auto_approve=auto_approve,
            live_transcript_path=live_transcript_path,
        )
        total_tool_calls += calls
        all_events.extend(events)
        reports.append(report)
        report_errors.extend(round_report_errors)
        print(
            f"Round {round_number}: {calls} tool calls "
            f"({total_tool_calls}/{config.max_total_tool_calls} total)"
        )
        _print_context_progress(report)
        continue_rounds = _should_continue(
            report,
            round_number=round_number,
            total_tool_calls=total_tool_calls,
            config=config,
        )
        if continue_rounds and gate_between_rounds:
            round_action = _confirm_additional_context_round(
                report,
                auto_approve=auto_approve,
                review_path=(
                    live_transcript_path.with_name(f"context-review-round-{round_number}.json")
                    if live_transcript_path is not None else None
                ),
            )
            if round_action == "proceed":
                report["context_control"] = {"proceed_anyway": True}
            elif round_action == "stop":
                report["context_control"] = {"stop_requested": True}
            continue_rounds = round_action == "continue"
        if not continue_rounds:
            break

    if report is None:
        raise ValueError("Serena exploration ended before producing a context report")
    return (
        report,
        all_events,
        reports,
        [tool["function"]["name"] for tool in tools],
        total_tool_calls,
        report_errors,
        request_limiter,
    )


async def _run_serena(
    request: str,
    config: SerenaContextConfig,
    stderr_log,
    *,
    initial_report: dict[str, Any] | None = None,
    active_questions: list[str] | None = None,
    gate_between_rounds: bool = False,
    auto_approve: bool = False,
    live_transcript_path: Path | None = None,
):
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as error:
        raise ValueError("The MCP Python package is required; reinstall devflow") from error

    parameters = StdioServerParameters(
        command=config.command,
        args=list(config.args),
    )
    async with stdio_client(parameters, errlog=stderr_log) as (read, write):
        async with ClientSession(read, write) as session:
            return await _explore_with_session(
                request,
                config,
                session,
                initial_report=initial_report,
                active_questions=active_questions,
                gate_between_rounds=gate_between_rounds,
                auto_approve=auto_approve,
                live_transcript_path=live_transcript_path,
            )


def run_serena_context(
    request: str,
    config: SerenaContextConfig,
    *,
    initial_report: dict[str, Any] | None = None,
    active_questions: list[str] | None = None,
    gate_between_rounds: bool = False,
    auto_approve: bool = False,
) -> dict[str, Any]:
    repo_path = getattr(config, "repo_path", ".")

    def git_output(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=repo_path, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    branch = git_output("branch", "--show-current")
    head_commit = git_output("rev-parse", "HEAD")
    git_status = git_output("status", "--short")
    print(
        "Context repository state: "
        f"branch={branch or 'unavailable'} commit={(head_commit or 'unavailable')[:12]} "
        f"dirty={bool(git_status) if git_status is not None else 'unavailable'}"
    )
    root = Path(config.output_dir)
    run_dir = root / "runs" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "serena.log"
    try:
        with log_path.open("w", encoding="utf-8") as stderr_log:
            (
                report,
                events,
                reports,
                available_tools,
                total_tool_calls,
                report_errors,
                request_limiter,
            ) = asyncio.run(_run_serena(
                request,
                config,
                stderr_log,
                initial_report=initial_report,
                active_questions=active_questions,
                gate_between_rounds=gate_between_rounds,
                auto_approve=auto_approve,
                live_transcript_path=run_dir / "serena-live-transcript.json",
            ))
    except Exception as error:
        diagnostic_path = run_dir / "serena-error.json"
        diagnostic_path.write_text(
            json.dumps({
                "request": request,
                "active_questions": active_questions or [],
                "schema_version": SERENA_SCHEMA_VERSION,
                "structured_output_method": SERENA_STRUCTURED_OUTPUT_METHOD,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "exception_repr": repr(error),
                "details": getattr(error, "details", None),
                "traceback": traceback.format_exc(),
                "serena_log": str(log_path.resolve()),
            }, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        latest = root / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(Path("runs") / run_dir.name, target_is_directory=True)
        print(f"SERENA ERROR: {type(error).__name__}: {error}")
        print(f"Diagnostic: {diagnostic_path.resolve()}")
        raise SerenaContextRunError(
            str(error), str(diagnostic_path.resolve())
        ) from error
    report_path = run_dir / "context.json"
    transcript_path = run_dir / "serena-transcript.json"
    rounds_path = run_dir / "round-reports.json"
    evidence_path = run_dir / "evidence.json"
    brief_path = run_dir / "research-brief.json"
    inquiry_path = run_dir / "inquiry-ledger.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    transcript_path.write_text(
        json.dumps(events, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rounds_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    brief_path.write_text(
        json.dumps(report.get("research_brief", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    inquiry_path.write_text(
        json.dumps(report.get("inquiry_ledger", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps({
            "request": request,
            "active_questions": active_questions or [],
            "repo_path": repo_path,
            "branch": branch,
            "head_commit": head_commit,
            "git_status": git_status,
            "dirty": bool(git_status) if git_status is not None else None,
            "serena_command": [config.command, *config.args],
            "allowed_tools": sorted(READ_ONLY_SERENA_TOOLS),
            "available_allowed_tools": available_tools,
            "max_rounds": config.max_rounds,
            "max_tool_calls_per_round": config.max_tool_calls_per_round,
            "max_total_tool_calls": config.max_total_tool_calls,
            "total_tool_calls": total_tool_calls,
            "rounds_completed": len(reports),
            "max_tool_result_chars": config.max_tool_result_chars,
            "max_transcript_chars": config.max_transcript_chars,
            "max_report_output_tokens": config.max_report_output_tokens,
            "max_explorer_output_tokens": config.max_explorer_output_tokens,
            "context_input_target_tokens": config.context_input_target_tokens,
            "max_round_tool_result_chars": config.max_round_tool_result_chars,
            "schema_version": SERENA_SCHEMA_VERSION,
            "structured_output_method": SERENA_STRUCTURED_OUTPUT_METHOD,
            "model_request_min_interval_seconds": (
                config.model_request_min_interval_seconds
            ),
            "model_requests": request_limiter.request_count,
            "model_request_waits": request_limiter.wait_count,
            "model_request_wait_seconds": round(
                request_limiter.total_wait_seconds,
                3,
            ),
            "estimated_model_input_tokens": request_limiter.estimated_input_tokens,
            "actual_model_input_tokens": request_limiter.actual_input_tokens,
            "actual_model_output_tokens": request_limiter.actual_output_tokens,
            "report_errors": report_errors,
            "model": {
                "provider": config.model.provider,
                "model": config.model.model,
            },
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(Path("runs") / run_dir.name, target_is_directory=True)
    return {
        "report": report,
        "paths": {
            "context": str(report_path.resolve()),
            "transcript": str(transcript_path.resolve()),
            "rounds": str(rounds_path.resolve()),
            "evidence": str(evidence_path.resolve()),
            "log": str(log_path.resolve()),
            "research_brief": str(brief_path.resolve()),
            "inquiry_ledger": str(inquiry_path.resolve()),
        },
    }
