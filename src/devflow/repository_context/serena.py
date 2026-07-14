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
ROUND_EXTENSION_CALLS = 6


SERENA_SCHEMA_VERSION = "portable-v1"
SERENA_STRUCTURED_OUTPUT_METHOD = "function_calling"


class SerenaSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class SerenaContextRunError(RuntimeError):
    def __init__(self, message: str, diagnostic_path: str):
        super().__init__(message)
        self.diagnostic_path = diagnostic_path


class _SerenaReportGenerationError(RuntimeError):
    def __init__(self, errors: list[dict[str, Any]]):
        super().__init__("Serena context report generation failed after two attempts")
        self.details = errors


class _ModelRequestLimiter:
    def __init__(self, minimum_interval_seconds: float):
        self.minimum_interval_seconds = minimum_interval_seconds
        self.last_started_at: float | None = None
        self.request_count = 0
        self.wait_count = 0
        self.total_wait_seconds = 0.0

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
        try:
            return await model.ainvoke(messages)
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
            _references_generated_artifacts(key)
            or _references_generated_artifacts(item)
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
            f"Exploration round: {round_number}\n"
            f"Prior grounded report:\n{prior_text}\n\n"
            f"{_round_focus_instruction(is_final_round)}"
        )),
    ]
    events: list[dict[str, Any]] = []
    calls_attempted = 0
    active_budget = tool_call_budget
    stop_round = False
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
            events.append({"assistant_summary": str(response.content)})
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
                result = await session.call_tool(name, arguments=arguments)
                elapsed = round(perf_counter() - started_at, 3)
                result_text = _tool_result_text(result, config.max_tool_result_chars)
                events.append({
                    "round": round_number,
                    "tool": name,
                    "arguments": arguments,
                    "duplicate": False,
                    "result": result_text,
                    "elapsed_seconds": elapsed,
                    "is_error": bool(
                        getattr(result, "isError", False)
                        or getattr(result, "is_error", False)
                    ),
                })
            messages.append(ToolMessage(
                content=result_text,
                tool_call_id=call["id"],
            ))
            calls_attempted += 1
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
            return report.model_dump(), events, calls_attempted, attempt_errors
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

    model = get_code_review_model(config.model)
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
    evidence_path.write_text(
        json.dumps({
            "request": request,
            "active_questions": active_questions or [],
            "repo_path": config.repo_path,
            "head_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "git_status": subprocess.run(
                ["git", "status", "--short"],
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                check=True,
            ).stdout,
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
        },
    }
