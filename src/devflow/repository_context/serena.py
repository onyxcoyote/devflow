from __future__ import annotations

import asyncio
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

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


PathText = Annotated[str, Field(max_length=500)]
SymbolText = Annotated[str, Field(max_length=300)]
EvidenceText = Annotated[str, Field(max_length=500)]


class RelevantFile(BaseModel):
    path: PathText
    role: Literal[
        "probable_change_target",
        "candidate_change_target",
        "supporting_context",
    ]
    reason: str = Field(max_length=800)
    symbols: list[SymbolText] = Field(default_factory=list, max_length=20)


class ContextEvidence(BaseModel):
    claim: EvidenceText
    source: EvidenceText


class QuestionResolution(BaseModel):
    question: str = Field(max_length=1000)
    resolution: str = Field(max_length=1000)
    source: EvidenceText


class MissingContextItem(BaseModel):
    kind: Literal[
        "repository",
        "user_decision",
        "tool_failure",
        "external_information",
    ]
    description: str = Field(max_length=1000)
    suggested_action: str = Field(max_length=1000)
    related_files: list[PathText] = Field(default_factory=list, max_length=10)
    related_symbols: list[SymbolText] = Field(default_factory=list, max_length=20)


class SerenaContextReport(BaseModel):
    status: Literal[
        "sufficient",
        "needs_repository_context",
        "needs_user_decision",
        "blocked",
    ]
    architecture_summary: str = Field(max_length=3000)
    relevant_files: list[RelevantFile] = Field(default_factory=list, max_length=30)
    relevant_symbols: list[SymbolText] = Field(default_factory=list, max_length=50)
    evidence: list[ContextEvidence] = Field(default_factory=list, max_length=30)
    question_resolutions: list[QuestionResolution] = Field(
        default_factory=list,
        max_length=10,
    )
    missing_context: list[MissingContextItem] = Field(
        default_factory=list,
        max_length=10,
    )


class _ModelRequestLimiter:
    def __init__(self, minimum_interval_seconds: float):
        self.minimum_interval_seconds = minimum_interval_seconds
        self.last_started_at: float | None = None
        self.request_count = 0
        self.wait_count = 0
        self.total_wait_seconds = 0.0

    async def invoke(self, model, messages):
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
        return await model.ainvoke(messages)


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
    is_final_round: bool,
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

    while calls_attempted < tool_call_budget:
        response = await request_limiter.invoke(explorer, messages)
        messages.append(response)
        tool_calls = response.tool_calls or []
        if not tool_calls:
            events.append({"assistant_summary": str(response.content)})
            break
        for call in tool_calls:
            if calls_attempted >= tool_call_budget:
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

    structured_report_model = report_model.with_structured_output(SerenaContextReport)
    transcript_text, included_events = _bounded_transcript(
        events,
        config.max_transcript_chars,
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
        "Keep the complete report comfortably below the output-token limit.\n\n"
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
            report = await request_limiter.invoke(structured_report_model, prompt)
            return report.model_dump(), events, calls_attempted, attempt_errors
        except Exception as error:
            attempt_errors.append({
                "attempt": attempt,
                "type": type(error).__name__,
                "message": str(error)[:2_000],
                "round": round_number,
                "transcript_events_included": included_events,
                "transcript_events_total": len(events),
            })

    blocked_report = SerenaContextReport(
        status="blocked",
        architecture_summary=(
            "Serena exploration completed, but the structured context report "
            "could not be generated after two attempts."
        ),
        missing_context=[MissingContextItem(
            kind="tool_failure",
            description="Both context-report model calls failed.",
            suggested_action=(
                "Inspect the saved Serena transcript and retry report generation."
            ),
        )],
    )
    return blocked_report.model_dump(), events, calls_attempted, attempt_errors


async def _explore_with_session(request: str, config: SerenaContextConfig, session):
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
    report: dict[str, Any] | None = None
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
            is_final_round=is_final_round,
        )
        total_tool_calls += calls
        all_events.extend(events)
        reports.append(report)
        report_errors.extend(round_report_errors)
        print(
            f"Round {round_number}: {calls} tool calls "
            f"({total_tool_calls}/{config.max_total_tool_calls} total)"
        )
        if not _should_continue(
            report,
            round_number=round_number,
            total_tool_calls=total_tool_calls,
            config=config,
        ):
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
            return await _explore_with_session(request, config, session)


def run_serena_context(request: str, config: SerenaContextConfig) -> dict[str, Any]:
    root = Path(config.output_dir)
    run_dir = root / "runs" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "serena.log"
    with log_path.open("w", encoding="utf-8") as stderr_log:
        (
            report,
            events,
            reports,
            available_tools,
            total_tool_calls,
            report_errors,
            request_limiter,
        ) = asyncio.run(_run_serena(request, config, stderr_log))
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
