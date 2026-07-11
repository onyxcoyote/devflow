from __future__ import annotations

import asyncio
import json
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import PlanningConfig

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


class RelevantFile(BaseModel):
    path: str
    reason: str
    symbols: list[str] = Field(default_factory=list)


class SerenaContextReport(BaseModel):
    status: Literal["sufficient", "insufficient"]
    architecture_summary: str
    relevant_files: list[RelevantFile] = Field(default_factory=list)
    relevant_symbols: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SerenaSpikeConfig:
    repo_path: str
    output_dir: str
    command: str
    args: tuple[str, ...]
    max_tool_calls: int
    max_tool_result_chars: int
    max_transcript_chars: int
    model: Any


def load_serena_spike_config(planning: PlanningConfig) -> SerenaSpikeConfig:
    settings: dict[str, Any] = {}
    for source in planning.config_sources:
        with Path(source).open("rb") as file:
            settings.update(tomllib.load(file).get("serena", {}))

    output_dir = Path(settings.get("output_dir", ".devflow/serena-spikes"))
    if not output_dir.is_absolute():
        output_dir = Path(planning.repo_path) / output_dir
    args = settings.get("args", [
        "start-mcp-server",
        "--context",
        "ide",
        "--project",
        "{repo}",
    ])
    return SerenaSpikeConfig(
        repo_path=planning.repo_path,
        output_dir=str(output_dir),
        command=settings.get("command", "serena"),
        args=tuple(str(item).replace("{repo}", planning.repo_path) for item in args),
        max_tool_calls=max(1, min(30, int(settings.get("max_tool_calls", 12)))),
        max_tool_result_chars=max(
            500,
            int(settings.get("max_tool_result_chars", 8_000)),
        ),
        max_transcript_chars=max(
            5_000,
            int(settings.get("max_transcript_chars", 60_000)),
        ),
        model=planning.model,
    )


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


def _tool_result_text(result, max_chars: int) -> str:
    structured = (
        getattr(result, "structuredContent", None)
        or getattr(result, "structured_content", None)
    )
    if structured is not None:
        text = json.dumps(structured, ensure_ascii=False, default=str)
    else:
        blocks = []
        for block in getattr(result, "content", []):
            block_text = getattr(block, "text", None)
            blocks.append(block_text if block_text is not None else str(block))
        text = "\n".join(blocks)
    return text[:max_chars]


async def _explore_with_session(request: str, config: SerenaSpikeConfig, session):
    try:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        from devflow.code_review.models import get_code_review_model
    except ImportError as error:
        raise ValueError("LangChain is required; reinstall devflow") from error

    await session.initialize()
    listed = await session.list_tools()
    tools = _langchain_tools(listed.tools)
    if not tools:
        raise ValueError("Serena exposed no permitted read-only retrieval tools")

    model = get_code_review_model(config.model)
    explorer = model.bind_tools(tools)
    messages = [
        SystemMessage(content=(
            "You are exploring a code repository to identify context for an implementation plan. "
            "Use Serena's semantic retrieval tools. Do not propose edits and do not claim a path, "
            "symbol, or relationship unless a tool result supports it. Start broad, then follow "
            "definitions and references. Stop when you can identify the actual relevant files, "
            "symbols, architecture, tests, and remaining unknowns."
        )),
        HumanMessage(content=f"Development request:\n{request}"),
    ]
    events: list[dict[str, Any]] = []
    tool_calls_used = 0

    while tool_calls_used < config.max_tool_calls:
        response = await explorer.ainvoke(messages)
        messages.append(response)
        tool_calls = response.tool_calls or []
        if not tool_calls:
            events.append({"assistant_summary": str(response.content)})
            break
        for call in tool_calls:
            if tool_calls_used >= config.max_tool_calls:
                break
            name = call["name"]
            if name not in READ_ONLY_SERENA_TOOLS:
                continue
            arguments = call.get("args", {})
            started_at = perf_counter()
            result = await session.call_tool(name, arguments=arguments)
            elapsed = round(perf_counter() - started_at, 3)
            result_text = _tool_result_text(result, config.max_tool_result_chars)
            events.append({
                "tool": name,
                "arguments": arguments,
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
            tool_calls_used += 1

    report_model = model.with_structured_output(SerenaContextReport)
    transcript_text = json.dumps(events, ensure_ascii=False)
    if len(transcript_text) > config.max_transcript_chars:
        transcript_text = transcript_text[:config.max_transcript_chars]
        transcript_text += "\n[Transcript truncated by devflow]"
    report_prompt = (
        "Create a grounded repository-context report for the development request below. "
        "Use only the Serena exploration transcript. Include only paths and symbols explicitly "
        "supported by tool results. Set status to sufficient or insufficient.\n\n"
        f"DEVELOPMENT REQUEST\n{request}\n\n"
        f"SERENA TRANSCRIPT\n{transcript_text}"
    )
    report = await report_model.ainvoke(report_prompt)
    return report.model_dump(), events, [tool["function"]["name"] for tool in tools]


async def _run_serena(request: str, config: SerenaSpikeConfig):
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as error:
        raise ValueError("The MCP Python package is required; reinstall devflow") from error

    parameters = StdioServerParameters(
        command=config.command,
        args=list(config.args),
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            return await _explore_with_session(request, config, session)


def run_serena_spike(request: str, config: SerenaSpikeConfig) -> dict[str, Any]:
    report, events, available_tools = asyncio.run(_run_serena(request, config))
    root = Path(config.output_dir)
    run_dir = root / "runs" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "context.json"
    transcript_path = run_dir / "serena-transcript.json"
    evidence_path = run_dir / "evidence.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    transcript_path.write_text(
        json.dumps(events, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps({
            "request": request,
            "repo_path": config.repo_path,
            "serena_command": [config.command, *config.args],
            "allowed_tools": sorted(READ_ONLY_SERENA_TOOLS),
            "available_allowed_tools": available_tools,
            "max_tool_calls": config.max_tool_calls,
            "max_tool_result_chars": config.max_tool_result_chars,
            "max_transcript_chars": config.max_transcript_chars,
            "model": {
                "provider": config.model.provider,
                "model": config.model.model,
            },
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "report": report,
        "paths": {
            "context": str(report_path.resolve()),
            "transcript": str(transcript_path.resolve()),
            "evidence": str(evidence_path.resolve()),
        },
    }
