from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devflow.config import ModelConfig, resolve_config


@dataclass(frozen=True)
class SerenaContextConfig:
    repo_path: str
    output_dir: str
    command: str
    args: tuple[str, ...]
    max_rounds: int
    max_tool_calls_per_round: int
    max_total_tool_calls: int
    max_tool_result_chars: int
    max_transcript_chars: int
    max_report_output_tokens: int
    model_request_min_interval_seconds: float
    model: ModelConfig
    config_sources: tuple[str, ...]


def load_serena_context_config(
    repo_path: str | Path = ".",
    config_path: str | Path | None = None,
    *,
    global_config_path: str | Path | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> SerenaContextConfig:
    shared = resolve_config(
        repo_path,
        config_path,
        global_config_path=global_config_path,
        provider_override=provider_override,
        model_override=model_override,
    )
    settings: dict[str, Any] = shared.raw.get("serena", {})
    output_dir = Path(settings.get("output_dir", ".devflow/serena-context"))
    if not output_dir.is_absolute():
        output_dir = Path(shared.repo_path) / output_dir
    args = settings.get("args", [
        "start-mcp-server",
        "--context",
        "ide",
        "--project",
        "{repo}",
    ])
    return SerenaContextConfig(
        repo_path=shared.repo_path,
        output_dir=str(output_dir),
        command=settings.get("command", "serena"),
        args=tuple(
            str(item).replace("{repo}", shared.repo_path) for item in args
        ),
        max_rounds=max(1, min(5, int(settings.get("max_rounds", 3)))),
        max_tool_calls_per_round=max(
            1,
            min(30, int(settings.get(
                "max_tool_calls_per_round",
                settings.get("max_tool_calls", 12),
            ))),
        ),
        max_total_tool_calls=max(
            1,
            min(60, int(settings.get("max_total_tool_calls", 36))),
        ),
        max_tool_result_chars=max(
            500,
            int(settings.get("max_tool_result_chars", 8_000)),
        ),
        max_transcript_chars=max(
            5_000,
            int(settings.get("max_transcript_chars", 60_000)),
        ),
        max_report_output_tokens=max(
            500,
            min(10_000, int(settings.get("max_report_output_tokens", 5_000))),
        ),
        model_request_min_interval_seconds=max(
            0.0,
            float(settings.get("model_request_min_interval_seconds", 2.0)),
        ),
        model=shared.model,
        config_sources=shared.config_sources,
    )
