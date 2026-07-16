from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from devflow.config import ModelConfig, resolve_config


@dataclass(frozen=True)
class CodeReviewConfig:
    repo_path: str
    base_ref: str
    output_dir: str
    check_commands: tuple[tuple[str, ...], ...] = ()
    test_commands: tuple[tuple[str, ...], ...] = ()
    max_diff_chars: int = 40_000
    max_command_output_chars: int = 12_000
    save_model_exchange: bool = False
    model: ModelConfig = ModelConfig(
        provider="ollama",
        model="",
        base_url="http://127.0.0.1:11434",
    )
    config_sources: tuple[str, ...] = ()


def _commands(values: list[str] | None) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(shlex.split(value)) for value in (values or []))


def load_code_review_config(
    repo_path: str | Path = ".",
    config_path: str | Path | None = None,
    *,
    global_config_path: str | Path | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> CodeReviewConfig:
    shared = resolve_config(
        repo_path,
        config_path,
        global_config_path=global_config_path,
        provider_override=provider_override,
        model_override=model_override,
    )
    raw = shared.raw
    review = raw.get("review", {})
    command_config = raw.get("commands", {})
    output_dir = Path(review.get("output_dir", ".devflow/reviews"))
    if not output_dir.is_absolute():
        output_dir = Path(shared.repo_path) / output_dir

    return CodeReviewConfig(
        repo_path=shared.repo_path,
        base_ref=review.get("base_ref", "HEAD~1"),
        output_dir=str(output_dir),
        check_commands=_commands(command_config.get("check")),
        test_commands=_commands(command_config.get("test")),
        max_diff_chars=int(review.get("max_diff_chars", 40_000)),
        max_command_output_chars=int(
            review.get("max_command_output_chars", 12_000)
        ),
        save_model_exchange=bool(review.get("save_model_exchange", False)),
        model=shared.model,
        config_sources=shared.config_sources,
    )
