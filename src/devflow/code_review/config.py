from __future__ import annotations

import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ProviderName = Literal["ollama", "openrouter"]


@dataclass(frozen=True)
class ModelConfig:
    provider: ProviderName
    model: str
    base_url: str | None = None
    temperature: float = 0.0


@dataclass(frozen=True)
class CodeReviewConfig:
    repo_path: str
    base_ref: str
    output_dir: str
    check_commands: tuple[tuple[str, ...], ...] = ()
    test_commands: tuple[tuple[str, ...], ...] = ()
    max_diff_chars: int = 40_000
    max_command_output_chars: int = 12_000
    model: ModelConfig = ModelConfig(provider="ollama", model="")


def _commands(values: list[str] | None) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(shlex.split(value)) for value in (values or []))


def load_code_review_config(
    repo_path: str | Path = ".",
    config_path: str | Path | None = None,
) -> CodeReviewConfig:
    repo = Path(repo_path).expanduser().resolve()
    path = Path(config_path).expanduser().resolve() if config_path else repo / ".devflow.toml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Copy .devflow.example.toml to {repo / '.devflow.toml'} and edit it."
        )

    with path.open("rb") as file:
        raw = tomllib.load(file)

    review = raw.get("review", {})
    command_config = raw.get("commands", {})
    model_config = raw.get("model", {})
    provider = model_config.get("provider", "ollama")
    if provider not in {"ollama", "openrouter"}:
        raise ValueError(f"Unsupported model provider: {provider}")

    default_model = os.getenv("OLLAMA_MODEL", "") if provider == "ollama" else ""
    model_name = model_config.get("model", default_model)
    if not model_name:
        raise ValueError("A model name is required in [model].model")

    output_dir = Path(review.get("output_dir", ".devflow/reviews"))
    if not output_dir.is_absolute():
        output_dir = repo / output_dir

    return CodeReviewConfig(
        repo_path=str(repo),
        base_ref=review.get("base_ref", "HEAD~1"),
        output_dir=str(output_dir),
        check_commands=_commands(command_config.get("check")),
        test_commands=_commands(command_config.get("test")),
        max_diff_chars=int(review.get("max_diff_chars", 40_000)),
        max_command_output_chars=int(review.get("max_command_output_chars", 12_000)),
        model=ModelConfig(
            provider=provider,
            model=model_name,
            base_url=model_config.get("base_url"),
            temperature=float(model_config.get("temperature", 0)),
        ),
    )
