from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from devflow.code_review.config import ModelConfig, load_code_review_config


@dataclass(frozen=True)
class PlanningConfig:
    repo_path: str
    output_dir: str
    max_context_chars: int
    save_model_exchange: bool
    model: ModelConfig
    config_sources: tuple[str, ...]


def _plan_settings(config_sources: tuple[str, ...]) -> dict:
    settings: dict = {}
    for source in config_sources:
        with Path(source).open("rb") as file:
            settings.update(tomllib.load(file).get("plan", {}))
    return settings


def load_planning_config(
    repo_path: str | Path = ".",
    config_path: str | Path | None = None,
    *,
    global_config_path: str | Path | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> PlanningConfig:
    shared = load_code_review_config(
        repo_path,
        config_path,
        global_config_path=global_config_path,
        provider_override=provider_override,
        model_override=model_override,
    )
    settings = _plan_settings(shared.config_sources)
    output_dir = Path(settings.get("output_dir", ".devflow/plans"))
    if not output_dir.is_absolute():
        output_dir = Path(shared.repo_path) / output_dir

    return PlanningConfig(
        repo_path=shared.repo_path,
        output_dir=str(output_dir),
        max_context_chars=int(settings.get("max_context_chars", 30_000)),
        save_model_exchange=bool(settings.get("save_model_exchange", False)),
        model=shared.model,
        config_sources=shared.config_sources,
    )
