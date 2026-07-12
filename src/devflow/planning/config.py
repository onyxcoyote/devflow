from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from devflow.config import ModelConfig, resolve_config


@dataclass(frozen=True)
class PlanningConfig:
    repo_path: str
    output_dir: str
    save_model_exchange: bool
    max_output_tokens: int
    compact_retry_output_tokens: int
    model: ModelConfig
    config_sources: tuple[str, ...]


def load_planning_config(
    repo_path: str | Path = ".",
    config_path: str | Path | None = None,
    *,
    global_config_path: str | Path | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> PlanningConfig:
    shared = resolve_config(
        repo_path,
        config_path,
        global_config_path=global_config_path,
        provider_override=provider_override,
        model_override=model_override,
    )
    settings = shared.raw.get("plan", {})
    output_dir = Path(settings.get("output_dir", ".devflow/plans"))
    if not output_dir.is_absolute():
        output_dir = Path(shared.repo_path) / output_dir

    return PlanningConfig(
        repo_path=shared.repo_path,
        output_dir=str(output_dir),
        save_model_exchange=bool(settings.get("save_model_exchange", False)),
        max_output_tokens=max(
            1000,
            min(20_000, int(settings.get("max_output_tokens", 8_000))),
        ),
        compact_retry_output_tokens=max(
            500,
            min(10_000, int(settings.get("compact_retry_output_tokens", 4_000))),
        ),
        model=shared.model,
        config_sources=shared.config_sources,
    )
