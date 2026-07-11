from __future__ import annotations

import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ProviderName = Literal["ollama", "openrouter"]
DEFAULT_GLOBAL_CONFIG = Path("~/.config/devflow/config.toml").expanduser()


@dataclass(frozen=True)
class ModelConfig:
    provider: ProviderName
    model: str
    base_url: str
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
    save_model_exchange: bool = False
    model: ModelConfig = ModelConfig(
        provider="ollama",
        model="",
        base_url="http://127.0.0.1:11434",
    )
    config_sources: tuple[str, ...] = ()


def _commands(values: list[str] | None) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(shlex.split(value)) for value in (values or []))


def _load_toml(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Config file not found: {path}")
        return {}

    with path.open("rb") as file:
        return tomllib.load(file)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def _provider_defaults(provider: ProviderName) -> tuple[str, str]:
    if provider == "ollama":
        return (
            os.getenv("OLLAMA_MODEL", ""),
            os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        )

    return (
        os.getenv("OPENROUTER_MODEL", ""),
        "https://openrouter.ai/api/v1",
    )


def load_code_review_config(
    repo_path: str | Path = ".",
    config_path: str | Path | None = None,
    *,
    global_config_path: str | Path | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> CodeReviewConfig:
    repo = Path(repo_path).expanduser().resolve()

    global_path = Path(
        global_config_path
        or os.getenv("DEVFLOW_GLOBAL_CONFIG", str(DEFAULT_GLOBAL_CONFIG))
    ).expanduser().resolve()
    repo_path_config = (
        Path(config_path).expanduser().resolve()
        if config_path
        else repo / ".devflow.toml"
    )

    global_raw = _load_toml(global_path, required=global_config_path is not None)
    repo_raw = _load_toml(repo_path_config, required=config_path is not None)
    raw = _deep_merge(global_raw, repo_raw)

    sources = tuple(
        str(path)
        for path in (global_path, repo_path_config)
        if path.is_file()
    )

    review = raw.get("review", {})
    command_config = raw.get("commands", {})
    model_config = raw.get("model", {})
    providers = raw.get("providers", {})

    configured_provider = model_config.get("provider", "ollama")
    provider = provider_override or configured_provider
    if provider not in {"ollama", "openrouter"}:
        raise ValueError(f"Unsupported model provider: {provider}")

    provider_name: ProviderName = provider
    provider_config = providers.get(provider_name, {})
    default_model, default_base_url = _provider_defaults(provider_name)

    # Backward compatibility with the older single [model] block. Do not carry
    # its URL/model across providers when --provider changes the provider.
    use_legacy_model_fields = provider_override is None or provider_name == configured_provider

    model_name = (
        model_override
        or provider_config.get("model")
        or (model_config.get("model") if use_legacy_model_fields else None)
        or default_model
    )
    if not model_name:
        raise ValueError(
            f"No model configured for provider '{provider_name}'. "
            f"Set [providers.{provider_name}].model, use --model, or set the provider environment variable."
        )

    base_url = (
        provider_config.get("base_url")
        or (model_config.get("base_url") if use_legacy_model_fields else None)
        or default_base_url
    )

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
        max_command_output_chars=int(
            review.get("max_command_output_chars", 12_000)
        ),
        save_model_exchange=bool(review.get("save_model_exchange", False)),
        model=ModelConfig(
            provider=provider_name,
            model=model_name,
            base_url=base_url,
            temperature=float(model_config.get("temperature", 0)),
        ),
        config_sources=sources,
    )
