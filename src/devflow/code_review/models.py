from __future__ import annotations

import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import ModelConfig


def get_code_review_model(
    config: ModelConfig,
    *,
    max_output_tokens: int | None = None,
):
    if config.provider == "ollama":
        output_options = (
            {"num_predict": max_output_tokens}
            if max_output_tokens is not None
            else {}
        )
        return ChatOllama(
            model=config.model,
            base_url=config.base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=config.temperature,
            **output_options,
        )

    if config.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for the OpenRouter provider")
        output_options = (
            {"max_tokens": max_output_tokens}
            if max_output_tokens is not None
            else {}
        )
        return ChatOpenAI(
            model=config.model,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=config.temperature,
            **output_options,
        )

    raise ValueError(f"Unsupported model provider: {config.provider}")
