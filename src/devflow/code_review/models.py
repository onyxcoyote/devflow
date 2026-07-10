from __future__ import annotations

import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import ModelConfig


def get_code_review_model(config: ModelConfig):
    if config.provider == "ollama":
        return ChatOllama(
            model=config.model,
            base_url=config.base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=config.temperature,
        )

    if config.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for the OpenRouter provider")
        return ChatOpenAI(
            model=config.model,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=config.temperature,
        )

    raise ValueError(f"Unsupported model provider: {config.provider}")
