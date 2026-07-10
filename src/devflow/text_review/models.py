# src/devflow/text_review/models.py

import os

from langchain_ollama import ChatOllama


def get_review_model() -> ChatOllama:
    return ChatOllama(
        model=os.getenv(
            "OLLAMA_MODEL",
            "your-current-model-name",
        ),
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://YOUR_OLLAMA_IP:11434",
        ),
        temperature=0,
    )
