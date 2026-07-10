import os

from langchain_ollama import ChatOllama


def get_code_review_model() -> ChatOllama:
    """Create the model used by the code-review graph."""

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "your-current-model-name"),
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://YOUR_OLLAMA_IP:11434",
        ),
        temperature=0,
    )
