# src/devflow/text_review/tasks.py

from pathlib import Path

from prefect import task

from .graph import text_review_graph


@task
def prepare_input(text: str) -> dict:
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    return {
        "original_text": text,
        "normalized_text": "",
        "word_count": 0,
        "character_count": 0,
        "ai_review": {
            "verdict": "clear",
            "summary": "",
            "issues": [],
        },
        "suggestion": {},
        "report": "",
    }

@task
def run_review_graph(initial_state: dict) -> dict:
    return text_review_graph.invoke(initial_state)


@task
def save_report(report: str, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")

    return str(path)

@task
def prepare_input(text: str) -> dict:
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    return {
        "original_text": text,
        "normalized_text": "",
        "word_count": 0,
        "character_count": 0,
        "ai_review": "",
        "report": "",
    }
