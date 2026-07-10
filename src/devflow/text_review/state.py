# src/devflow/text_review/state.py

from typing import Any, TypedDict


class TextReviewState(TypedDict):
    original_text: str
    normalized_text: str
    word_count: int
    character_count: int
    ai_review: dict[str, Any]
    report: str
