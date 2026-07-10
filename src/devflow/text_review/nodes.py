# src/devflow/text_review/nodes.py

from .models import get_review_model
from .schemas import AIReview, TextSuggestion
from .state import TextReviewState


def route_after_review(
    state: TextReviewState,
) -> str:
    if state["ai_review"]["verdict"] == "needs_improvement":
        return "suggest_improvement"

    return "create_report"

def normalize_text(state: TextReviewState) -> dict:
    normalized = " ".join(state["original_text"].split())

    return {
        "normalized_text": normalized,
    }


def analyze_text(state: TextReviewState) -> dict:
    text = state["normalized_text"]

    return {
        "word_count": len(text.split()),
        "character_count": len(text),
    }


def review_text_with_ai(state: TextReviewState) -> dict:
    model = get_review_model()

    structured_model = model.with_structured_output(AIReview)

    prompt = (
        "Review the following text for clarity and wording.\n"
        "Do not rewrite it.\n"
        "Return a concise summary and a list of specific issues.\n"
        "Use an empty issues list when the text is already clear.\n\n"
        f"Text:\n{state['normalized_text']}"
    )

    review = structured_model.invoke(prompt)

    return {
        "ai_review": review.model_dump(),
    }

def suggest_improvement(state: TextReviewState) -> dict:
    model = get_review_model()
    structured_model = model.with_structured_output(TextSuggestion)

    issues = "\n".join(
        f"- {issue['description']}"
        for issue in state["ai_review"]["issues"]
    )

    prompt = (
        "Rewrite the following text to address the identified issues.\n"
        "Preserve its original meaning.\n"
        "Keep the rewrite concise.\n\n"
        f"Original text:\n{state['normalized_text']}\n\n"
        f"Identified issues:\n{issues}"
    )

    suggestion = structured_model.invoke(prompt)

    return {
        "suggestion": suggestion.model_dump(),
    }

def create_report(state: TextReviewState) -> dict:
    review = state["ai_review"]
    suggestion = state["suggestion"]

    issue_lines = [
        f"- [{issue['severity']}] {issue['description']}"
        for issue in review["issues"]
    ]

    issues_text = (
        "\n".join(issue_lines)
        if issue_lines
        else "- No specific issues found."
    )

    suggestion_text = ""

    if suggestion:
        suggestion_text = (
            "\nSuggested improvement\n"
            "---------------------\n"
            f"{suggestion['improved_text']}\n\n"
            "Explanation\n"
            "-----------\n"
            f"{suggestion['explanation']}\n"
        )

    report = (
        "Text review\n"
        "===========\n"
        f"Normalized text: {state['normalized_text']}\n"
        f"Word count: {state['word_count']}\n"
        f"Character count: {state['character_count']}\n\n"
        "AI review\n"
        "---------\n"
        f"Verdict: {review['verdict']}\n"
        f"Summary: {review['summary']}\n\n"
        "Issues\n"
        "------\n"
        f"{issues_text}\n"
        f"{suggestion_text}"
    )

    return {
        "report": report,
    }
