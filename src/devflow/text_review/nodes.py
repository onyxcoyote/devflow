# src/devflow/text_review/nodes.py

from .models import get_review_model
from .schemas import AIReview
from .state import TextReviewState

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

def create_report(state: TextReviewState) -> dict:
    review = state["ai_review"]

    issue_lines = []

    for issue in review["issues"]:
        issue_lines.append(
            f"- [{issue['severity']}] {issue['description']}"
        )

    issues_text = (
        "\n".join(issue_lines)
        if issue_lines
        else "- No specific issues found."
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
    )

    return {
        "report": report,
    }
