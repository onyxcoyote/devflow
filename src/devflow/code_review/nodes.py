import json

from .models import get_code_review_model
from .schemas import CodeReview
from .state import CodeReviewState


def prepare_review_context(state: CodeReviewState) -> dict:
    """Format deterministic evidence into bounded reviewer context."""

    command_text = json.dumps(
        state["command_results"],
        indent=2,
        ensure_ascii=False,
    )
    truncation_note = (
        "The diff was truncated; treat missing context as uncertainty."
        if state["diff_truncated"]
        else "The supplied diff was not truncated."
    )

    context = (
        f"Base ref: {state['base_ref']}\n"
        f"Changed files: {', '.join(state['changed_files']) or '(none)'}\n"
        f"{truncation_note}\n\n"
        "BUILD AND TEST RESULTS\n"
        "======================\n"
        f"{command_text}\n\n"
        "GIT DIFF\n"
        "========\n"
        f"{state['diff']}"
    )

    return {"review_context": context}


def review_code(state: CodeReviewState) -> dict:
    """Ask the model for a schema-validated, read-only code review."""

    model = get_code_review_model().with_structured_output(CodeReview)
    prompt = (
        "You are reviewing a proposed code change.\n"
        "Use only the supplied diff and command results.\n"
        "Prioritize real correctness bugs, regressions, and missing tests.\n"
        "Do not invent files, callers, requirements, or test results.\n"
        "If evidence is missing, record it under uncertainties or use "
        "insufficient_context.\n"
        "Passing tests do not prove correctness.\n"
        "Use blocked_by_failed_checks when failed checks prevent a reliable "
        "approval.\n"
        "Return approve only when no actionable finding is supported by the "
        "evidence.\n\n"
        f"{state['review_context']}"
    )

    review = model.invoke(prompt)
    return {"review": review.model_dump()}


def create_report(state: CodeReviewState) -> dict:
    """Render the structured review as Markdown."""

    review = state["review"]
    lines = [
        "# Code review",
        "",
        f"**Verdict:** `{review['verdict']}`",
        "",
        review["summary"],
        "",
        "## Findings",
        "",
    ]

    if review["findings"]:
        for index, finding in enumerate(review["findings"], start=1):
            location = finding.get("file") or "Location not specified"
            if finding.get("line") is not None:
                location += f":{finding['line']}"

            lines.extend(
                [
                    f"### {index}. {finding['summary']}",
                    "",
                    f"- Severity: `{finding['severity']}`",
                    f"- Confidence: `{finding['confidence']}`",
                    f"- Category: `{finding['category']}`",
                    f"- Location: `{location}`",
                    "",
                    finding["reasoning"],
                    "",
                    f"**Suggested action:** {finding['suggested_action']}",
                    "",
                ]
            )
    else:
        lines.extend(["No actionable findings.", ""])

    lines.extend(["## Uncertainties", ""])
    if review["uncertainties"]:
        lines.extend(f"- {item}" for item in review["uncertainties"])
    else:
        lines.append("- None reported.")

    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Base ref: `{state['base_ref']}`",
            f"- Changed files: {len(state['changed_files'])}",
            f"- Diff truncated: {state['diff_truncated']}",
            f"- Commands run: {len(state['command_results'])}",
            "",
        ]
    )

    return {"report": "\n".join(lines)}
