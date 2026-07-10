from __future__ import annotations

import json

from .schemas import CodeReview
from .state import CodeReviewState


def prepare_review_context(state: CodeReviewState) -> dict:
    command_text = json.dumps(state["command_results"], indent=2, ensure_ascii=False)
    truncation_note = (
        "The diff was truncated; treat missing context as uncertainty."
        if state["diff_truncated"]
        else "The supplied diff was not truncated."
    )
    return {
        "review_context": (
            f"Base ref: {state['base_ref']}\n"
            f"Changed files: {', '.join(state['changed_files']) or '(none)'}\n"
            f"{truncation_note}\n\n"
            "BUILD AND TEST RESULTS\n======================\n"
            f"{command_text}\n\nGIT DIFF\n========\n{state['diff']}"
        )
    }


def make_review_node(model):
    structured_model = model.with_structured_output(CodeReview)

    def review_code(state: CodeReviewState) -> dict:
        prompt = (
            "You are reviewing a proposed code change.\n"
            "Use only the supplied diff and command results.\n"
            "Prioritize correctness bugs, regressions, security issues, and missing tests.\n"
            "Do not invent files, callers, requirements, or test results.\n"
            "Passing tests do not prove correctness.\n"
            "Use insufficient_context when the evidence is not enough.\n\n"
            f"{state['review_context']}"
        )
        return {"review": structured_model.invoke(prompt).model_dump()}

    return review_code


def assess_review(state: CodeReviewState) -> dict:
    review = state["review"]
    command_failed = any(not result["passed"] for result in state["command_results"])
    penalties = {"high": 25, "medium": 10, "low": 3}
    score = max(0, 100 - sum(penalties[item["severity"]] for item in review["findings"]))
    if command_failed:
        score = max(0, score - 30)

    if review["verdict"] == "insufficient_context":
        verdict = "inconclusive"
    elif command_failed or any(item["severity"] == "high" for item in review["findings"]) or score < 70:
        verdict = "fail"
    elif review["findings"] or score < 90:
        verdict = "pass_with_warnings"
    else:
        verdict = "pass"

    confidences = [item["confidence"] for item in review["findings"]]
    confidence = "low" if "low" in confidences else "medium" if "medium" in confidences else "high"
    return {"assessment": {**review, "score": score, "verdict": verdict, "confidence": confidence}}


def create_report(state: CodeReviewState) -> dict:
    assessment = state["assessment"]
    lines = [
        "# Code review", "", f"**Verdict:** `{assessment['verdict']}`", f"**Score:** `{assessment['score']}/100`",
        f"**Confidence:** `{assessment['confidence']}`", "", assessment["summary"], "", "## Findings", "",
    ]
    if assessment["findings"]:
        for index, finding in enumerate(assessment["findings"], start=1):
            location = finding.get("file") or "Location not specified"
            if finding.get("line") is not None:
                location += f":{finding['line']}"
            lines.extend([
                f"### {index}. {finding['summary']}", "",
                f"- Severity: `{finding['severity']}`", f"- Confidence: `{finding['confidence']}`",
                f"- Category: `{finding['category']}`", f"- Location: `{location}`", "",
                finding["reasoning"], "", f"**Suggested action:** {finding['suggested_action']}", "",
            ])
    else:
        lines.extend(["No actionable findings.", ""])
    lines.extend(["## Uncertainties", ""])
    lines.extend(f"- {item}" for item in assessment["uncertainties"] or ["None reported."])
    return {"report": "\n".join(lines) + "\n"}
