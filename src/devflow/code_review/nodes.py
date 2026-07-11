from __future__ import annotations

import json

from .schemas import CodeReview
from .state import CodeReviewState
import logging

logger = logging.getLogger(__name__)

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
    structured_model = model.with_structured_output(
        CodeReview,
        include_raw=True,
    )

    def review_code(state: CodeReviewState) -> dict:
        prompt = (
            "You are a senior engineer reviewing a proposed code change.\n"
            "Use only the supplied diff and command results.\n"
            "First infer the change's intended behavior from the diff. Then trace the changed "
            "control and data paths and examine boundary cases, error handling, regressions, "
            "security risks, and whether the tests would detect plausible broken implementations.\n"
            "Report only concrete, actionable problems introduced or exposed by this change. "
            "Do not invent findings to justify a negative verdict, and do not report preferences "
            "or speculative improvements as defects.\n"
            "Return no more than 8 findings. Prefer fewer, higher-confidence findings. "
            "Omit low-value observations instead of filling the limit. "
            "Each finding must describe one complete, concrete, actionable issue. "
            "Never emit a partial finding. "
            "If more than 8 credible issues exist, return the 8 with the greatest "
            "combination of severity and confidence.\n"
            "Do not invent files, callers, requirements, or test results.\n"
            "Passing tests do not prove correctness.\n"
            "For every supplied command result, emit exactly one check_assessment using its "
            "zero-based position as command_index. A failed command is change_failure only when "
            "the supplied evidence connects the failure to this diff; otherwise classify it as "
            "unrelated_failure, environment_failure, or uncertain.\n"
            "Use insufficient_context when missing evidence prevents a reliable code review. "
            "Before responding, ensure the verdict, summary, findings, check assessments, "
            "uncertainties, and overall confidence are mutually consistent.\n\n"
            f"{state['review_context']}"
        )
        result = structured_model.invoke(prompt)

        raw_response = result["raw"]
        parsed_review = result["parsed"]
        parsing_error = result["parsing_error"]

        response_metadata = getattr(raw_response, "response_metadata", {})
        finish_reason = (
            response_metadata.get("finish_reason")
            or response_metadata.get("done_reason")
        )

        if finish_reason in {"length", "max_tokens"}:
            error_summary = (
                "The model response reached its output-token limit and was truncated."
            )
        elif parsing_error is not None:
            error_summary = (
                "The model returned a response that did not match the required "
                "code-review structure."
            )
        else:
            error_summary = None

        if error_summary is not None:
            # Since the review could not be parsed, failed commands cannot be reliably
            # attributed to the change. Mark them uncertain; retain successful commands
            # as passed.
            check_assessments = [
                {
                    "command_index": index,
                    "status": "passed" if command_result["passed"] else "uncertain",
                    "reasoning": (
                        "The command passed."
                        if command_result["passed"]
                        else "The review response could not be parsed, so this failure "
                            "could not be attributed to the proposed change."
                    ),
                }
                for index, command_result in enumerate(state["command_results"])
            ]

            parsing_detail = (
                f" Parser detail: {parsing_error}"
                if parsing_error is not None
                else ""
            )

            #limit error text to 1000 characters 
            parsing_detail = str(parsing_error)[:1000]

            return {
                "review": {
                    "verdict": "insufficient_context",
                    "confidence": "low",
                    "summary": error_summary,
                    "findings": [],
                    "check_assessments": check_assessments,
                    "uncertainties": [
                        f"{error_summary} Parser detail: {parsing_detail}",
                    ],
                }
            }

        return {"review": parsed_review.model_dump()}

    return review_code


def assess_review(state: CodeReviewState) -> dict:
    review = state["review"]
    command_results = state["command_results"]
    check_assessments = review["check_assessments"]
    assessments_by_index = {
        item["command_index"]: item for item in check_assessments
    }

    invalid_check_assessments = (
        len(assessments_by_index) != len(check_assessments)
        or set(assessments_by_index) != set(range(len(command_results)))
        or any(
            result["passed"] != (assessments_by_index[index]["status"] == "passed")
            for index, result in enumerate(command_results)
        )
    )
    change_check_failed = any(
        item["status"] == "change_failure" for item in check_assessments
    )
    uncertain_check = invalid_check_assessments or any(
        item["status"] in {"environment_failure", "uncertain"}
        for item in check_assessments
    )
    penalties = {"high": 25, "medium": 10, "low": 3}
    score = max(0, 100 - sum(penalties[item["severity"]] for item in review["findings"]))
    if change_check_failed:
        score = max(0, score - 30)

    if review["verdict"] == "insufficient_context" or uncertain_check:
        verdict = "inconclusive"
    elif change_check_failed or any(item["severity"] == "high" for item in review["findings"]) or score < 70:
        verdict = "fail"
    elif review["findings"] or score < 90:
        verdict = "pass_with_warnings"
    else:
        verdict = "pass"

    uncertainties = list(review["uncertainties"])
    if invalid_check_assessments:
        uncertainties.append(
            "The model returned incomplete or inconsistent command-result assessments."
        )
    confidence = review["confidence"]
    if verdict == "inconclusive":
        confidence = "low"
    return {
        "assessment": {
            **review,
            "score": score,
            "verdict": verdict,
            "confidence": confidence,
            "uncertainties": uncertainties,
        }
    }


def create_report(state: CodeReviewState) -> dict:
    assessment = state["assessment"]
    model_info = state["model_info"]

    lines = [
        "# Code review",
        "",
        f"**Model:** `{model_info['model']}`",
        f"**Provider:** `{model_info['provider']}`",
        f"**Verdict:** `{assessment['verdict']}`",
        f"**Score:** `{assessment['score']}/100`",
        f"**Confidence:** `{assessment['confidence']}`",
        "",
        assessment["summary"],
        "",
        "## Findings",
        "",
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
    lines.extend(["## Command assessments", ""])
    if assessment["check_assessments"]:
        for item in assessment["check_assessments"]:
            lines.extend([
                f"- Command {item['command_index']}: `{item['status']}` — {item['reasoning']}"
            ])
    else:
        lines.extend(["No commands were assessed."])
    lines.append("")
    lines.extend(["## Uncertainties", ""])
    lines.extend(f"- {item}" for item in assessment["uncertainties"] or ["None reported."])
    return {"report": "\n".join(lines) + "\n"}
