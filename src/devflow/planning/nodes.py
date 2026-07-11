from __future__ import annotations

import json
from time import perf_counter

from devflow.code_review.nodes import _model_result_metadata, _raw_response_data

from .schemas import DevelopmentPlan
from .state import PlanningState


def prepare_plan_context(state: PlanningState) -> dict:
    return {
        "context_text": json.dumps(
            state["repository_context"],
            indent=2,
            ensure_ascii=False,
        )
    }


def make_plan_node(model):
    structured_model = model.with_structured_output(
        DevelopmentPlan,
        include_raw=True,
    )

    def create_plan(state: PlanningState) -> dict:
        prompt = (
            "You are a senior software engineer planning a repository change.\n"
            "Produce an implementation contract, not code and not a patch.\n"
            "Use only the supplied request and repository context. Do not invent files, "
            "requirements, APIs, or current behavior.\n"
            "Focus on the intended outcome, affected areas, acceptance criteria, verification, "
            "risks, assumptions, and unresolved decisions.\n"
            "Treat likely_files as advisory; include only files supported by the supplied context.\n"
            "Use needs_context when specific missing repository information prevents a responsible "
            "plan. Use needs_user_decision when a product or architectural choice materially changes "
            "the implementation. A ready plan must have concrete acceptance criteria and verification.\n\n"
            f"DEVELOPMENT REQUEST\n===================\n{state['request']}\n\n"
            f"REPOSITORY CONTEXT\n==================\n{state['context_text']}"
        )

        started_at = perf_counter()
        result = structured_model.invoke(prompt)
        elapsed_seconds = perf_counter() - started_at
        raw_response = result["raw"]
        parsed_plan = result["parsed"]
        parsing_error = result["parsing_error"]
        model_result = _model_result_metadata(
            raw_response,
            parsing_error,
            elapsed_seconds,
        )
        model_exchange = (
            {
                "request": {"prompt": prompt},
                "response": _raw_response_data(raw_response),
            }
            if state["save_model_exchange"]
            else {}
        )

        response_truncated = model_result["finish_reason"] in {"length", "max_tokens"}
        if response_truncated or parsing_error is not None or parsed_plan is None:
            detail = (
                "The model response reached its output-token limit."
                if response_truncated
                else model_result["parsing_error"] or "No parsed plan was returned."
            )
            plan = {
                "status": "needs_context",
                "objective": state["request"],
                "understanding": "The model response could not be parsed into a development plan.",
                "assumptions": [],
                "uncertainties": [detail],
                "proposed_changes": [],
                "acceptance_criteria": [],
                "verification": [],
                "risks": [],
            }
        else:
            plan = parsed_plan.model_dump()

        return {
            "plan": plan,
            "model_result": model_result,
            "model_exchange": model_exchange,
        }

    return create_plan


def create_plan_report(state: PlanningState) -> dict:
    plan = state["plan"]
    model_info = state["model_info"]
    lines = [
        "# Development plan",
        "",
        f"**Status:** `{plan['status']}`",
        f"**Model:** `{model_info['model']}`",
        f"**Provider:** `{model_info['provider']}`",
        "",
        "## Objective",
        "",
        plan["objective"],
        "",
        "## Understanding",
        "",
        plan["understanding"],
        "",
        "## Proposed changes",
        "",
    ]
    if plan["proposed_changes"]:
        for index, change in enumerate(plan["proposed_changes"], start=1):
            files = ", ".join(change["likely_files"]) or "Not yet identified"
            lines.extend([
                f"### {index}. {change['area']}",
                "",
                change["description"],
                "",
                f"- Likely files: {files}",
                f"- Reason: {change['reason']}",
                "",
            ])
    else:
        lines.extend(["No changes proposed yet.", ""])

    sections = (
        ("Acceptance criteria", "acceptance_criteria"),
        ("Verification", "verification"),
        ("Assumptions", "assumptions"),
        ("Uncertainties", "uncertainties"),
        ("Risks", "risks"),
    )
    for title, key in sections:
        lines.extend([f"## {title}", ""])
        lines.extend(f"- {item}" for item in plan[key] or ["None reported."])
        lines.append("")

    return {"report": "\n".join(lines)}
