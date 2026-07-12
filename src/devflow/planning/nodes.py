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
        previous_plan = state["previous_plan"]
        mode_instruction = (
            "Create a new implementation plan from the grounded repository context."
            if previous_plan is None
            else (
                "Critically revise the previous plan. Preserve supported content, correct "
                "unsupported assumptions, and make only substantive improvements. Record the "
                "important changes in revision.changes. The previous plan is a draft, not truth."
            )
        )
        previous_text = (
            "None"
            if previous_plan is None
            else json.dumps(previous_plan, indent=2, ensure_ascii=False)
        )
        prompt = (
            "You are a senior software engineer producing a structured implementation plan.\n"
            "Produce an implementation contract, not code and not a patch.\n"
            f"{mode_instruction}\n"
            "Use repository context for factual claims. Do not invent paths, symbols, APIs, or "
            "current behavior. Repository evidence should identify a supplied file, symbol, or "
            "evidence source. Design recommendations may be planner reasoning.\n"
            "Each proposed change must name one file and describe its responsibility. Include "
            "tests and validation in verification and acceptance criteria.\n"
            "Use needs_repository_context when a specific repository question prevents a "
            "responsible plan, needs_user_decision when a material product or architecture choice "
            "remains, not_feasible when the requested outcome cannot responsibly be planned, and "
            "ready only when the plan is actionable. Preserve unresolved questions explicitly in "
            "outstanding_items and decisions.\n\n"
            f"DEVELOPMENT REQUEST\n===================\n{state['request']}\n\n"
            f"REPOSITORY CONTEXT\n==================\n{state['context_text']}\n\n"
            f"PREVIOUS PLAN\n=============\n{previous_text}"
        )

        started_at = perf_counter()
        result = structured_model.invoke(prompt)
        elapsed_seconds = perf_counter() - started_at
        raw_response = result["raw"]
        parsed_plan = result["parsed"]
        parsing_error = result["parsing_error"]
        plan_metadata = _model_result_metadata(
            raw_response,
            parsing_error,
            elapsed_seconds,
        )
        model_result = {**state["model_result"], "plan": plan_metadata}
        model_exchange = dict(state["model_exchange"])
        if state["save_model_exchange"]:
            model_exchange["plan"] = {
                "request": {"prompt": prompt},
                "response": _raw_response_data(raw_response),
            }

        response_truncated = plan_metadata["finish_reason"] in {
            "length",
            "max_tokens",
        }
        if response_truncated or parsing_error is not None or parsed_plan is None:
            detail = (
                "The model response reached its output-token limit."
                if response_truncated
                else plan_metadata["parsing_error"] or "No parsed plan was returned."
            )
            plan = DevelopmentPlan(
                status="needs_repository_context",
                objective=state["request"],
                design_summary="The model response could not be parsed into a plan.",
                outstanding_items=[{
                    "kind": "repository_context",
                    "question": detail,
                    "impact": "No reliable structured plan was produced.",
                    "suggested_action": "Inspect the model result and retry planning.",
                }],
            ).model_dump()
        else:
            plan = parsed_plan.model_dump()
        if previous_plan is not None:
            plan["revision"]["based_on"] = state["context_source"].get(
                "previous_plan_path"
            )

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
        "## Design summary",
        "",
        plan["design_summary"],
        "",
        "## Proposed changes",
        "",
    ]
    for index, change in enumerate(plan["proposed_changes"], start=1):
        symbols = ", ".join(change["symbols"]) or "None identified"
        evidence = ", ".join(change["evidence"]) or "Planner reasoning"
        lines.extend([
            f"### {index}. `{change['path']}`",
            "",
            change["change"],
            "",
            f"- Symbols: {symbols}",
            f"- Reason: {change['reason']}",
            f"- Evidence: {evidence}",
            "",
        ])
    if not plan["proposed_changes"]:
        lines.extend(["No changes proposed yet.", ""])

    lines.extend(["## Outstanding items", ""])
    for item in plan["outstanding_items"]:
        lines.extend([
            f"- **{item['kind']}:** {item['question']}",
            f"  - Impact: {item['impact']}",
            f"  - Next action: {item['suggested_action']}",
        ])
    if not plan["outstanding_items"]:
        lines.append("- None reported.")
    lines.extend(["", "## Decisions and ambiguities", ""])
    for item in plan["decisions"]:
        decision = item["decision"] or "Unresolved"
        lines.append(
            f"- **{item['question']}** — {decision} "
            f"({item['status']}, source: {item['source']}). {item['rationale']}"
        )
    if not plan["decisions"]:
        lines.append("- None reported.")

    for title, key in (
        ("Acceptance criteria", "acceptance_criteria"),
        ("Verification", "verification"),
        ("Assumptions", "assumptions"),
    ):
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- {item}" for item in plan[key] or ["None reported."])

    lines.extend(["", "## Risks", ""])
    for risk in plan["risks"]:
        lines.append(
            f"- **{risk['likelihood']} likelihood / {risk['impact']} impact:** "
            f"{risk['description']} Mitigation: {risk['mitigation']}"
        )
    if not plan["risks"]:
        lines.append("- None reported.")

    if plan["revision"]["based_on"]:
        lines.extend(["", "## Revision", ""])
        lines.append(f"Based on: `{plan['revision']['based_on']}`")
        lines.extend(f"- {item}" for item in plan["revision"]["changes"])
    lines.append("")
    return {"report": "\n".join(lines)}
