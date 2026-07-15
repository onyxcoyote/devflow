from __future__ import annotations

import json

from .research import flatten_context_for_planning
from time import perf_counter

from devflow.code_review.nodes import _model_result_metadata, _raw_response_data

from .schemas import DevelopmentPlan, PLAN_SCHEMA_VERSION, PLAN_STRUCTURED_OUTPUT_METHOD
from .state import PlanningState


def _configured_output_limit(model):
    for name in ("max_tokens", "num_predict", "max_output_tokens"):
        value = getattr(model, name, None)
        if value is not None:
            return value
    model_kwargs = getattr(model, "model_kwargs", {}) or {}
    return (
        model_kwargs.get("max_tokens")
        or model_kwargs.get("num_predict")
        or model_kwargs.get("max_output_tokens")
    )


def _serialize_partial_completion(error):
    completion = getattr(error, "completion", None)
    if completion is None:
        return None
    model_dump = getattr(completion, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except (TypeError, ValueError):
            return model_dump()
    return str(completion)


def _plan_quality_issue(plan: dict) -> str | None:
    if plan["status"] in {"ready", "needs_user_decision"} and not plan[
        "proposed_changes"
    ]:
        return (
            f"status={plan['status']} requires concrete proposed_changes; "
            "provide the recommended implementation path even when a user decision remains"
        )
    if plan["status"] == "ready":
        code_scopes = {
            "code_ownership", "code_availability", "type_membership",
            "data_flow", "current_behavior",
        }
        unsupported = [
            item.get("claim", "") for item in plan.get("grounding_claims", [])
            if item.get("scope") in code_scopes and (
                item.get("status") == "unverified"
                or (
                    item.get("status") == "verified"
                    and item.get("source") != "repository"
                )
                or (
                    item.get("status") == "verified"
                    and not item.get("evidence")
                )
            )
        ]
        if unsupported:
            return (
                "status=ready contains unsupported existing-code claims: "
                + "; ".join(unsupported)
                + ". Request repository context or explicitly propose the missing mapping/change"
            )
        if plan.get("outstanding_items"):
            return "status=ready cannot contain outstanding_items"
        unresolved = [
            item.get("question", "") for item in plan.get("decisions", [])
            if item.get("status") == "unresolved"
        ]
        if unresolved:
            return "status=ready cannot contain unresolved decisions: " + "; ".join(unresolved)
        deferred_phrases = (
            "consider whether", "decide whether", "determine whether", "choose between",
            "either approach", "a or b",
        )
        decision_texts = [plan.get("design_summary", "")]
        for change in plan.get("proposed_changes", []):
            decision_texts.extend([change.get("change", ""), change.get("reason", "")])
        for decision in plan.get("decisions", []):
            decision_texts.extend([decision.get("decision", ""), decision.get("rationale", "")])
        deferred = next(
            (phrase for text in decision_texts for phrase in deferred_phrases if phrase in text.lower()),
            None,
        )
        if deferred:
            return (
                f"status=ready defers a known decision using '{deferred}'; choose and justify an "
                "approach, request specific repository context, or request a user decision"
            )
    return None


def prepare_plan_context(state: PlanningState) -> dict:
    approved_file_excerpts = state.get("approved_file_excerpts", {})
    return {
        "context_text": json.dumps(
            {
                "report": flatten_context_for_planning(state["repository_context"]),
                "context_approved_file_excerpts": approved_file_excerpts,
            },
            indent=2,
            ensure_ascii=False,
        )
    }


def make_plan_node(model, compact_retry_model, logger):
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
            "evidence source. Source excerpts are provided only for files approved by context; "
            "use them directly for detailed schemas and signatures instead of requesting the "
            "same repository facts again. Design recommendations may be planner reasoning.\n"
            "Audit semantic claims, not only dotted identifiers. Statements such as Player's "
            "statsSlice, available on Player, or HistoryService provides the total are ownership, "
            "availability, or data-flow claims. Record every material claim in grounding_claims. "
            "A human statement that data exists in a database does not verify that a particular "
            "object exposes a field. Existing-code claims are verified only with repository "
            "path:symbol evidence. If the bridge is absent, mark it unverified and request context, "
            "or mark it proposed and include the concrete mapping/type changes.\n"
            "Each proposed change must name one file and describe its responsibility. Include "
            "tests and validation in verification and acceptance criteria.\n"
            "Make ordinary engineering decisions when repository evidence supports them. The "
            "existence of multiple reasonable approaches is not by itself a blocker: choose and "
            "justify a recommended approach.\n"
            "Be concise. Do not repeat the repository context, source code, or the same change "
            "in multiple fields. Prefer short actionable statements.\n"
            "Use needs_repository_context when a specific repository question prevents a "
            "responsible plan. Use needs_user_decision only when a choice materially changes "
            "external behavior, compatibility, security, data handling, or product requirements "
            "and cannot responsibly be inferred. For needs_user_decision, still provide concrete "
            "file changes for the recommended default and explain how the alternative differs. "
            "Use not_feasible when the requested outcome cannot responsibly be planned, and ready "
            "only when the plan is actionable. Preserve genuine unresolved questions explicitly "
            "in outstanding_items and decisions. ready and needs_user_decision must both include "
            "at least one proposed change.\n\n"
            f"DEVELOPMENT REQUEST\n===================\n{state['request']}\n\n"
            f"REPOSITORY CONTEXT\n==================\n{state['context_text']}\n\n"
            f"PREVIOUS PLAN\n=============\n{previous_text}"
        )
        prompts = [
            prompt,
            (
                "The previous planning response was truncated, invalid, or failed. Return an "
                "especially compact plan. Include only actionable file changes, blockers, key "
                "decisions, acceptance criteria, verification, and material risks. Do not repeat "
                "context or explanatory prose.\n\n" + prompt
            ),
        ]
        models = [model, compact_retry_model]
        attempt_metadata = []
        model_exchange = dict(state["model_exchange"])
        exchange_attempts = []
        plan = None
        failure_details = []

        for attempt, (attempt_model, attempt_prompt) in enumerate(
            zip(models, prompts),
            start=1,
        ):
            started_at = perf_counter()
            configured_limit = _configured_output_limit(attempt_model)
            exchange_attempt = {
                "attempt": attempt,
                "configured_output_limit": configured_limit,
                "request": {"prompt": attempt_prompt},
            }
            logger.info(
                "Preparing structured planning model for attempt %d (%s); client_output_limit=%s",
                attempt,
                "initial" if attempt == 1 else "compact retry",
                configured_limit if configured_limit is not None else "unknown",
            )
            try:
                structured_model = attempt_model.with_structured_output(
                    DevelopmentPlan,
                    include_raw=True,
                    method=PLAN_STRUCTURED_OUTPUT_METHOD,
                )
                logger.info(
                    "Invoking planning model attempt %d; context_chars=%d previous_plan=%s",
                    attempt,
                    len(state["context_text"]),
                    previous_plan is not None,
                )
                result = structured_model.invoke(attempt_prompt)
                elapsed_seconds = perf_counter() - started_at
                raw_response = result["raw"]
                parsed_plan = result["parsed"]
                parsing_error = result["parsing_error"]
                metadata = _model_result_metadata(
                    raw_response,
                    parsing_error,
                    elapsed_seconds,
                )
                metadata["configured_output_limit"] = configured_limit
                metadata["structured_output_method"] = PLAN_STRUCTURED_OUTPUT_METHOD
                metadata["schema_version"] = PLAN_SCHEMA_VERSION
                attempt_metadata.append(metadata)
                if state["save_model_exchange"]:
                    exchange_attempt["response"] = _raw_response_data(raw_response)
                truncated = metadata["finish_reason"] in {"length", "max_tokens"}
                logger.info(
                    "Planning attempt %d completed in %.1fs: finish_reason=%s total_tokens=%s parsed=%s",
                    attempt,
                    elapsed_seconds,
                    metadata["finish_reason"],
                    metadata.get("total_tokens") or "unknown",
                    parsed_plan is not None and parsing_error is None,
                )
                if not truncated and parsing_error is None and parsed_plan is not None:
                    candidate_plan = parsed_plan.model_dump()
                    quality_issue = _plan_quality_issue(candidate_plan)
                    metadata["quality_issue"] = quality_issue
                    if quality_issue is None:
                        plan = candidate_plan
                        break
                    failure_details.append(quality_issue)
                    logger.warning(
                        "Planning attempt %d failed quality validation: %s",
                        attempt,
                        quality_issue,
                    )
                    if attempt == 1:
                        prompts[1] = (
                            "The previous response was structurally valid but not actionable: "
                            f"{quality_issue}. Correct that issue in an especially compact plan. "
                            "Make reasonable engineering decisions, include concrete file changes, "
                            "and reserve unresolved items for genuinely blocking questions.\n\n"
                            + prompt
                        )
                else:
                    if parsing_error is not None:
                        logger.warning(
                            "Planning attempt %d response validation failed: %s: %s",
                            attempt,
                            type(parsing_error).__name__,
                            str(parsing_error)[:4000],
                        )
                    failure_details.append(
                        "output limit reached" if truncated else (
                            metadata["parsing_error"] or "no parsed plan returned"
                        )
                    )
            except Exception as error:
                elapsed_seconds = perf_counter() - started_at
                failure = {
                    "attempt": attempt,
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "exception_type": type(error).__name__,
                    "exception_message": str(error)[:2000],
                    "configured_output_limit": configured_limit,
                    "structured_output_method": PLAN_STRUCTURED_OUTPUT_METHOD,
                    "schema_version": PLAN_SCHEMA_VERSION,
                }
                partial_completion = _serialize_partial_completion(error)
                failure["partial_completion_captured"] = (
                    partial_completion is not None
                )
                attempt_metadata.append(failure)
                failure_details.append(
                    f"{type(error).__name__}: {str(error)[:500]}"
                )
                logger.warning(
                    "Planning attempt %d failed after %.1fs: %s: %s",
                    attempt,
                    elapsed_seconds,
                    type(error).__name__,
                    str(error)[:4000],
                )
                if state["save_model_exchange"]:
                    exchange_attempt["error"] = {
                        **failure,
                        "partial_completion": partial_completion,
                    }
            finally:
                if state["save_model_exchange"]:
                    exchange_attempts.append(exchange_attempt)

        model_result = {
            **state["model_result"],
            "plan": attempt_metadata[-1],
            "plan_attempts": attempt_metadata,
        }
        if state["save_model_exchange"]:
            model_exchange["plan_attempts"] = exchange_attempts
        if plan is None:
            logger.error("Planning failed after two bounded attempts")
            plan = DevelopmentPlan(
                status="blocked",
                objective=state["request"],
                design_summary="The planning model did not return a valid bounded plan.",
                assumptions=[],
                proposed_changes=[],
                outstanding_items=[{
                    "kind": "external_information",
                    "question": "; ".join(failure_details)[:1500],
                    "impact": "No reliable structured implementation plan was produced.",
                    "suggested_action": "Inspect plan_attempts in evidence.json and retry with a different model or limits.",
                }],
                decisions=[],
                grounding_claims=[],
                acceptance_criteria=[],
                verification=[],
                risks=[],
                revision={"based_on": None, "changes": []},
            ).model_dump()
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

    lines.extend(["", "## Grounding claims", ""])
    for item in plan.get("grounding_claims", []):
        evidence = ", ".join(item.get("evidence", [])) or "None"
        lines.append(
            f"- **{item['status']} {item['scope']}:** {item['claim']} "
            f"(source: {item['source']}; evidence: {evidence})"
        )
    if not plan.get("grounding_claims"):
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
