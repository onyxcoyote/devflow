import json
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from prefect import flow, get_run_logger

from devflow.repository_context.config import SerenaContextConfig
from devflow.repository_context.flow import serena_context_flow

from .config import PlanningConfig
from .artifacts import load_context_artifact, load_previous_plan
from .research import (
    MAX_SUPPLEMENTAL_CONTEXT_ROUNDS,
    MAX_INITIAL_CONTEXT_REFINEMENT_ROUNDS,
    apply_user_answers_to_context,
    context_user_questions,
    normalize_supplemental_report,
    merge_context_refinement,
    question_key,
    read_context_approved_files,
    repository_context_questions,
    supplemental_prior_report,
    supplemental_serena_config,
    supplemental_context_request,
    supplemental_progress_signature,
    user_decision_questions,
)
from .tasks import (
    run_planning_graph,
    save_plan_outputs,
)


def _planning_state(
    request: str,
    config: PlanningConfig,
    repository_context: dict,
    context_source: dict,
    previous_plan: dict | None,
) -> dict:
    return {
        "request": request,
        "repo_path": config.repo_path,
        "repository_context": repository_context,
        "approved_file_excerpts": read_context_approved_files(
            config.repo_path,
            repository_context,
        ),
        "context_source": context_source,
        "previous_plan": previous_plan,
        "context_text": "",
        "model_info": {
            "provider": config.model.provider,
            "model": config.model.model,
            "base_url": config.model.base_url,
            "temperature": config.model.temperature,
        },
        "model_result": {},
        "save_model_exchange": config.save_model_exchange,
        "model_exchange": {},
        "plan": {},
        "report": "",
    }


def _confirm(
    prompt: str,
    *,
    auto_approve: bool,
    logger,
) -> bool:
    if auto_approve:
        logger.info("Human gate auto-approved: %s", prompt)
        return True
    if not sys.stdin.isatty():
        logger.warning("Human gate declined because stdin is not interactive: %s", prompt)
        return False
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    approved = answer in {"y", "yes"}
    logger.info("Human gate %s: %s", "approved" if approved else "declined", prompt)
    return approved


def _log_supplemental_answers(report: dict, logger) -> None:
    resolutions = report.get("question_resolutions", [])
    logger.info("Supplemental repository answers (%d)", len(resolutions))
    if not resolutions:
        logger.warning("Serena returned no explicit question resolutions")
    for index, resolution in enumerate(resolutions, start=1):
        logger.info("Answer %d question: %s", index, resolution.get("question", ""))
        logger.info("Answer %d resolution: %s", index, resolution.get("resolution", ""))
        logger.info("Answer %d source: %s", index, resolution.get("source", ""))
    for item in report.get("missing_context", []):
        logger.warning(
            "Supplemental context remains unresolved: %s; suggested action: %s",
            item.get("description", ""),
            item.get("suggested_action", ""),
        )
    checkpoints = report.get("research_checkpoints", [])
    if checkpoints:
        logger.info("Supplemental research checkpoints (%d)", len(checkpoints))
    for checkpoint in checkpoints:
        logger.info(
            "Checkpoint %s: %s",
            checkpoint.get("status", "unknown"),
            checkpoint.get("subquestion", ""),
        )
        if checkpoint.get("partial_findings"):
            logger.info("  Progress: %s", checkpoint["partial_findings"])
        if checkpoint.get("sources_inspected"):
            logger.info("  Inspected: %s", ", ".join(checkpoint["sources_inspected"]))
        if checkpoint.get("next_investigation"):
            logger.info("  Next: %s", checkpoint["next_investigation"])


def _log_initial_context(report: dict, context_source: dict, logger) -> None:
    files = report.get("relevant_files", [])
    missing = report.get("missing_context", [])
    logger.info("Initial repository context: status=%s", report.get("status", "unknown"))
    logger.info("Initial context relevant files (%d)", len(files))
    for item in files:
        logger.info("  %s: %s", item.get("role", "unknown"), item.get("path", ""))
    if missing:
        logger.info("Initial context unresolved items (%d)", len(missing))
        for item in missing:
            logger.info("  %s: %s", item.get("kind", "unknown"), item.get("description", ""))
    artifact = context_source.get("context") or context_source.get("context_path")
    if artifact:
        logger.info("Initial context artifact: %s", artifact)


def _repository_gaps(report: dict) -> list[dict]:
    return [
        item for item in report.get("missing_context", [])
        if item.get("kind") == "repository"
    ]


def _context_completion_choice(*, can_continue: bool, auto_approve: bool) -> str:
    if auto_approve:
        return "continue" if can_continue else "stop"
    if not sys.stdin.isatty():
        return "stop"
    choices = (
        "[C]ontinue context research, add a research [H]int, "
        "[O]pen context for human review, [P]roceed anyway to planning, or [S]top"
    )
    while True:
        answer = input(f"Context is incomplete. {choices}? [S]: ").strip().lower()
        selected = {
            "c": "continue", "continue": "continue",
            "h": "hint", "hint": "hint",
            "o": "open", "open": "open", "review": "open",
            "p": "proceed", "proceed": "proceed",
            "s": "stop", "stop": "stop", "": "stop",
        }.get(answer)
        if selected == "continue" and not can_continue:
            print("The bounded context-refinement rounds are exhausted.")
            continue
        if selected:
            return selected


def _open_context_review(repository_context: dict, run_dir: str) -> str:
    path = Path(run_dir) / "context-review.json"
    path.write_text(json.dumps(repository_context, indent=2), encoding="utf-8")
    resolved = path.resolve()
    print(f"Context review: {resolved}")
    try:
        subprocess.Popen(["xdg-open", str(resolved)])
    except OSError:
        webbrowser.open(resolved.as_uri())
    return str(resolved)


def _collect_research_hints(gaps: list[dict], run_dir: str) -> tuple[list[str], str | None]:
    hints = []
    for item in gaps:
        hint = input(
            f"Optional search hint for: {item.get('description', '')}\n> "
        ).strip()
        hints.append(hint)
    if any(hints):
        return hints, None
    path = Path(run_dir) / "context-input.json"
    path.write_text(json.dumps({
        "research_hints": [
            {"question": item.get("description", ""), "hint": ""}
            for item in gaps
        ]
    }, indent=2), encoding="utf-8")
    print(f"Context hints: {path.resolve()}")
    return hints, str(path.resolve())


def _load_research_hints(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    hint_path = Path(path).expanduser().resolve()
    values = json.loads(hint_path.read_text(encoding="utf-8"))
    items = values.get("research_hints", []) if isinstance(values, dict) else values
    if not isinstance(items, list):
        raise ValueError("Context hints file must contain a research_hints list")
    return {
        question_key(str(item["question"])): str(item["hint"])
        for item in items
        if isinstance(item, dict) and item.get("question") and item.get("hint")
    }


def _refine_incomplete_context(
    request: str,
    repository_context: dict,
    context_source: dict,
    serena_config: SerenaContextConfig,
    *,
    run_dir: str,
    auto_approve: bool,
    logger,
    supplied_hints: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    refinement_round = 0
    supplied_hints = dict(supplied_hints or {})
    while (
        repository_context.get("status") == "needs_repository_context"
        or _repository_gaps(repository_context)
    ):
        gaps = _repository_gaps(repository_context)
        print("Repository context remains incomplete:")
        for index, item in enumerate(gaps, start=1):
            print(f"  {index}. {item.get('description', '')}")
            if item.get("suggested_action"):
                print(f"     Next: {item['suggested_action']}")
        can_continue = bool(gaps) and refinement_round < MAX_INITIAL_CONTEXT_REFINEMENT_ROUNDS
        matching_hints = [
            supplied_hints.get(question_key(item.get("description", "")), "")
            for item in gaps
        ]
        choice = (
            "hint" if any(matching_hints) and can_continue
            else _context_completion_choice(
                can_continue=can_continue,
                auto_approve=auto_approve,
            )
        )
        if choice == "proceed":
            context_source["incomplete_context_override"] = True
            return "proceed", None
        if choice == "open":
            _open_context_review(repository_context, run_dir)
            continue
        if choice == "stop":
            refined_path = Path(run_dir) / "context-refined.json"
            refined_path.write_text(json.dumps(repository_context, indent=2), encoding="utf-8")
            return "stop", str(refined_path.resolve())

        hints = matching_hints
        if choice == "hint":
            if not any(hints):
                hints, hint_path = _collect_research_hints(gaps, run_dir)
                if hint_path:
                    return "stop", hint_path
            supplied_hints.clear()
        refinement_round += 1
        questions = []
        for item, hint in zip(gaps, hints):
            action = item.get("suggested_action", "")
            if hint:
                action += f" User search hint (not evidence): {hint}"
            questions.append({
                "question": item.get("description", ""),
                "impact": "This repository gap prevents complete implementation context.",
                "suggested_action": action,
            })
        logger.info(
            "Running targeted initial-context refinement %d/%d for %d gap(s)",
            refinement_round,
            MAX_INITIAL_CONTEXT_REFINEMENT_ROUNDS,
            len(questions),
        )
        result = serena_context_flow(
            supplemental_context_request(request, questions, refinement_round),
            supplemental_serena_config(serena_config),
            initial_report=supplemental_prior_report(repository_context),
            active_questions=[item["question"] for item in questions],
        )
        report = normalize_supplemental_report(result["report"], questions)
        _log_supplemental_answers(report, logger)
        repository_context.setdefault("supplemental_rounds", []).append({
            "phase": "initial_context_refinement",
            "round": refinement_round,
            "questions": questions,
            "report": report,
        })
        context_source.setdefault("supplemental_rounds", []).append({
            "phase": "initial_context_refinement",
            "round": refinement_round,
            "paths": result.get("paths", {}),
        })
        merge_context_refinement(repository_context, report)
        time.sleep(serena_config.model_request_min_interval_seconds)
    return "complete", None


def _load_user_answers(path: str | None) -> list[dict[str, str]]:
    if path is None:
        return []
    answer_path = Path(path).expanduser().resolve()
    values = json.loads(answer_path.read_text(encoding="utf-8"))
    answers = values.get("answers", values) if isinstance(values, dict) else values
    if not isinstance(answers, list):
        raise ValueError("User answers file must contain a list or an answers list")
    return [
        {"question": str(item["question"]), "answer": str(item["answer"])}
        for item in answers
        if isinstance(item, dict) and item.get("question") and item.get("answer")
    ]


def _collect_user_answers(
    questions: list[dict[str, str]],
    *,
    auto_approve: bool,
    run_dir: str,
    logger,
) -> tuple[list[dict[str, str]], str | None]:
    print("Planning needs user decisions:")
    for index, item in enumerate(questions, start=1):
        print(f"  {index}. {item['question']}")
        if item.get("impact"):
            print(f"     Impact: {item['impact']}")
    answers = []
    if not auto_approve and sys.stdin.isatty():
        for item in questions:
            answer = input(f"Answer (blank to defer) — {item['question']}\n> ").strip()
            if answer:
                answers.append({"question": item["question"], "answer": answer})
    if len(answers) == len(questions):
        logger.info("Received %d user decision answer(s)", len(answers))
        return answers, None

    answered = {question_key(item["question"]) for item in answers}
    template = {
        "answers": [*answers, *[
            {"question": item["question"], "answer": ""}
            for item in questions
            if question_key(item["question"]) not in answered
        ]]
    }
    path = Path(run_dir) / "user-input.json"
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(f"Deferred user decisions: {path.resolve()}")
    print("Fill in each answer, then refine with --answers and --from-plan.")
    return answers, str(path.resolve())


@flow(name="development-plan")
def planning_flow(
    request: str,
    config: PlanningConfig,
    serena_config: SerenaContextConfig,
    context_path: str | None = None,
    previous_plan_path: str | None = None,
    run_dir: str | None = None,
    auto_approve: bool = False,
    answers_path: str | None = None,
    context_hints_path: str | None = None,
) -> dict:
    logger = get_run_logger()
    if run_dir is None:
        from .artifacts import create_plan_run_dir
        run_dir = str(create_plan_run_dir(config.output_dir))
    previous_plan, resolved_previous_plan_path = load_previous_plan(
        previous_plan_path
    )
    if context_path:
        repository_context, context_source = load_context_artifact(
            context_path,
            config.repo_path,
        )
    else:
        discovery_request = request
        if previous_plan is not None:
            discovery_request += (
                "\n\nReview this previous implementation plan as a draft. Investigate "
                "repository facts needed to confirm, correct, or complete it:\n"
                + json.dumps(previous_plan, ensure_ascii=False)
            )
        context_result = serena_context_flow(
            discovery_request,
            serena_config,
            gate_between_rounds=True,
            auto_approve=auto_approve,
        )
        repository_context = context_result["report"]
        context_source = {
            "mode": "generated",
            **context_result["paths"],
        }
        time.sleep(serena_config.model_request_min_interval_seconds)
    if resolved_previous_plan_path:
        context_source["previous_plan_path"] = resolved_previous_plan_path
    if not isinstance(repository_context, dict):
        repository_context = {"initial": repository_context}
    repository_context.setdefault("supplemental_rounds", [])
    context_source.setdefault("supplemental_rounds", [])
    supplied_answers = _load_user_answers(answers_path)
    supplied_hints = _load_research_hints(context_hints_path)
    if supplied_answers:
        apply_user_answers_to_context(repository_context, supplied_answers)
        context_source["user_answers_path"] = str(Path(answers_path).resolve())
    if context_hints_path:
        context_source["context_hints_path"] = str(Path(context_hints_path).resolve())

    _log_initial_context(repository_context, context_source, logger)
    completion, completion_artifact = _refine_incomplete_context(
        request,
        repository_context,
        context_source,
        serena_config,
        run_dir=run_dir,
        auto_approve=auto_approve,
        logger=logger,
        supplied_hints=supplied_hints,
    )
    if completion == "stop":
        logger.info("Planning stopped with incomplete repository context")
        return {
            "stopped": True,
            "reason": "repository_context_incomplete",
            "context_source": context_source,
            "repository_context": repository_context,
            "context_input_path": completion_artifact,
        }
    context_questions = context_user_questions(repository_context)
    if context_questions:
        answers, user_input_path = _collect_user_answers(
            context_questions,
            auto_approve=auto_approve,
            run_dir=run_dir,
            logger=logger,
        )
        if answers:
            apply_user_answers_to_context(repository_context, answers)
        if user_input_path is not None:
            logger.info("Planning stopped for unresolved context-level user decisions")
            return {
                "stopped": True,
                "reason": "context_user_decision_deferred",
                "context_source": context_source,
                "repository_context": repository_context,
                "user_input_path": user_input_path,
            }
        _log_initial_context(repository_context, context_source, logger)
    if completion != "proceed":
        print(
            "Repository context is ready for the planning gate: "
            f"status={repository_context.get('status', 'unknown')}; "
            f"repository_gaps={len(_repository_gaps(repository_context))}"
        )
    if completion != "proceed" and not _confirm(
            "Proceed to the planning stage with this repository context?",
            auto_approve=auto_approve,
            logger=logger,
        ):
        context_source.setdefault("human_gates", []).append({
            "gate": "initial_context_to_plan",
            "approved": False,
        })
        logger.info("Planning stopped by user after initial context")
        return {
            "stopped": True,
            "reason": "initial_context_not_approved",
            "context_source": context_source,
            "repository_context": repository_context,
        }
    context_source.setdefault("human_gates", []).append({
        "gate": "initial_context_to_plan",
        "approved": True,
    })

    final_state = run_planning_graph(
        _planning_state(
            request,
            config,
            repository_context,
            context_source,
            previous_plan,
        ),
        config,
    )
    seen_questions = set()
    seen_supplemental_reports = set()
    previous_progress_signature = ()
    planning_rounds = []
    supplemental_rounds_completed = 0
    user_input_path = None

    for round_number in range(1, MAX_SUPPLEMENTAL_CONTEXT_ROUNDS + 1):
        decisions = user_decision_questions(final_state["plan"])
        if decisions:
            answers, user_input_path = _collect_user_answers(
                decisions,
                auto_approve=auto_approve,
                run_dir=run_dir,
                logger=logger,
            )
            if answers:
                repository_context.setdefault("user_answers", []).extend(answers)
                logger.info("Retrying planning with user answers")
                final_state = run_planning_graph(
                    _planning_state(request, config, repository_context, context_source, previous_plan),
                    config,
                )
                if user_input_path is None:
                    continue
            break
        questions = repository_context_questions(final_state["plan"])
        planning_rounds.append({
            "round": round_number,
            "status": final_state["plan"]["status"],
            "repository_questions": questions,
            "plan_attempts": final_state["model_result"].get("plan_attempts", []),
        })
        new_questions = [
            item for item in questions
            if question_key(item["question"]) not in seen_questions
        ]
        if not new_questions:
            if questions:
                logger.warning("Planner repeated previously investigated questions")
                new_questions = questions
            elif final_state["plan"]["status"] == "needs_repository_context":
                logger.warning(
                    "Stopping supplemental context: planner supplied no repository questions"
                )
            if not new_questions:
                break

        logger.info(
            "Planning requested additional repository context (round %d/%d)",
            round_number,
            MAX_SUPPLEMENTAL_CONTEXT_ROUNDS,
        )
        for index, item in enumerate(new_questions, start=1):
            logger.info("Context question %d: %s", index, item["question"])
            seen_questions.add(question_key(item["question"]))

        if not _confirm(
            "Send these repository questions to Serena context?",
            auto_approve=auto_approve,
            logger=logger,
        ):
            context_source.setdefault("human_gates", []).append({
                "round": round_number,
                "gate": "send_questions",
                "approved": False,
            })
            logger.info("Research stopped by user before supplemental context")
            break
        context_source.setdefault("human_gates", []).append({
            "round": round_number,
            "gate": "send_questions",
            "approved": True,
        })

        supplemental_request = supplemental_context_request(
            request,
            new_questions,
            round_number,
        )
        logger.info("Sending %d targeted question(s) to Serena context", len(new_questions))
        supplemental_result = serena_context_flow(
            supplemental_request,
            supplemental_serena_config(serena_config),
            initial_report=supplemental_prior_report(repository_context),
            active_questions=[item["question"] for item in new_questions],
        )
        supplemental_report = normalize_supplemental_report(
            supplemental_result["report"],
            new_questions,
        )
        _log_supplemental_answers(supplemental_report, logger)
        progress_signature = supplemental_progress_signature(supplemental_report)
        made_progress = bool(progress_signature) and progress_signature != previous_progress_signature
        if progress_signature:
            previous_progress_signature = progress_signature
        report_key = json.dumps(
            supplemental_report,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        if report_key in seen_supplemental_reports:
            logger.warning("Stopping supplemental context: Serena returned no new evidence")
            break
        seen_supplemental_reports.add(report_key)

        repository_context["supplemental_rounds"].append({
            "round": round_number,
            "questions": new_questions,
            "report": supplemental_report,
        })
        context_source["supplemental_rounds"].append({
            "round": round_number,
            "questions": new_questions,
            "paths": supplemental_result.get("paths", {}),
        })
        supplemental_rounds_completed += 1
        if not made_progress and supplemental_report.get("missing_context"):
            logger.warning(
                "Stopping supplemental context: no checkpoint progress was recorded"
            )
            break
        if not _confirm(
            "Proceed to planning with these supplemental answers?",
            auto_approve=auto_approve,
            logger=logger,
        ):
            context_source.setdefault("human_gates", []).append({
                "round": round_number,
                "gate": "refine_plan",
                "approved": False,
            })
            logger.info("Planning refinement stopped by user after supplemental context")
            break
        context_source.setdefault("human_gates", []).append({
            "round": round_number,
            "gate": "refine_plan",
            "approved": True,
        })
        logger.info(
            "Supplemental context round %d completed; retrying planning with added evidence",
            round_number,
        )
        time.sleep(serena_config.model_request_min_interval_seconds)
        final_state = run_planning_graph(
            _planning_state(
                request,
                config,
                repository_context,
                context_source,
                previous_plan,
            ),
            config,
        )
    else:
        planning_rounds.append({
            "round": MAX_SUPPLEMENTAL_CONTEXT_ROUNDS + 1,
            "status": final_state["plan"]["status"],
            "repository_questions": repository_context_questions(final_state["plan"]),
            "plan_attempts": final_state["model_result"].get("plan_attempts", []),
        })
        remaining_questions = repository_context_questions(final_state["plan"])
        if remaining_questions:
            logger.warning(
                "Stopping supplemental context: research budget exhausted with %d question(s) remaining",
                len(remaining_questions),
            )

    final_state["model_result"]["planning_rounds"] = planning_rounds
    final_state["context_source"] = context_source
    final_state["repository_context"] = repository_context
    logger.info(
        "Planning research completed after %d supplemental context round(s); final status=%s",
        supplemental_rounds_completed,
        final_state["plan"]["status"],
    )
    paths = save_plan_outputs(final_state, run_dir)
    return {
        "plan": final_state["plan"],
        "paths": paths,
        "user_input_path": user_input_path,
    }
