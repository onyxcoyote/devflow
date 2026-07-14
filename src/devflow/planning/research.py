from __future__ import annotations


MAX_SUPPLEMENTAL_CONTEXT_ROUNDS = 2


def repository_context_questions(plan: dict) -> list[dict[str, str]]:
    if plan.get("status") != "needs_repository_context":
        return []
    return [
        item
        for item in plan.get("outstanding_items", [])
        if item.get("kind") == "repository_context" and item.get("question")
    ]


def question_key(question: str) -> str:
    return " ".join(question.lower().split()).rstrip("?.!")


def supplemental_context_request(
    request: str,
    questions: list[dict[str, str]],
    round_number: int,
) -> str:
    lines = [
        "Perform a targeted supplemental repository investigation for implementation planning.",
        (
            "Return only new repository evidence that answers the questions below. Identify "
            "relevant files, symbols, call paths, and any repository facts that remain unresolved. "
            "For every planner question, add a question_resolutions entry when it is answered; "
            "otherwise retain it in missing_context. Do not make product decisions and do not "
            "repeat unrelated context."
        ),
        "",
        f"ORIGINAL DEVELOPMENT REQUEST\n{request}",
        "",
        f"PLANNER REPOSITORY QUESTIONS — ROUND {round_number}",
    ]
    for index, item in enumerate(questions, start=1):
        lines.extend([
            f"{index}. {item['question']}",
            f"   Impact: {item.get('impact', '')}",
            f"   Suggested investigation: {item.get('suggested_action', '')}",
        ])
    return "\n".join(lines)
