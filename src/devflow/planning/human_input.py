from __future__ import annotations

import json
import re
from pathlib import Path


def _question_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def default_research_disposition(item: dict) -> str:
    authority = item.get("authority", "unclassified")
    status = item.get("status", "unresolved")
    if status in {"self_answered", "resolved"}:
        return "resolved"
    if authority in {"authoritative_requirement", "explicit_assumption"} and item.get("answer"):
        return "resolved"
    if authority == "implementation_hypothesis":
        return "implementation_investigation"
    if authority in {"repository_hint", "documentation_hint", "unverified_fact"}:
        return "research_next"
    if status == "human_answered" and item.get("answer"):
        return "resolved"
    return "research_next"


def reconcile_human_input_entries(*groups: list[dict]) -> list[dict]:
    allowed = {
        "resolved", "research_next", "implementation_investigation",
        "blocks_planning", "optional", "out_of_scope", "superseded",
    }
    merged: dict[str, dict] = {}
    for group in groups:
        for raw in group or []:
            if not isinstance(raw, dict) or not raw.get("question"):
                continue
            key = _question_key(str(raw["question"]))
            current = merged.setdefault(key, {"question": str(raw["question"])})
            for field in ("status", "answer", "source", "authority", "resolution"):
                if raw.get(field) not in (None, ""):
                    current[field] = raw[field]
            disposition = raw.get("disposition")
            if disposition in allowed:
                current["disposition"] = disposition
    for item in merged.values():
        item.setdefault("status", "unresolved")
        item.setdefault("answer", "")
        item.setdefault("source", "human input ledger")
        item.setdefault("authority", "unclassified")
        item.setdefault("disposition", default_research_disposition(item))
        if item["disposition"] == "resolved" and not item.get("resolution"):
            item["resolution"] = item.get("answer", "")
    return list(merged.values())


def load_human_input_ledger(path: str | None) -> dict:
    if not path:
        return {"entries": [], "architecture_decisions": []}
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Human input ledger must be a JSON object")
    entries = reconcile_human_input_entries(
        value.get("clarification_answers", []),
        value.get("inquiry_ledger", []),
        value.get("active_research", []),
        value.get("implementation_investigations", []),
    )
    return {
        "path": str(resolved),
        "entries": entries,
        "architecture_decisions": value.get("approved_architecture_decisions", []),
    }


def apply_human_input_ledger(report: dict, ledger: dict) -> dict[str, str]:
    entries = ledger.get("entries", [])
    if not entries:
        return {}
    brief = report.setdefault("research_brief", {})
    brief["clarification_answers"] = reconcile_human_input_entries(
        brief.get("clarification_answers", []), entries
    )
    report["inquiry_ledger"] = reconcile_human_input_entries(
        report.get("inquiry_ledger", []), entries
    )
    active_hints = {}
    disposition_by_key = {
        _question_key(item["question"]): item["disposition"] for item in entries
    }
    report["missing_context"] = [
        item for item in report.get("missing_context", [])
        if disposition_by_key.get(_question_key(item.get("description", "")))
        in {None, "research_next", "blocks_planning"}
    ]
    existing_gaps = {
        _question_key(item.get("description", ""))
        for item in report.get("missing_context", [])
    }
    for item in entries:
        disposition = item["disposition"]
        key = _question_key(item["question"])
        if disposition in {"research_next", "blocks_planning"}:
            if key not in existing_gaps:
                report.setdefault("missing_context", []).append({
                    "kind": "repository",
                    "description": item["question"],
                    "suggested_action": "Resolve this reused research item with path:symbol evidence.",
                    "related_files": [],
                    "related_symbols": [],
                })
                existing_gaps.add(key)
            if item.get("answer"):
                active_hints[key] = item["answer"]
        elif disposition == "implementation_investigation":
            report["implementation_investigations"] = reconcile_human_input_entries(
                report.get("implementation_investigations", []), [item]
            )
        elif disposition == "resolved" and item.get("resolution"):
            resolutions = report.setdefault("question_resolutions", [])
            if not any(
                _question_key(existing.get("question", "")) == key
                for existing in resolutions
            ):
                resolutions.append({
                    "question": item["question"],
                    "resolution": item["resolution"],
                    "source": f"human input ledger ({item.get('authority', 'unclassified')})",
                })
    if any(item["disposition"] in {"research_next", "blocks_planning"} for item in entries):
        report["status"] = "needs_repository_context"
    elif report.get("status") == "needs_repository_context" and not report.get("missing_context"):
        report["status"] = "sufficient"
    return active_hints
