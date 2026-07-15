from __future__ import annotations

import re
from pathlib import Path


def grounding_preflight(plan: dict, repo_path: str) -> list[dict]:
    root = Path(repo_path).resolve()
    failures = []
    relationship_scopes = {"code_ownership", "code_availability", "type_membership"}
    code_scopes = relationship_scopes | {"data_flow", "current_behavior"}
    for claim in plan.get("grounding_claims", []):
        if claim.get("scope") not in code_scopes or claim.get("status") == "proposed":
            continue
        if claim.get("status") != "verified" or claim.get("source") != "repository":
            failures.append({
                "claim": claim.get("claim", ""),
                "reason": "unsupported_existing_code_claim",
                "remediation": claim.get("remediation", "Return to repository context."),
            })
            continue
        evidence_text = ""
        evidence_paths = []
        for reference in claim.get("evidence", []):
            relative = reference.split(":", 1)[0]
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if not candidate.is_file():
                continue
            try:
                content = candidate.read_text(encoding="utf-8")[:40_000]
            except (OSError, UnicodeError):
                continue
            evidence_paths.append(relative)
            evidence_text += "\n" + content
        subject = claim.get("subject", "").strip()
        member = claim.get("member", "").strip()
        missing = []
        if not evidence_paths:
            missing.append("evidence_file")
        if claim.get("scope") in relationship_scopes:
            if subject and not re.search(rf"\b{re.escape(subject)}\b", evidence_text):
                missing.append("subject")
            if member and not re.search(rf"\b{re.escape(member)}\b", evidence_text):
                missing.append("member")
        evidence_symbols = [
            reference.split(":", 1)[1]
            for reference in claim.get("evidence", []) if ":" in reference
        ]
        if (
            claim.get("scope") in relationship_scopes
            and subject
            and not any(
                re.search(rf"\b{re.escape(subject)}\b", symbol)
                for symbol in evidence_symbols
            )
        ):
            missing.append("evidence_subject")
        if claim.get("scope") in relationship_scopes and subject and member and not missing:
            subject_positions = [
                item.start() for item in re.finditer(rf"\b{re.escape(subject)}\b", evidence_text)
            ]
            member_positions = [
                item.start() for item in re.finditer(rf"\b{re.escape(member)}\b", evidence_text)
            ]
            if not any(
                abs(left - right) <= 4_000
                for left in subject_positions for right in member_positions
            ):
                missing.append("subject_member_relationship")
        if missing:
            failures.append({
                "claim": claim.get("claim", ""),
                "subject": subject,
                "member": member,
                "evidence": claim.get("evidence", []),
                "reason": "repository_grounding_not_found",
                "validation_mode": (
                    "subject_member_relationship"
                    if claim.get("scope") in relationship_scopes else "evidence_file"
                ),
                "missing": missing,
                "remediation": claim.get("remediation", "Trace and map the missing value."),
            })
    return failures
