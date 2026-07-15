import tempfile
import unittest
from pathlib import Path

try:
    from devflow.implementation.flow import (
        _apply_replacements,
        _planned_sources,
        _validate_replacements,
        ReplacementValidationError,
    )
except ModuleNotFoundError as error:
    if error.name == "prefect":
        raise unittest.SkipTest("Prefect is not installed in the test environment") from error
    raise

from devflow.implementation.grounding import grounding_preflight


class ImplementationEditTests(unittest.TestCase):
    def test_grounding_preflight_rejects_member_on_wrong_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "database.ts"
            target.write_text("interface Snapshot { statsSlice: number }\n", encoding="utf-8")
            plan = {"grounding_claims": [{
                "claim": "Player exposes statsSlice.",
                "scope": "code_ownership",
                "source": "repository",
                "status": "verified",
                "subject": "Player",
                "member": "statsSlice",
                "evidence": ["database.ts:Snapshot"],
                "remediation": "Trace or add the mapping to Player.",
            }]}

            failures = grounding_preflight(plan, temp_dir)

            self.assertEqual(failures[0]["reason"], "repository_grounding_not_found")
            self.assertIn("subject", failures[0]["missing"])

    def test_grounding_preflight_accepts_grounded_membership(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "player.ts"
            target.write_text("interface Player { statsSlice: number }\n", encoding="utf-8")
            plan = {"grounding_claims": [{
                "claim": "Player exposes statsSlice.",
                "scope": "type_membership",
                "source": "repository",
                "status": "verified",
                "subject": "Player",
                "member": "statsSlice",
                "evidence": ["player.ts:Player"],
                "remediation": "",
            }]}

            self.assertEqual(grounding_preflight(plan, temp_dir), [])

    def test_reads_and_applies_exact_planned_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "src" / "feature.py"
            target.parent.mkdir()
            target.write_text("enabled = False\n", encoding="utf-8")
            plan = {"proposed_changes": [{"path": "src/feature.py"}]}
            paths, sources = _planned_sources(plan, temp_dir)
            proposal = {"replacements": [{
                "path": "src/feature.py",
                "old_text": "enabled = False",
                "new_text": "enabled = True",
                "reason": "Enable the planned behavior.",
            }]}

            self.assertEqual(paths, ["src/feature.py"])
            self.assertIn("enabled = False", sources["src/feature.py"])
            _validate_replacements(proposal, paths, temp_dir)
            _apply_replacements(proposal, temp_dir)
            self.assertEqual(target.read_text(encoding="utf-8"), "enabled = True\n")

    def test_rejects_unplanned_file(self):
        proposal = {"replacements": [{
            "path": "secret.py", "old_text": "x", "new_text": "y", "reason": "",
        }]}
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "unplanned file"):
                _validate_replacements(proposal, ["planned.py"], temp_dir)

    def test_allows_ordered_replacements_in_same_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "feature.py"
            target.write_text("first = False\nsecond = False\n", encoding="utf-8")
            proposal = {"replacements": [
                {"path": "feature.py", "old_text": "first = False", "new_text": "first = True", "reason": ""},
                {"path": "feature.py", "old_text": "second = False", "new_text": "second = True", "reason": ""},
            ]}

            _validate_replacements(proposal, ["feature.py"], temp_dir)
            _apply_replacements(proposal, temp_dir)

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "first = True\nsecond = True\n",
            )

    def test_match_error_identifies_edit_count_and_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "feature.py"
            target.write_text("value = False\nvalue = False\n", encoding="utf-8")
            proposal = {"replacements": [{
                "path": "feature.py",
                "old_text": "value = False",
                "new_text": "value = True",
                "reason": "",
            }]}

            with self.assertRaises(ReplacementValidationError) as raised:
                _validate_replacements(proposal, ["feature.py"], temp_dir)

            self.assertEqual(raised.exception.details["edit_index"], 1)
            self.assertEqual(raised.exception.details["match_count"], 2)
            self.assertIn("matched 2 times", str(raised.exception))
