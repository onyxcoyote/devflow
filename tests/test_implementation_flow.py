import tempfile
import unittest
from pathlib import Path

try:
    from devflow.implementation.flow import (
        _apply_replacements,
        _planned_sources,
        _validate_replacements,
    )
except ModuleNotFoundError as error:
    if error.name == "prefect":
        raise unittest.SkipTest("Prefect is not installed in the test environment") from error
    raise


class ImplementationEditTests(unittest.TestCase):
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
