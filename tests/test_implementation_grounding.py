import tempfile
import unittest
from pathlib import Path

from devflow.implementation.grounding import grounding_preflight


class ImplementationGroundingTests(unittest.TestCase):
    def test_rejects_member_on_wrong_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "database.ts").write_text(
                "interface Snapshot { statsSlice: number }\n", encoding="utf-8"
            )
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

    def test_accepts_grounded_membership(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "player.ts").write_text(
                "interface Player { statsSlice: number }\n", encoding="utf-8"
            )
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


if __name__ == "__main__":
    unittest.main()
