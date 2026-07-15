import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from devflow.implementation.aider import (
    _aider_model,
    _repository_changes,
    build_aider_prompt,
)


class AiderImplementationTests(unittest.TestCase):
    def test_maps_openrouter_model_for_aider(self):
        config = SimpleNamespace(
            aider_model=None,
            model=SimpleNamespace(provider="openrouter", model="google/gemini-test"),
        )

        self.assertEqual(_aider_model(config), "openrouter/google/gemini-test")

    def test_configured_aider_model_takes_precedence(self):
        config = SimpleNamespace(
            aider_model="openrouter/custom/model",
            model=SimpleNamespace(provider="openrouter", model="ignored"),
        )

        self.assertEqual(_aider_model(config), "openrouter/custom/model")

    def test_prompt_requires_repository_inspection_and_plan_adherence(self):
        prompt = build_aider_prompt(
            {"objective": "Add history total."},
            {"implementation_investigations": [{"question": "Trace UI mapping."}]},
            ["server/history.ts"],
        )

        self.assertIn("Inspect the repository directly", prompt)
        self.assertIn("server/history.ts", prompt)
        self.assertIn("Trace UI mapping", prompt)
        self.assertIn("do not invent fields", prompt)

    def test_repository_cleanliness_ignores_devflow_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            import subprocess
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".devflow").mkdir()
            (root / ".devflow" / "run.log").write_text("log", encoding="utf-8")

            self.assertEqual(_repository_changes(temp_dir), [])


if __name__ == "__main__":
    unittest.main()
