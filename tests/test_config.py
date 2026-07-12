import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devflow.code_review.config import load_code_review_config
from devflow.planning.config import load_planning_config
from devflow.repository_context.config import load_serena_context_config


class SharedConfigTests(unittest.TestCase):
    def test_feature_loaders_share_model_and_source_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            config_path = repo / ".devflow.toml"
            config_path.write_text(
                """
[model]
provider = "ollama"
model = "test-model"

[review]
output_dir = "review-output"

[plan]
output_dir = "plan-output"

[serena]
output_dir = "serena-output"
""".strip(),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"DEVFLOW_GLOBAL_CONFIG": str(repo / "missing.toml")}):
                review = load_code_review_config(repo)
                plan = load_planning_config(repo)
                serena = load_serena_context_config(repo)

            self.assertEqual(review.model, plan.model)
            self.assertEqual(plan.model, serena.model)
            self.assertEqual(review.config_sources, (str(config_path),))
            self.assertEqual(plan.config_sources, review.config_sources)
            self.assertEqual(serena.config_sources, review.config_sources)
            self.assertEqual(review.output_dir, str(repo / "review-output"))
            self.assertEqual(plan.output_dir, str(repo / "plan-output"))
            self.assertEqual(plan.max_output_tokens, 8000)
            self.assertEqual(plan.compact_retry_output_tokens, 4000)
            self.assertEqual(serena.output_dir, str(repo / "serena-output"))

    def test_serena_uses_context_output_default(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "DEVFLOW_GLOBAL_CONFIG": str(repo / "missing.toml"),
                    "OLLAMA_MODEL": "test-model",
                },
            ):
                config = load_serena_context_config(repo)

            self.assertEqual(
                config.output_dir,
                str(repo / ".devflow" / "serena-context"),
            )


if __name__ == "__main__":
    unittest.main()
