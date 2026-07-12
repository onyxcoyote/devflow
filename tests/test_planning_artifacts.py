import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from devflow.planning.artifacts import load_context_artifact, load_previous_plan


class PlanningArtifactTests(unittest.TestCase):
    def make_repo(self, directory: str) -> tuple[Path, str]:
        repo = Path(directory)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo,
            check=True,
        )
        (repo / "tracked.txt").write_text("test", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return repo, head

    def write_context(self, repo: Path, head: str) -> Path:
        run = repo / "artifacts"
        run.mkdir()
        context_path = run / "context.json"
        context_path.write_text('{"status": "sufficient"}', encoding="utf-8")
        (run / "evidence.json").write_text(
            json.dumps({"repo_path": str(repo), "head_commit": head}),
            encoding="utf-8",
        )
        return context_path

    def test_loads_context_for_matching_repository_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, head = self.make_repo(directory)
            context, source = load_context_artifact(
                str(self.write_context(repo, head)),
                str(repo),
            )

            self.assertEqual(context["status"], "sufficient")
            self.assertEqual(source["head_commit"], head)

    def test_rejects_stale_context(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, _ = self.make_repo(directory)
            context_path = self.write_context(repo, "old-commit")

            with self.assertRaisesRegex(ValueError, "Context is stale"):
                load_context_artifact(str(context_path), str(repo))

    def test_loads_previous_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text('{"status": "ready"}', encoding="utf-8")

            plan, resolved = load_previous_plan(str(path))

            self.assertEqual(plan["status"], "ready")
            self.assertEqual(resolved, str(path.resolve()))
