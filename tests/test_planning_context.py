import tempfile
import unittest
from pathlib import Path

from devflow.planning.context import directory_summary, gather_requested_context


class DirectorySummaryTests(unittest.TestCase):
    def test_collapses_large_asset_directories(self):
        tracked_files = [f"assets/image-{index}.png" for index in range(60)]
        tracked_files += [
            "src/devflow/cli.py",
            "src/devflow/planning/nodes.py",
            "README.md",
        ]

        summary = directory_summary(tracked_files)

        self.assertIn("assets/ (60 files)", summary)
        self.assertIn("cli.py", summary)
        self.assertNotIn("image-0.png", summary)

    def test_enforces_global_summary_limit(self):
        tracked_files = [f"module-{index}/file.py" for index in range(500)]

        summary = directory_summary(tracked_files, max_lines=25)

        self.assertLessEqual(len(summary.splitlines()), 26)
        self.assertIn("directory summary truncated", summary)


class GatherRequestedContextTests(unittest.TestCase):
    def test_reads_only_valid_tracked_files_and_runs_literal_searches(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "src").mkdir()
            (repo / "src/app.py").write_text(
                "class HistoryService:\n    pass\n",
                encoding="utf-8",
            )
            (repo / "secret.txt").write_text("not tracked", encoding="utf-8")
            tracked_files = ["src/app.py"]

            result = gather_requested_context(
                str(repo),
                tracked_files,
                {
                    "files": ["src/app.py", "../secret.txt", "secret.txt"],
                    "searches": ["HistoryService"],
                    "reason": "Find the service.",
                },
                max_requested_files=5,
                max_searches=5,
                max_context_chars=1000,
                max_search_results_chars=1000,
            )

        self.assertEqual(list(result["selected_files"]), ["src/app.py"])
        self.assertEqual(
            result["search_results"]["HistoryService"][0]["file"],
            "src/app.py",
        )

    def test_enforces_file_and_search_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for name in ("one.py", "two.py"):
                (repo / name).write_text("target\n", encoding="utf-8")

            result = gather_requested_context(
                str(repo),
                ["one.py", "two.py"],
                {
                    "files": ["one.py", "two.py"],
                    "searches": ["target", "other"],
                    "reason": "Test limits.",
                },
                max_requested_files=1,
                max_searches=1,
                max_context_chars=1000,
                max_search_results_chars=1000,
            )

        self.assertEqual(list(result["selected_files"]), ["one.py"])
        self.assertEqual(list(result["search_results"]), ["target"])
