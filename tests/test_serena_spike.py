import json
import unittest
from types import SimpleNamespace

from devflow.planning.serena import (
    READ_ONLY_SERENA_TOOLS,
    SerenaSpikeConfig,
    SerenaContextReport,
    _call_signature,
    _bounded_transcript,
    _langchain_tools,
    _round_focus_instruction,
    _should_continue,
    _tool_result_text,
)


class SerenaToolFilteringTests(unittest.TestCase):
    def test_exposes_only_allowlisted_read_tools(self):
        tools = [
            SimpleNamespace(
                name="find_symbol",
                description="Find symbols",
                inputSchema={"type": "object"},
            ),
            SimpleNamespace(
                name="execute_shell_command",
                description="Execute shell",
                inputSchema={"type": "object"},
            ),
            SimpleNamespace(
                name="replace_symbol_body",
                description="Edit symbol",
                inputSchema={"type": "object"},
            ),
        ]

        converted = _langchain_tools(tools)
        names = [tool["function"]["name"] for tool in converted]

        self.assertEqual(names, ["find_symbol"])
        self.assertNotIn("execute_shell_command", READ_ONLY_SERENA_TOOLS)
        self.assertNotIn("replace_symbol_body", READ_ONLY_SERENA_TOOLS)


class SerenaResultFormattingTests(unittest.TestCase):
    def test_prefers_structured_content_and_truncates(self):
        result = SimpleNamespace(
            structuredContent={"symbols": ["one", "two"]},
            content=[],
        )

        text = _tool_result_text(result, 20)

        self.assertEqual(text, json.dumps(
            {"symbols": ["one", "two"]},
            ensure_ascii=False,
        )[:20])

    def test_joins_text_blocks(self):
        result = SimpleNamespace(
            structuredContent=None,
            content=[SimpleNamespace(text="first"), SimpleNamespace(text="second")],
        )

        self.assertEqual(_tool_result_text(result, 100), "first\nsecond")


class SerenaContinuationTests(unittest.TestCase):
    def config(self):
        return SerenaSpikeConfig(
            repo_path="/repo",
            output_dir="/output",
            command="serena",
            args=(),
            max_rounds=3,
            max_tool_calls_per_round=12,
            max_total_tool_calls=36,
            max_tool_result_chars=8000,
            max_transcript_chars=60000,
            max_report_output_tokens=5000,
            model=None,
        )

    def test_continues_for_repository_gaps_with_budget(self):
        report = {
            "status": "needs_repository_context",
            "missing_context": [{"kind": "repository"}],
        }

        self.assertTrue(_should_continue(
            report,
            round_number=1,
            total_tool_calls=12,
            config=self.config(),
        ))

    def test_stops_for_user_decision(self):
        report = {
            "status": "needs_user_decision",
            "missing_context": [{"kind": "user_decision"}],
        }

        self.assertFalse(_should_continue(
            report,
            round_number=1,
            total_tool_calls=8,
            config=self.config(),
        ))

    def test_stops_when_total_budget_is_exhausted(self):
        report = {
            "status": "needs_repository_context",
            "missing_context": [{"kind": "repository"}],
        }

        self.assertFalse(_should_continue(
            report,
            round_number=2,
            total_tool_calls=36,
            config=self.config(),
        ))

    def test_call_signature_ignores_argument_order(self):
        first = _call_signature("find_symbol", {"name": "Plan", "depth": 1})
        second = _call_signature("find_symbol", {"depth": 1, "name": "Plan"})

        self.assertEqual(first, second)

    def test_final_round_prioritizes_known_repository_gaps(self):
        instruction = _round_focus_instruction(True)

        self.assertIn("inspect that file first", instruction)
        self.assertIn("schema and type references", instruction)
        self.assertIn("preserve them for the human", instruction)

    def test_nonfinal_round_keeps_user_decisions_separate(self):
        instruction = _round_focus_instruction(False)

        self.assertIn("repository-answerable gaps", instruction)
        self.assertIn("user decisions", instruction)


class SerenaTranscriptTests(unittest.TestCase):
    def test_truncates_only_between_complete_events(self):
        events = [
            {"tool": "find_symbol", "result": "first"},
            {"tool": "read_file", "result": "x" * 500},
        ]

        text, included = _bounded_transcript(events, 100)

        self.assertEqual(included, 1)
        self.assertIn('"find_symbol"', text)
        self.assertNotIn('"read_file"', text)
        self.assertIn("1 of 2 complete events", text)


class SerenaReportSchemaTests(unittest.TestCase):
    def test_accepts_concise_structured_evidence(self):
        report = SerenaContextReport(
            status="sufficient",
            architecture_summary="Planning is coordinated by the planning graph.",
            evidence=[{
                "claim": "The graph invokes the planning node.",
                "source": "src/devflow/planning/graph.py:build_planning_graph",
            }],
        )

        self.assertEqual(report.evidence[0].claim, "The graph invokes the planning node.")

    def test_rejects_oversized_architecture_summary(self):
        with self.assertRaises(ValueError):
            SerenaContextReport(
                status="sufficient",
                architecture_summary="x" * 3001,
            )
