import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from devflow.repository_context.config import SerenaContextConfig
from devflow.repository_context.serena import (
    READ_ONLY_SERENA_TOOLS,
    SERENA_SCHEMA_VERSION,
    SERENA_STRUCTURED_OUTPUT_METHOD,
    SerenaContextRunError,
    SerenaContextReport,
    _call_signature,
    _bounded_transcript,
    _langchain_tools,
    _round_focus_instruction,
    _should_continue,
    _tool_result_text,
    _ModelRequestLimiter,
    _confirm_additional_context_round,
    _print_context_progress,
    _references_generated_artifacts,
    run_serena_context,
)


def complete_report(**overrides):
    values = {
        "status": "sufficient",
        "architecture_summary": "Planning is coordinated by the planning graph.",
        "relevant_files": [],
        "relevant_symbols": [],
        "evidence": [],
        "question_resolutions": [],
        "missing_context": [],
        "research_checkpoints": [],
        "impact_chains": [],
        "architecture_decisions": [],
    }
    values.update(overrides)
    return values


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

    def test_identifies_generated_artifact_tool_arguments(self):
        self.assertTrue(_references_generated_artifacts({
            "relative_path": ".devflow/serena-context/b/context.json",
        }))
        self.assertFalse(_references_generated_artifacts({
            "relative_path": "src/devflow/cli.py",
        }))


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

    def test_removes_generated_artifact_results(self):
        result = SimpleNamespace(
            structuredContent={
                "files": [
                    "src/devflow/cli.py",
                    ".devflow/serena-context/b/context.json",
                ]
            },
            content=[],
        )

        text = _tool_result_text(result, 1000)

        self.assertIn("src/devflow/cli.py", text)
        self.assertNotIn(".devflow", text)


class SerenaContinuationTests(unittest.TestCase):
    def config(self):
        return SerenaContextConfig(
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
            model_request_min_interval_seconds=2.0,
            model=None,
            config_sources=(),
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

    def test_round_gate_displays_only_repository_gaps(self):
        report = {"missing_context": [
            {"kind": "repository", "description": "Find Schema X."},
            {"kind": "user_decision", "description": "Choose compatibility."},
        ]}
        with patch("builtins.print") as output:
            approved = _confirm_additional_context_round(
                report,
                auto_approve=True,
            )

        self.assertEqual(approved, "continue")
        rendered = "\n".join(call.args[0] for call in output.call_args_list)
        self.assertIn("Find Schema X", rendered)
        self.assertNotIn("Choose compatibility", rendered)

    def test_prints_existing_resolution_without_model_call(self):
        report = complete_report(question_resolutions=[{
            "question": "Where does data originate?",
            "resolution": "CombatService calculates it.",
            "source": "combat.ts:CombatService",
        }])
        with patch("builtins.print") as output:
            _print_context_progress(report)
        rendered = "\n".join(call.args[0] for call in output.call_args_list)
        self.assertIn("CombatService calculates it", rendered)


class SerenaModelRequestLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_invokes_model_without_waiting_for_first_request(self):
        class Model:
            async def ainvoke(self, value):
                return value

        limiter = _ModelRequestLimiter(0.0)

        with patch("builtins.print") as output:
            result = await limiter.invoke(Model(), "request", "Choose repository searches")

        self.assertEqual(result, "request")
        output.assert_called_once_with("LLM call 1: Choose repository searches")
        self.assertEqual(limiter.request_count, 1)
        self.assertEqual(limiter.wait_count, 0)

    async def test_records_wait_between_quick_requests(self):
        class Model:
            async def ainvoke(self, value):
                return value

        limiter = _ModelRequestLimiter(0.001)
        await limiter.invoke(Model(), "first", "Choose repository searches")
        await limiter.invoke(Model(), "second", "Review tool results and continue")

        self.assertEqual(limiter.request_count, 2)
        self.assertEqual(limiter.wait_count, 1)
        self.assertGreater(limiter.total_wait_seconds, 0)

    async def test_prints_model_error_before_raising(self):
        class Model:
            async def ainvoke(self, value):
                raise RuntimeError("invalid argument")

        limiter = _ModelRequestLimiter(0.0)
        with patch("builtins.print") as output:
            with self.assertRaisesRegex(RuntimeError, "invalid argument"):
                await limiter.invoke(Model(), "request", "Create grounded context report")

        self.assertIn("LLM ERROR", output.call_args_list[-1].args[0])
        self.assertIn("invalid argument", output.call_args_list[-1].args[0])


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
        report = SerenaContextReport(**complete_report(
            relevant_files=[{
                "path": "src/devflow/planning/graph.py",
                "role": "probable_change_target",
                "reason": "The requested behavior is implemented by this graph.",
                "symbols": [],
            }],
            evidence=[{
                "claim": "The graph invokes the planning node.",
                "source": "src/devflow/planning/graph.py:build_planning_graph",
            }],
            question_resolutions=[{
                "question": "Which graph owns planning?",
                "resolution": "The planning graph invokes the planning node.",
                "source": "src/devflow/planning/graph.py:build_planning_graph",
            }],
        ))

        self.assertEqual(report.evidence[0].claim, "The graph invokes the planning node.")
        self.assertEqual(
            report.relevant_files[0].role,
            "probable_change_target",
        )
        self.assertEqual(
            report.question_resolutions[0].question,
            "Which graph owns planning?",
        )

    def test_relevant_file_role_is_required(self):
        with self.assertRaises(ValueError):
            SerenaContextReport(**complete_report(
                relevant_files=[{
                    "path": "src/devflow/planning/graph.py",
                    "reason": "The graph is relevant.",
                    "symbols": [],
                }],
            ))

    def test_rejects_question_resolution_without_source(self):
        with self.assertRaises(ValueError):
            SerenaContextReport(**complete_report(
                question_resolutions=[{
                    "question": "Which graph owns planning?",
                    "resolution": "The planning graph owns it.",
                }],
            ))

    def test_rejects_unknown_relevant_file_role(self):
        with self.assertRaises(ValueError):
            SerenaContextReport(**complete_report(
                relevant_files=[{
                    "path": "src/devflow/planning/graph.py",
                    "role": "maybe",
                    "reason": "The graph might be relevant.",
                    "symbols": [],
                }],
            ))

    def test_schema_is_portable_and_fully_required(self):
        schema = SerenaContextReport.model_json_schema()

        def check_objects(value):
            if isinstance(value, dict):
                if value.get("type") == "object":
                    self.assertFalse(value.get("additionalProperties", True))
                    self.assertEqual(set(value.get("required", [])), set(value["properties"]))
                self.assertNotIn("maxLength", value)
                self.assertNotIn("maxItems", value)
                for nested in value.values():
                    check_objects(nested)
            elif isinstance(value, list):
                for nested in value:
                    check_objects(nested)

        check_objects(schema)
        self.assertEqual(SERENA_SCHEMA_VERSION, "portable-v1")
        self.assertEqual(SERENA_STRUCTURED_OUTPUT_METHOD, "function_calling")


class SerenaFailureArtifactTests(unittest.TestCase):
    def test_writes_diagnostic_and_raises_hard_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(output_dir=temp_dir)
            with patch(
                "devflow.repository_context.serena._run_serena",
                side_effect=RuntimeError("provider rejected arguments"),
            ):
                with self.assertRaises(SerenaContextRunError) as raised:
                    run_serena_context("Investigate failure", config)

            path = Path(raised.exception.diagnostic_path)
            diagnostic = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostic["exception_type"], "RuntimeError")
            self.assertIn("provider rejected arguments", diagnostic["exception_message"])
