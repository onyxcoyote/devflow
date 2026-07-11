import unittest
from types import SimpleNamespace

from devflow.planning.nodes import (
    create_plan_report,
    make_context_request_node,
    make_plan_node,
)
from devflow.planning.schemas import DevelopmentPlan, PlanningContextRequest


class FakeStructuredModel:
    def __init__(self, result):
        self.result = result

    def invoke(self, prompt):
        return self.result


class FakeModel:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema, include_raw=False):
        self.schema = schema
        self.include_raw = include_raw
        return FakeStructuredModel(self.result)


class CreatePlanReportTests(unittest.TestCase):
    def test_renders_structured_plan(self):
        state = {
            "model_info": {"model": "test-model", "provider": "ollama"},
            "plan": {
                "status": "ready",
                "objective": "Add a planning command.",
                "understanding": "Create plans without editing files.",
                "proposed_changes": [{
                    "area": "CLI",
                    "description": "Add the plan command.",
                    "likely_files": ["src/devflow/cli.py"],
                    "reason": "Expose planning to users.",
                }],
                "acceptance_criteria": ["The command produces plan.md."],
                "verification": ["Run unit tests."],
                "assumptions": [],
                "uncertainties": [],
                "risks": [],
            },
        }

        report = create_plan_report(state)["report"]

        self.assertIn("**Status:** `ready`", report)
        self.assertIn("src/devflow/cli.py", report)
        self.assertIn("The command produces plan.md.", report)


class CreatePlanTests(unittest.TestCase):
    def test_returns_structured_plan_and_model_metadata(self):
        parsed_plan = DevelopmentPlan(
            status="ready",
            objective="Add planning.",
            understanding="Add a read-only command.",
            acceptance_criteria=["A plan is saved."],
            verification=["Run tests."],
        )
        raw_response = SimpleNamespace(
            id="response-1",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"total_tokens": 20},
            content="",
            additional_kwargs={},
        )
        node = make_plan_node(FakeModel({
            "raw": raw_response,
            "parsed": parsed_plan,
            "parsing_error": None,
        }))

        result = node({
            "request": "Add planning.",
            "context_text": "{}",
            "save_model_exchange": False,
            "model_result": {},
            "model_exchange": {},
        })

        self.assertEqual(result["plan"]["status"], "ready")
        self.assertEqual(result["model_result"]["plan"]["finish_reason"], "stop")
        self.assertEqual(result["model_exchange"], {})

    def test_truncated_response_returns_needs_context(self):
        raw_response = SimpleNamespace(
            id="response-1",
            response_metadata={"finish_reason": "length"},
            usage_metadata={},
            content="",
            additional_kwargs={},
        )
        node = make_plan_node(FakeModel({
            "raw": raw_response,
            "parsed": None,
            "parsing_error": ValueError("incomplete"),
        }))

        result = node({
            "request": "Add planning.",
            "context_text": "{}",
            "save_model_exchange": True,
            "model_result": {},
            "model_exchange": {},
        })

        self.assertEqual(result["plan"]["status"], "needs_context")
        self.assertIn("output-token limit", result["plan"]["uncertainties"][0])
        self.assertIn("prompt", result["model_exchange"]["plan"]["request"])


class ContextRequestTests(unittest.TestCase):
    def test_returns_bounded_context_request_and_metadata(self):
        parsed_request = PlanningContextRequest(
            files=["src/devflow/cli.py"],
            searches=["planning_flow"],
            reason="Find the CLI and planning entry point.",
        )
        raw_response = SimpleNamespace(
            id="response-1",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"total_tokens": 12},
            content="",
            additional_kwargs={},
        )
        node = make_context_request_node(FakeModel({
            "raw": raw_response,
            "parsed": parsed_request,
            "parsing_error": None,
        }))

        result = node({
            "request": "Improve planning context.",
            "repository_context": {"directory_summary": "src/"},
            "max_requested_files": 8,
            "max_searches": 6,
            "save_model_exchange": True,
            "model_result": {},
            "model_exchange": {},
        })

        self.assertEqual(result["context_request"]["files"], ["src/devflow/cli.py"])
        self.assertEqual(
            result["model_result"]["context_request"]["finish_reason"],
            "stop",
        )
        self.assertIn("context_request", result["model_exchange"])
