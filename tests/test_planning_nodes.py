import unittest
from types import SimpleNamespace

from devflow.planning.nodes import create_plan_report, make_plan_node
from devflow.planning.schemas import DevelopmentPlan


class FakeLogger:
    def info(self, *args):
        pass

    def warning(self, *args):
        pass

    def error(self, *args):
        pass


class FakeStructuredModel:
    def __init__(self, result):
        self.result = result

    def invoke(self, prompt):
        self.prompt = prompt
        return self.result


class FakeModel:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema, include_raw=False):
        self.schema = schema
        return FakeStructuredModel(self.result)


def state(previous_plan=None):
    return {
        "request": "Add planning.",
        "context_text": "{}",
        "context_source": {"previous_plan_path": "/plans/old.json"},
        "previous_plan": previous_plan,
        "save_model_exchange": False,
        "model_result": {},
        "model_exchange": {},
    }


class CreatePlanReportTests(unittest.TestCase):
    def test_renders_structured_plan(self):
        plan = DevelopmentPlan(
            status="ready",
            objective="Add a planning command.",
            design_summary="Create plans without editing files.",
            proposed_changes=[{
                "path": "src/devflow/cli.py",
                "symbols": ["_run_plan"],
                "change": "Add the plan command.",
                "reason": "Expose planning to users.",
                "evidence": ["src/devflow/cli.py:_run_plan"],
            }],
            acceptance_criteria=["The command produces plan.md."],
            verification=["Run unit tests."],
        ).model_dump()
        report = create_plan_report({
            "model_info": {"model": "test-model", "provider": "ollama"},
            "plan": plan,
        })["report"]

        self.assertIn("**Status:** `ready`", report)
        self.assertIn("src/devflow/cli.py", report)
        self.assertIn("The command produces plan.md.", report)


class CreatePlanTests(unittest.TestCase):
    def raw_response(self, finish_reason="stop"):
        return SimpleNamespace(
            id="response-1",
            response_metadata={"finish_reason": finish_reason},
            usage_metadata={"total_tokens": 20},
            content="",
            additional_kwargs={},
        )

    def test_returns_structured_plan_and_model_metadata(self):
        parsed_plan = DevelopmentPlan(
            status="ready",
            objective="Add planning.",
            design_summary="Add a read-only command.",
            acceptance_criteria=["A plan is saved."],
            verification=["Run tests."],
        )
        model = FakeModel({
            "raw": self.raw_response(),
            "parsed": parsed_plan,
            "parsing_error": None,
        })
        node = make_plan_node(model, model, FakeLogger())
        result = node(state())

        self.assertEqual(result["plan"]["status"], "ready")
        self.assertEqual(result["model_result"]["plan"]["finish_reason"], "stop")

    def test_refinement_records_previous_plan(self):
        parsed_plan = DevelopmentPlan(
            status="ready",
            objective="Improve planning.",
            design_summary="Revise the prior plan.",
            revision={"changes": ["Added validation."]},
        )
        model = FakeModel({
            "raw": self.raw_response(),
            "parsed": parsed_plan,
            "parsing_error": None,
        })
        node = make_plan_node(model, model, FakeLogger())
        result = node(state(previous_plan={"status": "ready"}))

        self.assertEqual(result["plan"]["revision"]["based_on"], "/plans/old.json")
        self.assertEqual(result["plan"]["revision"]["changes"], ["Added validation."])

    def test_two_truncated_responses_return_blocked_status(self):
        model = FakeModel({
            "raw": self.raw_response("length"),
            "parsed": None,
            "parsing_error": ValueError("incomplete"),
        })
        node = make_plan_node(model, model, FakeLogger())
        result = node(state())

        self.assertEqual(result["plan"]["status"], "blocked")
        self.assertIn(
            "output limit reached",
            result["plan"]["outstanding_items"][0]["question"],
        )
        self.assertEqual(len(result["model_result"]["plan_attempts"]), 2)
