import unittest
from types import SimpleNamespace

from devflow.planning.nodes import create_plan_report, make_plan_node
from devflow.planning.schemas import (
    DevelopmentPlan,
    PLAN_SCHEMA_VERSION,
    PLAN_STRUCTURED_OUTPUT_METHOD,
)


class FakeLogger:
    def info(self, *args):
        pass

    def warning(self, *args):
        pass

    def error(self, *args):
        pass


class FakeCompletion:
    def model_dump(self, mode=None):
        return {"choices": [{"message": {"content": "partial output"}}]}


class FakeLengthError(Exception):
    def __init__(self):
        super().__init__("length limit reached")
        self.completion = FakeCompletion()


class FakeModel:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        self.schema = schema
        self.structured_output_options = kwargs
        return FakeStructuredModel(self.result)


class FailingModel(FakeModel):
    max_tokens = 8000

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        return FakeStructuredModel(FakeLengthError())


class FakeStructuredModel:
    def __init__(self, result):
        self.result = result

    def invoke(self, prompt):
        self.prompt = prompt
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


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
                "dependencies": [],
            }],
            assumptions=[],
            outstanding_items=[],
            decisions=[],
            acceptance_criteria=["The command produces plan.md."],
            verification=["Run unit tests."],
            risks=[],
            revision={"based_on": None, "changes": []},
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

    def proposed_change(self):
        return {
            "path": "src/devflow/planning/nodes.py",
            "symbols": ["make_plan_node"],
            "change": "Improve planning behavior.",
            "reason": "Produce an actionable implementation plan.",
            "evidence": ["src/devflow/planning/nodes.py:make_plan_node"],
            "dependencies": [],
        }

    def test_returns_structured_plan_and_model_metadata(self):
        parsed_plan = DevelopmentPlan(
            status="ready",
            objective="Add planning.",
            design_summary="Add a read-only command.",
            assumptions=[],
            proposed_changes=[self.proposed_change()],
            outstanding_items=[],
            decisions=[],
            acceptance_criteria=["A plan is saved."],
            verification=["Run tests."],
            risks=[],
            revision={"based_on": None, "changes": []},
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
        self.assertEqual(
            model.structured_output_options["method"],
            PLAN_STRUCTURED_OUTPUT_METHOD,
        )
        self.assertEqual(
            result["model_result"]["plan"]["schema_version"],
            PLAN_SCHEMA_VERSION,
        )

    def test_refinement_records_previous_plan(self):
        parsed_plan = DevelopmentPlan(
            status="ready",
            objective="Improve planning.",
            design_summary="Revise the prior plan.",
            assumptions=[],
            proposed_changes=[self.proposed_change()],
            outstanding_items=[],
            decisions=[],
            acceptance_criteria=[],
            verification=[],
            risks=[],
            revision={"based_on": None, "changes": ["Added validation."]},
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

    def test_empty_actionable_plan_retries_with_concrete_changes(self):
        empty_plan = DevelopmentPlan(
            status="needs_user_decision",
            objective="Improve planning.",
            design_summary="Choose an approach after clarification.",
            assumptions=[],
            proposed_changes=[],
            outstanding_items=[],
            decisions=[],
            acceptance_criteria=[],
            verification=[],
            risks=[],
            revision={"based_on": None, "changes": []},
        )
        actionable_plan = DevelopmentPlan.model_validate({
            **empty_plan.model_dump(),
            "proposed_changes": [self.proposed_change()],
        })
        first_model = FakeModel({
            "raw": self.raw_response(),
            "parsed": empty_plan,
            "parsing_error": None,
        })
        retry_model = FakeModel({
            "raw": self.raw_response(),
            "parsed": actionable_plan,
            "parsing_error": None,
        })

        result = make_plan_node(first_model, retry_model, FakeLogger())(state())

        self.assertEqual(len(result["plan"]["proposed_changes"]), 1)
        self.assertIn(
            "requires concrete proposed_changes",
            result["model_result"]["plan_attempts"][0]["quality_issue"],
        )
        self.assertIsNone(result["model_result"]["plan_attempts"][1]["quality_issue"])

    def test_blocked_plan_can_omit_proposed_changes(self):
        blocked_plan = DevelopmentPlan(
            status="needs_repository_context",
            objective="Improve planning.",
            design_summary="A required repository fact is missing.",
            assumptions=[],
            proposed_changes=[],
            outstanding_items=[{
                "kind": "repository_context",
                "question": "Which module owns this behavior?",
                "impact": "Responsible file changes cannot be identified.",
                "suggested_action": "Collect the missing module context.",
            }],
            decisions=[],
            acceptance_criteria=[],
            verification=[],
            risks=[],
            revision={"based_on": None, "changes": []},
        )
        model = FakeModel({
            "raw": self.raw_response(),
            "parsed": blocked_plan,
            "parsing_error": None,
        })

        result = make_plan_node(model, model, FakeLogger())(state())

        self.assertEqual(result["plan"]["status"], "needs_repository_context")
        self.assertEqual(len(result["model_result"]["plan_attempts"]), 1)

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

    def test_plan_schema_uses_portable_required_object_subset(self):
        schema = DevelopmentPlan.model_json_schema()

        def inspect(value):
            if isinstance(value, dict):
                if value.get("type") == "object":
                    self.assertFalse(value.get("additionalProperties", True))
                    self.assertEqual(
                        set(value.get("required", [])),
                        set(value.get("properties", {})),
                    )
                self.assertNotIn("maxLength", value)
                self.assertNotIn("maxItems", value)
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)

        inspect(schema)

    def test_exception_exchange_captures_request_limit_and_partial_completion(self):
        current_state = state()
        current_state["save_model_exchange"] = True
        model = FailingModel(None)
        node = make_plan_node(model, model, FakeLogger())

        result = node(current_state)

        attempt = result["model_exchange"]["plan_attempts"][0]
        self.assertEqual(attempt["configured_output_limit"], 8000)
        self.assertIn("DEVELOPMENT REQUEST", attempt["request"]["prompt"])
        self.assertEqual(
            attempt["error"]["partial_completion"]["choices"][0]["message"]["content"],
            "partial output",
        )
