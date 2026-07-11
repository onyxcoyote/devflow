import unittest
from types import SimpleNamespace

from devflow.code_review.nodes import _model_result_metadata, assess_review


def review(**overrides):
    value = {
        "verdict": "approve",
        "confidence": "high",
        "summary": "No defects found.",
        "findings": [],
        "check_assessments": [],
        "uncertainties": [],
    }
    value.update(overrides)
    return value


def command_result(*, passed):
    return {"passed": passed}


def assess(review_value, command_results=()):
    return assess_review({
        "review": review_value,
        "command_results": list(command_results),
    })["assessment"]


class AssessReviewTests(unittest.TestCase):
    def test_unrelated_command_failure_does_not_fail_review(self):
        result = assess(
            review(check_assessments=[{
                "command_index": 0,
                "status": "unrelated_failure",
                "reasoning": "The failure is in an unchanged workspace.",
            }]),
            [command_result(passed=False)],
        )

        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["score"], 100)

    def test_change_related_command_failure_fails_review(self):
        result = assess(
            review(check_assessments=[{
                "command_index": 0,
                "status": "change_failure",
                "reasoning": "A changed test fails.",
            }]),
            [command_result(passed=False)],
        )

        self.assertEqual(result["verdict"], "fail")
        self.assertEqual(result["score"], 70)

    def test_environment_failure_is_inconclusive(self):
        result = assess(
            review(check_assessments=[{
                "command_index": 0,
                "status": "environment_failure",
                "reasoning": "The test executable is unavailable.",
            }]),
            [command_result(passed=False)],
        )

        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(result["confidence"], "low")

    def test_missing_command_assessment_is_inconclusive(self):
        result = assess(review(), [command_result(passed=True)])

        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(result["confidence"], "low")
        self.assertTrue(result["uncertainties"])

    def test_review_confidence_is_not_inferred_from_empty_findings(self):
        result = assess(review(confidence="medium"))

        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["confidence"], "medium")


class ModelResultMetadataTests(unittest.TestCase):
    def test_extracts_langchain_usage_metadata(self):
        raw_response = SimpleNamespace(
            id="response-123",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        )

        result = _model_result_metadata(raw_response, None, 1.23456)

        self.assertEqual(result["response_id"], "response-123")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 20)
        self.assertEqual(result["total_tokens"], 120)
        self.assertEqual(result["elapsed_seconds"], 1.235)
        self.assertTrue(result["parsing_succeeded"])

    def test_falls_back_to_openai_token_usage(self):
        raw_response = SimpleNamespace(
            id=None,
            response_metadata={
                "done_reason": "length",
                "token_usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                    "total_tokens": 60,
                },
            },
            usage_metadata=None,
        )

        result = _model_result_metadata(
            raw_response,
            ValueError("invalid response"),
            2,
        )

        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["input_tokens"], 50)
        self.assertEqual(result["output_tokens"], 10)
        self.assertEqual(result["total_tokens"], 60)
        self.assertFalse(result["parsing_succeeded"])
        self.assertEqual(result["parsing_error"], "invalid response")


if __name__ == "__main__":
    unittest.main()
