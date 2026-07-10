import unittest

from devflow.code_review.nodes import assess_review


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


if __name__ == "__main__":
    unittest.main()
