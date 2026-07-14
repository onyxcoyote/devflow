import unittest

from devflow.planning.research import (
    question_key,
    repository_context_questions,
    supplemental_context_request,
)


class PlanningFlowTests(unittest.TestCase):
    def test_extracts_repository_questions_only_when_context_is_needed(self):
        question = {
            "kind": "repository_context",
            "question": "Where is the provider constructed?",
            "impact": "The responsible file is unknown.",
            "suggested_action": "Trace model construction.",
        }
        plan = {
            "status": "needs_repository_context",
            "outstanding_items": [
                question,
                {**question, "kind": "user_decision"},
            ],
        }

        self.assertEqual(repository_context_questions(plan), [question])
        self.assertEqual(
            repository_context_questions({**plan, "status": "ready"}),
            [],
        )

    def test_builds_targeted_supplemental_request(self):
        result = supplemental_context_request(
            "Improve planning.",
            [{
                "question": "Where is the provider constructed?",
                "impact": "The responsible file is unknown.",
                "suggested_action": "Trace model construction.",
            }],
            1,
        )

        self.assertIn("ORIGINAL DEVELOPMENT REQUEST\nImprove planning.", result)
        self.assertIn("Where is the provider constructed?", result)
        self.assertIn("Trace model construction.", result)

    def test_question_keys_ignore_case_spacing_and_terminal_punctuation(self):
        self.assertEqual(
            question_key(" Where  is the provider? "),
            question_key("where is the provider."),
        )


if __name__ == "__main__":
    unittest.main()
