import unittest
import tempfile
from pathlib import Path

from devflow.planning.research import (
    apply_user_answers_to_context,
    context_user_questions,
    impact_context_request,
    impact_delta_from_report,
    merge_impact_delta,
    normalize_supplemental_report,
    merge_context_refinement,
    read_context_approved_files,
    question_key,
    repository_context_questions,
    supplemental_prior_report,
    supplemental_context_request,
    supplemental_progress_signature,
    user_decision_questions,
)


class PlanningFlowTests(unittest.TestCase):
    def test_impact_delta_cannot_rewrite_established_resolutions(self):
        report = {
            "status": "sufficient",
            "architecture_summary": "Incorrect replacement summary.",
            "question_resolutions": [{
                "question": "Where is food stored?", "resolution": "Wrong answer", "source": "x.ts"
            }],
            "impact_chains": [],
            "architecture_decisions": [],
            "relevant_files": [],
            "evidence": [],
            "missing_context": [],
        }

        delta = impact_delta_from_report(report)

        self.assertNotIn("question_resolutions", delta)
        self.assertNotIn("architecture_summary", delta)

    def test_impact_delta_requires_qualified_semantic_identity(self):
        report = {
            "status": "sufficient",
            "impact_chains": [{
                "concept": "food",
                "owner_type": "Player",
                "semantic_meaning": "Lifetime food consumed",
                "source_of_truth": "",
                "lifecycle": "lifetime cumulative",
                "stages": [{
                    "stage": "mapping", "path": "server/player.ts", "symbol": "mapPlayer",
                    "object_type": "", "relationship": "maps food", "status": "resolved",
                }],
                "affected_definitions": ["server/player.ts:mapPlayer"],
                "callers": [], "consumers": [], "persistence_effects": [],
                "closure_gaps": [], "potential_side_effects": [],
            }],
            "architecture_decisions": [], "relevant_files": [], "evidence": [],
            "missing_context": [],
        }

        delta = impact_delta_from_report(report)

        self.assertEqual(delta["impact_chains"][0]["stages"][0]["status"], "unresolved")
        self.assertTrue(delta["missing_context"])

    def test_impact_merge_preserves_facts_and_distinguishes_same_named_values(self):
        context = {
            "status": "sufficient",
            "architecture_summary": "Established summary.",
            "question_resolutions": [{"question": "Meaning?", "resolution": "Established."}],
            "missing_context": [],
            "impact_chains": [],
        }
        base = {
            "concept": "food", "source_of_truth": "db", "lifecycle": "current",
            "stages": [], "affected_definitions": [], "callers": [], "consumers": [],
            "persistence_effects": [], "closure_gaps": [], "potential_side_effects": [],
        }
        delta = {
            "impact_chains": [
                {**base, "owner_type": "Fridge", "semantic_meaning": "Current inventory"},
                {**base, "owner_type": "Player", "semantic_meaning": "Lifetime consumed"},
            ],
            "architecture_decisions": [], "relevant_files": [], "evidence": [],
            "missing_context": [],
        }

        merge_impact_delta(context, delta)

        self.assertEqual(context["architecture_summary"], "Established summary.")
        self.assertEqual(context["question_resolutions"][0]["resolution"], "Established.")
        self.assertEqual(len(context["impact_chains"]), 2)

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

    def test_extracts_user_decisions_for_console_input(self):
        item = {"kind": "user_decision", "question": "Preserve compatibility?"}
        plan = {"status": "needs_user_decision", "outstanding_items": [item]}
        self.assertEqual(user_decision_questions(plan), [item])

    def test_context_user_answer_resolves_before_planning(self):
        report = {
            "status": "needs_user_decision",
            "missing_context": [{
                "kind": "user_decision",
                "description": "Preserve compatibility?",
                "suggested_action": "Choose compatibility behavior.",
            }],
            "question_resolutions": [],
        }
        self.assertEqual(
            context_user_questions(report)[0]["question"],
            "Preserve compatibility?",
        )
        apply_user_answers_to_context(report, [{
            "question": "Preserve compatibility?",
            "answer": "Yes, preserve it.",
        }])
        self.assertEqual(report["status"], "sufficient")
        self.assertEqual(report["missing_context"], [])
        self.assertEqual(report["question_resolutions"][0]["source"], "user input")

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
        self.assertIn("question_resolutions", result)
        self.assertIn("do not merely state where it is defined", result)

    def test_prior_report_drops_inherited_questions(self):
        prior = supplemental_prior_report({
            "status": "needs_user_decision",
            "relevant_files": [{"path": "src/schema.py"}],
            "missing_context": [{"description": "Old unrelated question"}],
            "question_resolutions": [{"question": "Old question"}],
            "supplemental_rounds": [{"round": 1}],
        })

        self.assertEqual(prior["relevant_files"], [{"path": "src/schema.py"}])
        self.assertEqual(prior["missing_context"], [])
        self.assertEqual(prior["question_resolutions"], [])
        self.assertNotIn("supplemental_rounds", prior)

    def test_normalizes_report_to_active_questions(self):
        questions = [{
            "question": "What fields are in Schema X?",
            "suggested_action": "Read src/schema.py.",
        }]
        normalized = normalize_supplemental_report({
            "status": "needs_user_decision",
            "question_resolutions": [],
            "missing_context": [{"description": "Old unrelated question"}],
            "research_checkpoints": [],
        }, questions)

        self.assertEqual(normalized["status"], "needs_repository_context")
        self.assertEqual(
            normalized["missing_context"][0]["description"],
            "What fields are in Schema X?",
        )
        self.assertNotIn("Old unrelated question", str(normalized))

    def test_preserves_partial_subquestion_progress(self):
        question = "How does X work, and if Y occurs, where is Z handled?"
        normalized = normalize_supplemental_report({
            "question_resolutions": [],
            "research_checkpoints": [{
                "original_question": question,
                "subquestion": "Where is Z handled when Y occurs?",
                "status": "unresolved",
                "answer": "",
                "partial_findings": "Y is detected in service.py.",
                "sources_inspected": ["service.py:detect_y"],
                "next_investigation": "Trace detect_y callers.",
            }],
        }, [{"question": question, "suggested_action": "Trace the flow."}])

        self.assertEqual(normalized["missing_context"][0]["description"], "Where is Z handled when Y occurs?")
        self.assertIn("Y is detected", str(normalized["research_checkpoints"]))
        self.assertTrue(supplemental_progress_signature(normalized))

    def test_context_refinement_replaces_unresolved_set(self):
        context = {
            "status": "needs_repository_context",
            "missing_context": [{"kind": "repository", "description": "Old gap"}],
            "evidence": [],
            "research_checkpoints": [],
        }
        merge_context_refinement(context, {
            "status": "sufficient",
            "missing_context": [],
            "evidence": [{"claim": "Data originates in combat.", "source": "combat.ts"}],
            "research_checkpoints": [],
        })
        self.assertEqual(context["status"], "sufficient")
        self.assertEqual(context["missing_context"], [])
        self.assertEqual(context["evidence"][0]["source"], "combat.ts")

    def test_context_refinement_upgrades_file_without_duplicate(self):
        context = {
            "status": "needs_repository_context",
            "missing_context": [],
            "relevant_files": [{
                "path": "history.ts", "role": "supporting_context",
                "reason": "Reads history.", "symbols": ["readHistory"],
            }],
        }
        merge_context_refinement(context, {
            "status": "sufficient",
            "missing_context": [],
            "relevant_files": [{
                "path": "history.ts", "role": "probable_change_target",
                "reason": "Writes the new field.", "symbols": ["saveHistory"],
            }],
        })

        self.assertEqual(len(context["relevant_files"]), 1)
        self.assertEqual(context["relevant_files"][0]["role"], "probable_change_target")
        self.assertEqual(
            context["relevant_files"][0]["symbols"], ["readHistory", "saveHistory"]
        )

    def test_impact_request_requires_end_to_end_closure(self):
        request, questions = impact_context_request("Add history statistics.")
        self.assertIn("IMPACT CLOSURE", request)
        self.assertIn("concrete object types", questions[0]["question"])
        self.assertIn("persistence", questions[0]["question"])

    def test_question_keys_ignore_case_spacing_and_terminal_punctuation(self):
        self.assertEqual(
            question_key(" Where  is the provider? "),
            question_key("where is the provider."),
        )

    def test_reads_only_context_approved_repository_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "schema.py").write_text("class Schema:\n    field: str\n", encoding="utf-8")
            outside = root.parent / "not-approved.txt"
            context = {
                "relevant_files": [{
                    "path": "schema.py",
                    "role": "probable_change_target",
                }],
                "question_resolutions": [{
                    "source": "../not-approved.txt",
                }],
            }

            excerpts = read_context_approved_files(str(root), context)

            self.assertIn("class Schema", excerpts["schema.py"])
            self.assertNotIn("../not-approved.txt", excerpts)


if __name__ == "__main__":
    unittest.main()
