import unittest
from copy import deepcopy

from evaluate import ROOT, check, grade, read


class GraderTests(unittest.TestCase):
    def setUp(self):
        self.key = read(ROOT / "data/key.json")

    def test_answer_key_and_balance(self):
        report = grade({"outputs": self.key})
        self.assertEqual(report["strict_pass"], 24)
        self.assertEqual(list(report["by_category"].values()), [[8, 8]] * 3)
        cases = read(ROOT / "data/cases.json")
        self.assertEqual(len(cases), len(self.key))
        self.assertEqual({c["id"] for c in cases}, set(self.key))
        for case in cases:
            self.assertTrue(set(self.key[case["id"]]["sources"]) <= case["sources"].keys())

    def test_deliberate_errors(self):
        gold = dict(status="answered", value="4", sources=["S1"])
        for change, reason in [({"sources": ["S2"]}, "citations"), ({"value": 7}, "value"),
                               ({"status": "conflicting_evidence"}, "decision")]:
            self.assertEqual(check(gold | change, gold), [reason])
        self.assertEqual(check(gold | {"value": 4}, gold), [])
        conflict = dict(status="conflicting_evidence", value=None, sources=["S1", "S2"])
        self.assertEqual(check(conflict | {"sources": ["S1"]}, conflict), ["citations"])
        self.assertTrue(check(None, conflict))
        self.assertTrue(check({"status": conflict["status"], "sources": conflict["sources"]}, conflict))

    def test_incomplete_run(self):
        answers = deepcopy(self.key)
        answers.pop(next(iter(answers)))
        with self.assertRaises(ValueError):
            grade({"outputs": answers})


class PublishedRunsTests(unittest.TestCase):
    def test_saved_language_model_outputs_and_metadata(self):
        from run_model import PROMPTS, SETTINGS, parse_answer

        expected_scores = {"basic": 6, "evidence": 4}
        runs = {condition: read(ROOT / "runs" / f"qwen-{condition}.json")
                for condition in expected_scores}
        for condition, run in runs.items():
            with self.subTest(condition=condition):
                self.assertEqual(run["prompt"], PROMPTS[condition])
                self.assertEqual(run["settings"], SETTINGS)
                self.assertEqual(run["outputs"], {
                    case_id: parse_answer(raw) for case_id, raw in run["raw"].items()
                })
                self.assertEqual(grade(run)["strict_pass"], expected_scores[condition])
        for field in ("model", "model_digest", "runtime_version", "dataset_sha256"):
            self.assertTrue(runs["basic"][field])
            self.assertEqual(runs["basic"][field], runs["evidence"][field])

    def test_numeric_normalization_and_exact_citation_set(self):
        gold = dict(status="answered", value="4", sources=["S2"])
        self.assertEqual(check(gold | {"value": 4.0}, gold), [])
        self.assertEqual(check(gold | {"value": "4 crore"}, gold), ["value"])
        self.assertEqual(check(gold | {"sources": ["S2", "S2"]}, gold), ["citations"])

    def test_model_prompt_contains_only_public_case_fields(self):
        import json

        from run_model import PROMPTS, messages_for

        case = read(ROOT / "data/cases.json")[0]
        system, user = messages_for(PROMPTS["basic"], case)
        self.assertEqual(system["role"], "system")
        self.assertEqual(user["role"], "user")
        self.assertEqual(json.loads(user["content"]), {
            field: case[field] for field in ("company", "question", "sources")
        })
