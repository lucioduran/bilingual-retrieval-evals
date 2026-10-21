import math
import unittest
from dataclasses import replace

from bilingual_retrieval.contracts import parse_dataset
from bilingual_retrieval.metrics import aggregate, breakdown, family_interval, score
from bilingual_retrieval.retrieval import bm25, cosine, dense, tokens
from tests.test_contracts import fixture


class MetricsTests(unittest.TestCase):
    def test_hand_calculated(self):
        grades = {"a": 2, "b": 1, "c": 0}
        got = score(["c", "b", "a"], grades, 2)
        self.assertEqual(got["recall"], 0.5)
        self.assertEqual(got["reciprocal_rank"], 0.5)
        self.assertAlmostEqual(got["ndcg"], (1 / math.log2(3)) / (3 + 1 / math.log2(3)))
        self.assertEqual(score(["a", "b", "c"], grades, 9)["ndcg"], 1.0)
        self.assertEqual(score([], grades, 3)["recall"], 0)
        self.assertEqual(score(["c"], grades, 1)["reciprocal_rank"], 0)

    def test_unanswerable_denominator(self):
        row = score(["a"], {"a": 0}, 1)
        self.assertEqual(row, dict(recall=None, ndcg=None, reciprocal_rank=None))
        summary = aggregate([row, score(["a"], {"a": 2}, 1)])
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["recall"], {"n": 1, "mean": 1})
        self.assertEqual(aggregate([])["ndcg"], {"n": 0, "mean": None})

    def test_invalid_rankings(self):
        for ranking, grades, k in [
            (["a", "a"], {"a": 1}, 1),
            (["b"], {"a": 1}, 1),
            (["a"], {"a": 3}, 1),
            (["a"], {"a": True}, 1),
            (["a"], {"a": 1}, 0),
            (["a"], {"a": 1}, True),
        ]:
            with self.subTest(ranking=ranking, k=k), self.assertRaises(ValueError):
                score(ranking, grades, k)

    def test_breakdowns(self):
        q = parse_dataset(fixture()).queries[0]
        q2 = replace(q, id="q2", language="es", accent="omitted", entity="present", kind="cultural")
        result = breakdown([q, q2], {"q": score(["a"], {"a": 2}, 1), "q2": score([], {"a": 2}, 1)})
        self.assertEqual(result["overall"]["recall"], {"n": 2, "mean": 0.5})
        for group in ("direction/es->en", "accent/omitted", "entity/present", "kind/cultural"):
            self.assertEqual(result[group]["recall"], {"n": 1, "mean": 0})

    def test_family_weighting_and_reproducibility(self):
        values = {"a": [1] * 6, "b": [0]}
        a = family_interval(values)
        self.assertEqual(a["mean"], 0.5)
        self.assertEqual(a, family_interval(dict(reversed(list(values.items())))))
        self.assertEqual(family_interval({"a": [0.2]}, resamples=1)["low"], 0.2)
        for invalid in ({}, {"a": []}, {"a": [float("nan")]}, {"a": [float("inf")]}):
            with self.assertRaises(ValueError):
                family_interval(invalid)
        with self.assertRaises(ValueError):
            family_interval(values, resamples=0)


class RetrievalTests(unittest.TestCase):
    def test_tokens_and_accents(self):
        self.assertEqual(
            tokens("CAF\u00c9 Cafe\u0301 foo_bar 12"),
            ["caf\u00e9", "caf\u00e9", "foo", "bar", "12"],
        )
        self.assertEqual(tokens("caf\u00e9", True), ["cafe"])
        self.assertEqual(tokens("..."), [])

    def test_bm25_length_and_ties(self):
        self.assertEqual(bm25({"a": "water other other", "b": "water"}, "water")[0], "b")
        self.assertEqual(bm25({"z": "water", "a": "water"}, "unknown"), ["a", "z"])
        self.assertEqual(bm25({"z": "...", "a": ""}, "water"), ["a", "z"])
        self.assertEqual(bm25({"a": "unrelated", "b": "caf\u00e9"}, "cafe", True)[0], "b")
        self.assertEqual(bm25({"a": "water", "b": "cold"}, "water water"), ["a", "b"])
        with self.assertRaises(ValueError):
            bm25({}, "water")

    def test_cosine(self):
        self.assertAlmostEqual(cosine([3, 4], [3, 4]), 1)
        self.assertEqual(cosine([1, 0], [0, 1]), 0)
        self.assertEqual(cosine([1, 0], [-1, 0]), -1)
        self.assertAlmostEqual(cosine([1e300, 1e300], [1e300, 1e300]), 1)
        for left, right in [
            ([], []),
            ([1], [1, 2]),
            ([0], [1]),
            ([1], [0]),
            ([float("inf")], [1]),
            ([float("nan")], [1]),
            ([True], [1]),
        ]:
            with self.assertRaises(ValueError):
                cosine(left, right)
        self.assertEqual(dense({"z": [1, 0], "a": [1, 0], "b": [-1, 0]}, [1, 0]), ["a", "z", "b"])
        with self.assertRaises(ValueError):
            dense({}, [1])
