import copy
import json
import tempfile
import unittest
from pathlib import Path

from bilingual_retrieval.contracts import load_dataset, parse_dataset, read_json

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return {
        "version": 1,
        "name": "test",
        "documents": [
            {"id": "a", "language": "en", "text": "Hot water is available.", "source": "original"},
            {"id": "b", "language": "en", "text": "Cold water only.", "source": "original"},
        ],
        "queries": [
            {
                "id": "q",
                "family": "water",
                "split": "dev",
                "language": "en",
                "target_language": "en",
                "text": "hot water",
                "kind": "parallel",
                "entity": "absent",
                "accent": "none",
                "judgments": [
                    {
                        "document_id": "a",
                        "grade": 2,
                        "evidence": "Hot water",
                        "rationale": "Answers intent",
                        "assessor": "test author",
                        "review": "authored",
                    },
                    {
                        "document_id": "b",
                        "grade": 0,
                        "evidence": "",
                        "rationale": "Wrong temperature",
                        "assessor": "test author",
                        "review": "pending",
                    },
                ],
            }
        ],
    }


class ContractsTests(unittest.TestCase):
    def test_real_dataset(self):
        data = load_dataset(ROOT / "data/dataset.json")
        self.assertEqual(len(data.documents), 32)
        self.assertEqual(len(data.queries), 68)
        self.assertEqual(
            {q.language + q.target_language for q in data.queries}, {"enen", "enes", "esen", "eses"}
        )
        self.assertEqual({j.review for q in data.queries for j in q.judgments}, {"authored"})
        for q in data.queries:
            self.assertEqual(len(q.judgments), 16)

    def test_valid(self):
        data = parse_dataset(fixture())
        self.assertEqual(data.queries[0].judgments[0].grade, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text(json.dumps(fixture()), encoding="utf-8")
            self.assertEqual(load_dataset(path), data)

    def test_invalid_roots(self):
        for raw in (None, [], {}, {**fixture(), "unknown": 1}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_dataset(raw)
        for key, value in (
            ("version", True),
            ("version", 2),
            ("name", " "),
            ("documents", []),
            ("documents", {}),
            ("queries", []),
        ):
            raw = fixture()
            raw[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                parse_dataset(raw)

    def test_bad_documents(self):
        for key, value in (("language", "xx"), ("text", 4), ("id", ""), ("source", None)):
            raw = fixture()
            raw["documents"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_dataset(raw)
        raw = fixture()
        raw["documents"].append(raw["documents"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_dataset(raw)

    def test_bad_queries(self):
        for key, value in (
            ("split", "test"),
            ("kind", "other"),
            ("entity", "yes"),
            ("accent", "maybe"),
            ("language", "xx"),
            ("family", ""),
            ("target_language", "es"),
            ("judgments", []),
        ):
            raw = fixture()
            raw["queries"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_dataset(raw)
        raw = fixture()
        raw["queries"].append(copy.deepcopy(raw["queries"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_dataset(raw)
        raw["queries"][1]["id"] = "q2"
        raw["queries"][1]["split"] = "eval"
        with self.assertRaisesRegex(ValueError, "cross splits"):
            parse_dataset(raw)

    def test_judgments(self):
        for key, value in (
            ("document_id", "missing"),
            ("grade", True),
            ("grade", 3),
            ("grade", -1),
            ("grade", 1.0),
            ("evidence", ""),
            ("evidence", None),
            ("evidence", "absent quote"),
            ("assessor", ""),
            ("review", "independent"),
            ("rationale", ""),
        ):
            raw = fixture()
            raw["queries"][0]["judgments"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                parse_dataset(raw)
        raw = fixture()
        raw["queries"][0]["judgments"].pop()
        with self.assertRaisesRegex(ValueError, "every candidate"):
            parse_dataset(raw)
        raw = fixture()
        raw["queries"][0]["judgments"].append(raw["queries"][0]["judgments"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_dataset(raw)

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                read_json(path)
