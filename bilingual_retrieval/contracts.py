"""Strict dataset loading with complete, traceable relevance judgments."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    language: str
    text: str
    source: str


@dataclass(frozen=True)
class Judgment:
    document_id: str
    grade: int
    evidence: str
    rationale: str
    assessor: str
    review: str


@dataclass(frozen=True)
class Query:
    id: str
    family: str
    split: str
    language: str
    target_language: str
    text: str
    kind: str
    entity: str
    accent: str
    judgments: tuple[Judgment, ...]


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]


def fields(value: Any, expected: str, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError(f"{context}: expected exactly {expected}")
    return value


def string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected nonempty string")
    return value


def choice(value: Any, options: str, context: str) -> str:
    result = string(value, context)
    if result not in options.split():
        raise ValueError(f"{context}: expected one of {options}")
    return result


def array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}: expected nonempty array")
    return value


def unique(ids: list[str], context: str) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context}: duplicate identifiers")


def parse_dataset(raw: Any) -> Dataset:
    root = fields(raw, "version name documents queries", "dataset")
    if type(root["version"]) is not int or root["version"] != 1:
        raise ValueError("dataset: version must be 1")
    docs = []
    for raw_doc in array(root["documents"], "documents"):
        doc = fields(raw_doc, "id language text source", "document")
        docs.append(Document(string(doc["id"], "id"), choice(doc["language"], "en es", "language"),
                             string(doc["text"], "text"), string(doc["source"], "source")))
    unique([doc.id for doc in docs], "documents")
    by_id = {doc.id: doc for doc in docs}
    queries = []
    families: dict[str, str] = {}
    for raw_query in array(root["queries"], "queries"):
        q = fields(raw_query, "id family split language target_language text kind entity accent judgments", "query")
        family = string(q["family"], "family")
        split = choice(q["split"], "dev eval", "split")
        if family in families and families[family] != split:
            raise ValueError("query family cannot cross splits")
        families[family] = split
        target = choice(q["target_language"], "en es", "target_language")
        judgments = []
        for raw_judgment in array(q["judgments"], "judgments"):
            j = fields(raw_judgment, "document_id grade evidence rationale assessor review", "judgment")
            did = string(j["document_id"], "document_id")
            if did not in by_id or by_id[did].language != target:
                raise ValueError("judgment references an unknown or wrong-language document")
            grade = j["grade"]
            if type(grade) is not int or grade not in (0, 1, 2):
                raise ValueError("grade must be an integer in [0, 2]")
            evidence = j["evidence"]
            if not isinstance(evidence, str) or (grade > 0 and not evidence.strip()):
                raise ValueError("positive judgments require evidence")
            if evidence and evidence not in by_id[did].text:
                raise ValueError("evidence must be an exact document substring")
            judgments.append(Judgment(did, grade, evidence, string(j["rationale"], "rationale"),
                                      string(j["assessor"], "assessor"),
                                      choice(j["review"], "authored pending reviewed", "review")))
        unique([j.document_id for j in judgments], "judgments")
        if {j.document_id for j in judgments} != {doc.id for doc in docs if doc.language == target}:
            raise ValueError("every candidate document must have an explicit judgment")
        queries.append(Query(string(q["id"], "id"), family, split,
                             choice(q["language"], "en es", "language"), target,
                             string(q["text"], "text"),
                             choice(q["kind"], "parallel cultural control", "kind"),
                             choice(q["entity"], "present absent", "entity"),
                             choice(q["accent"], "preserved omitted none", "accent"), tuple(judgments)))
    unique([q.id for q in queries], "queries")
    return Dataset(string(root["name"], "name"), tuple(docs), tuple(queries))


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def load_dataset(path: Path) -> Dataset:
    return parse_dataset(read_json(path))
