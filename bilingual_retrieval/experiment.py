"""Evidence production and deterministic replay without model dependencies."""

import importlib.metadata
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import Dataset, Query, fields, load_dataset, read_json
from .metrics import Scores, breakdown, family_interval, matched_contrasts, score
from .models import MAX_TOKENS, MODEL_ID, REVISION, Encoder, digest
from .retrieval import bm25, dense

METHODS = ("bm25", "bm25-folded", "dense")


def selected(dataset: Dataset, split: str) -> list[Query]:
    if split not in ("dev", "eval"):
        raise ValueError("split must be dev or eval")
    queries = [q for q in dataset.queries if q.split == split]
    if not queries:
        raise ValueError("selected split is empty")
    return queries


def evaluate(
    dataset: Dataset,
    split: str,
    k: int,
    document_vectors: dict[str, list[float]],
    query_vectors: dict[str, list[float]],
) -> dict[str, Any]:
    queries = selected(dataset, split)
    if set(document_vectors) != {doc.id for doc in dataset.documents}:
        raise ValueError("document vector identifiers do not match dataset")
    if set(query_vectors) != {q.id for q in queries}:
        raise ValueError("query vector identifiers do not match split")
    traces: dict[str, Any] = {}
    scores_by_method: dict[str, dict[str, Scores]] = {method: {} for method in METHODS}
    for q in queries:
        candidates = {
            doc.id: doc.text for doc in dataset.documents if doc.language == q.target_language
        }
        rankings = {
            "bm25": bm25(candidates, q.text),
            "bm25-folded": bm25(candidates, q.text, True),
            "dense": dense({did: document_vectors[did] for did in candidates}, query_vectors[q.id]),
        }
        grades = {j.document_id: j.grade for j in q.judgments}
        traces[q.id] = {}
        for method, ranking in rankings.items():
            scores = score(ranking, grades, k)
            scores_by_method[method][q.id] = scores
            traces[q.id][method] = {"ranking": ranking, "scores": scores}
    summaries = {method: breakdown(queries, scores) for method, scores in scores_by_method.items()}
    intervals: dict[str, Any] = {}
    for method in METHODS:
        values: dict[str, list[float]] = defaultdict(list)
        deltas: dict[str, list[float]] = defaultdict(list)
        for q in queries:
            value = scores_by_method[method][q.id]["ndcg"]
            baseline = scores_by_method["bm25"][q.id]["ndcg"]
            if value is not None and baseline is not None:
                values[q.family].append(value)
                deltas[q.family].append(value - baseline)
        intervals[method] = {
            "ndcg": family_interval(values) if values else None,
            "ndcg_minus_bm25": family_interval(deltas) if deltas else None,
        }
    original_queries = [q for q in queries if q.accent != "omitted"]
    return {
        "traces": traces,
        "breakdowns": summaries,
        "family_intervals": intervals,
        "original_breakdowns": {
            method: breakdown(original_queries, scores)
            for method, scores in scores_by_method.items()
        },
        "matched_contrasts": {
            method: matched_contrasts(queries, scores)
            for method, scores in scores_by_method.items()
        },
    }


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()


def environment(threads: int) -> dict[str, Any]:
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "threads": threads,
        "device": "cpu",
        "revision": git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "numpy",
                "tokenizers",
                "huggingface-hub",
                "safetensors",
            )
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run(dataset_path: Path, output: Path, cache: Path, split: str, k: int, threads: int) -> None:
    dataset = load_dataset(dataset_path)
    queries = selected(dataset, split)
    if type(k) is not int or k < 1:
        raise ValueError("k must be positive")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    encoder = Encoder(cache, threads)
    loaded = time.perf_counter()
    doc_vectors, doc_lengths = encoder.encode([doc.text for doc in dataset.documents])
    indexed = time.perf_counter()
    q_vectors, q_lengths = encoder.encode([q.text for q in queries])
    encoded = time.perf_counter()
    vectors = {
        "documents": dict(zip([doc.id for doc in dataset.documents], doc_vectors, strict=True)),
        "queries": dict(zip([q.id for q in queries], q_vectors, strict=True)),
    }
    result = evaluate(dataset, split, k, vectors["documents"], vectors["queries"])
    ended = time.perf_counter()
    report = {
        "version": 1,
        "dataset_sha256": digest(dataset_path),
        "split": split,
        "k": k,
        "created_utc": datetime.now(UTC).isoformat(),
        "environment": environment(threads),
        "model": {
            "id": MODEL_ID,
            "revision": REVISION,
            "files": encoder.files,
            "max_tokens": MAX_TOKENS,
            "pooling": "attention-mask mean",
            "batch_size": 8,
        },
        "token_lengths": {"documents": doc_lengths, "queries": q_lengths},
        "timing_seconds": {
            "load": loaded - started,
            "documents": indexed - loaded,
            "queries": encoded - indexed,
            "ranking_and_scoring": ended - encoded,
        },
        "vectors": vectors,
        "evaluation": result,
    }
    write_json(output / "report.json", report)
    verify(dataset_path, output / "report.json")
    write_json(output / "summary.json", result["breakdowns"])


def equivalent(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float):
        return (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12)
        )
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            equivalent(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            equivalent(left, right) for left, right in zip(expected, actual, strict=True)
        )
    return bool(expected == actual)


def verify(dataset_path: Path, report_path: Path) -> None:
    report = fields(
        read_json(report_path),
        "version dataset_sha256 split k created_utc environment model token_lengths timing_seconds vectors evaluation",
        "report",
    )
    if type(report["version"]) is not int or report["version"] != 1:
        raise ValueError("unsupported report version")
    if report["dataset_sha256"] != digest(dataset_path):
        raise ValueError("dataset hash mismatch")
    model = report["model"]
    if (
        model["id"] != MODEL_ID
        or model["revision"] != REVISION
        or model["max_tokens"] != MAX_TOKENS
    ):
        raise ValueError("model configuration mismatch")
    dataset = load_dataset(dataset_path)
    vectors = fields(report["vectors"], "documents queries", "vectors")
    expected = evaluate(
        dataset, report["split"], report["k"], vectors["documents"], vectors["queries"]
    )
    if not equivalent(expected, report["evaluation"]):
        raise ValueError("rankings or metrics differ from evidence replay")
