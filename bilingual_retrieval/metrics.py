"""Explicit denominators and family-level bootstrap summaries."""

import math
import random
from collections import defaultdict
from typing import TypedDict

from .contracts import Query


class Scores(TypedDict):
    recall: float | None
    ndcg: float | None
    reciprocal_rank: float | None


def score(ranking: list[str], grades: dict[str, int], k: int) -> Scores:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if len(ranking) != len(set(ranking)) or any(d not in grades for d in ranking):
        raise ValueError("ranking contains duplicate or unjudged documents")
    if any(type(g) is not int or g not in (0, 1, 2) for g in grades.values()):
        raise ValueError("invalid relevance grade")
    relevant = sum(g > 0 for g in grades.values())
    if not relevant:
        return Scores(recall=None, ndcg=None, reciprocal_rank=None)
    selected = [grades[d] for d in ranking[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]

    def dcg(values: list[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(values))

    return Scores(recall=sum(g > 0 for g in selected) / relevant,
                  ndcg=dcg(selected) / dcg(ideal),
                  reciprocal_rank=next((1 / (i + 1) for i, g in enumerate(selected) if g > 0), 0.0))


def aggregate(rows: list[Scores]) -> dict[str, object]:
    result: dict[str, object] = {"queries": len(rows)}
    for name in ("recall", "ndcg", "reciprocal_rank"):
        values = [r[name] for r in rows if r[name] is not None]  # type: ignore[literal-required]
        result[name] = {"n": len(values), "mean": sum(values) / len(values) if values else None}
    return result


def breakdown(queries: list[Query], scores: dict[str, Scores]) -> dict[str, object]:
    groups: dict[str, list[Scores]] = defaultdict(list)
    for q in queries:
        groups["overall"].append(scores[q.id])
        for dimension, value in (("language", q.language),
                                 ("direction", f"{q.language}->{q.target_language}"),
                                 ("entity", q.entity), ("accent", q.accent), ("kind", q.kind)):
            groups[f"{dimension}/{value}"].append(scores[q.id])
    return {key: aggregate(rows) for key, rows in sorted(groups.items())}


def family_interval(values: dict[str, list[float]], seed: int = 1729,
                    resamples: int = 2000) -> dict[str, object]:
    if not values or any(not group for group in values.values()) or resamples < 1:
        raise ValueError("bootstrap requires nonempty families and positive resamples")
    if any(not math.isfinite(v) for group in values.values() for v in group):
        raise ValueError("bootstrap values must be finite")
    groups = [values[key] for key in sorted(values)]
    means = [sum(group) / len(group) for group in groups]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(resamples))
    return {"families": len(groups), "mean": sum(means) / len(means),
            "low": draws[int((resamples - 1) * 0.025)],
            "high": draws[int((resamples - 1) * 0.975)], "seed": seed, "resamples": resamples,
            "weighting": "equal family weight"}
