"""Deterministic lexical and vector ranking with ID tie breaking."""

import math
import re
import unicodedata
from collections import Counter


def tokens(text: str, fold_accents: bool = False) -> list[str]:
    text = unicodedata.normalize("NFC", text).casefold()
    if fold_accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text)
                       if not unicodedata.combining(c))
    return re.findall(r"[^\W_]+", text, re.UNICODE)


def bm25(documents: dict[str, str], query: str, fold_accents: bool = False) -> list[str]:
    if not documents:
        raise ValueError("cannot rank an empty corpus")
    counts = {did: Counter(tokens(text, fold_accents)) for did, text in documents.items()}
    lengths = {did: sum(c.values()) for did, c in counts.items()}
    average = sum(lengths.values()) / len(lengths)
    scores = dict.fromkeys(documents, 0.0)
    for term in set(tokens(query, fold_accents)):
        df = sum(c[term] > 0 for c in counts.values())
        idf = math.log(1 + (len(counts) - df + 0.5) / (df + 0.5))
        for did, count in counts.items():
            tf = count[term]
            if tf:
                scores[did] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lengths[did] / average))
    return sorted(scores, key=lambda did: (-scores[did], did))


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("vectors must have equal nonzero dimensions")
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in left + right):
        raise ValueError("vectors must contain finite numbers")
    a, b = math.hypot(*left), math.hypot(*right)
    if a == 0 or b == 0:
        raise ValueError("zero vectors cannot be ranked")
    return sum((x / a) * (y / b) for x, y in zip(left, right, strict=True))


def dense(documents: dict[str, list[float]], query: list[float]) -> list[str]:
    if not documents:
        raise ValueError("cannot rank an empty corpus")
    scores = {did: cosine(vector, query) for did, vector in documents.items()}
    return sorted(scores, key=lambda did: (-scores[did], did))
