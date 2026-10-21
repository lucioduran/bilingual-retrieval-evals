"""Expand original parallel fixtures into explicit, complete judgment records."""

import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
scenarios = json.loads((ROOT / "data/scenarios.json").read_text(encoding="utf-8"))
distractors = json.loads((ROOT / "data/distractors.json").read_text(encoding="utf-8"))
documents = [
    {
        "id": f"{s['id']}-{lang}",
        "language": lang,
        "text": s[f"{lang}_document"],
        "source": f"original fictional scenario: {s['id']}; MIT",
    }
    for s in scenarios
    for lang in ("en", "es")
] + [
    {
        "id": f"{d['id']}-{lang}",
        "language": lang,
        "text": d[lang],
        "source": f"original fictional distractor: {d['id']}; MIT",
    }
    for d in distractors
    for lang in ("en", "es")
]
queries = []
for s in scenarios:
    for language in ("en", "es"):
        text = s[f"{language}_query"]
        folded = "".join(
            c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
        )
        variants = [("original", text, "preserved" if text != folded else "none")]
        if text != folded:
            variants.append(("unaccented", folded, "omitted"))
        for variant, query_text, accent in variants:
            for target in ("en", "es"):
                judgments = []
                for doc in documents:
                    if doc["language"] != target:
                        continue
                    positive = doc["id"] == f"{s['id']}-{target}"
                    judgments.append(
                        {
                            "document_id": doc["id"],
                            "grade": 2 if positive else 0,
                            "evidence": doc["text"] if positive else "",
                            "rationale": s["intent"]
                            if positive
                            else "This document does not answer the scenario intent.",
                            "assessor": "fixture-authoring-process",
                            "review": "authored",
                        }
                    )
                queries.append(
                    {
                        "id": f"{s['id']}-{language}-{target}-{variant}",
                        "family": s["id"],
                        "split": s["split"],
                        "language": language,
                        "target_language": target,
                        "text": query_text,
                        "kind": s["kind"],
                        "entity": s["entity"],
                        "accent": accent,
                        "judgments": judgments,
                    }
                )
output = {
    "version": 1,
    "name": "fictional-community-v1",
    "documents": documents,
    "queries": queries,
}
(ROOT / "data/dataset.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
