# Fixture provenance

All scenarios, document text, queries, distractors, and judgments are original synthetic material created for this project and licensed under MIT. Named institutions are fictional. No facts about actual transit systems, hospitals, libraries, or businesses should be inferred from these examples.

`scenarios.json` holds the bilingual source fixtures and English intent rationales. `distractors.json` holds lexical controls. `dataset.json` is the expanded evaluation input with explicit judgments and accent variants. Rebuild with `python scripts/build_dataset.py`; the output must remain byte-identical unless the fixture version is intentionally changed.

Spanish text appears only in evaluation fixture files. English metadata, code, documentation, reports, and commit messages remain English. The user explicitly authorized this limited language exception before the fixtures were created.

The current labels are authored, not independently human-reviewed. Their evidence spans are mechanically checked against documents. A proposed manual review process and all interpretive limits are documented in [methodology](../docs/methodology.md). Relevance is defined by the scenario intent, which is visible to the authoring process. This is a deliberately transparent diagnostic set, not hidden test data.
