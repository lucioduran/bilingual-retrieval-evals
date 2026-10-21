# Evaluation breakdowns

All tables below are derived from `results/eval/report.json`. Original-only tables exclude accent-omission variants. NDCG is at cutoff three. Groups are descriptive and correlated; they are not independent experimental treatments.

| Original query group | Queries | BM25 | Folded BM25 | Dense |
| --- | ---: | ---: | ---: | ---: |
| accent/none | 18 | 0.8333 | 0.8333 | 0.9795 |
| accent/preserved | 14 | 0.5714 | 0.5714 | 0.9736 |
| direction/en->en | 8 | 1.0000 | 1.0000 | 1.0000 |
| direction/en->es | 8 | 0.6250 | 0.6250 | 0.9539 |
| direction/es->en | 8 | 0.3750 | 0.3750 | 0.9539 |
| direction/es->es | 8 | 0.8750 | 0.8750 | 1.0000 |
| entity/absent | 20 | 0.5500 | 0.5500 | 0.9631 |
| entity/present | 12 | 1.0000 | 1.0000 | 1.0000 |
| kind/cultural | 16 | 0.6875 | 0.6875 | 0.9769 |
| kind/parallel | 16 | 0.7500 | 0.7500 | 0.9769 |
| language/en | 16 | 0.8125 | 0.8125 | 0.9769 |
| language/es | 16 | 0.6250 | 0.6250 | 0.9769 |
| overall | 32 | 0.7188 | 0.7188 | 0.9769 |

## Matched accent contrasts

Difference is omitted minus original Spanish NDCG@3, paired within family and target language. Seven evaluation families have accented Spanish originals.

| Target documents | Pairs | BM25 | Folded BM25 | Dense |
| --- | ---: | ---: | ---: | ---: |
| en | 7 | +0.0000 | +0.0000 | +0.0000 |
| es | 7 | -0.0527 | +0.0000 | +0.0000 |

## Family uncertainty

These intervals use all variants, averaged within family, with equal family weights. They do not use the original-only pooled estimator in the first table. There are eight families and 2,000 seeded percentile bootstrap resamples.

| Method | Family mean NDCG@3 | 95% interval | Paired difference from BM25 | 95% difference interval |
| --- | ---: | --- | ---: | --- |
| bm25 | 0.6798 | [0.5055, 0.8542] | +0.0000 | [0.0000, 0.0000] |
| bm25-folded | 0.6875 | [0.5208, 0.8542] | +0.0077 | [0.0000, 0.0231] |
| dense | 0.9769 | [0.9462, 1.0000] | +0.2971 | [0.1250, 0.4539] |

## Error inspection

Dense retrieval puts the relevant document at rank two for three variants: `mate-es-en-original`, `mate-es-en-unaccented`, and `parcel-en-es-original`. The first two are one underlying intent, not two independent failures. All other dense variants rank the relevant document first.

The shared-drink Spanish query retrieves the English bait distractor above the relevant drink document. This is a concrete cultural ambiguity failure despite perfect dense Recall@3. The English parcel query retrieves the Spanish lost-transit-pass document before the parcel document.

For BM25, the unaccented Spanish post-meal conversation query against Spanish documents drops from rank one to rank two; folding accents restores rank one. Several cross-language failures remain with either lexical setting. Named fictional institutions provide shared tokens, which helps explain the entity-present subgroup results, but this dataset does not isolate that effect experimentally.

[All imperfect query/method rankings](../results/eval/errors.json) include top-three IDs, relevant ranks, and NDCG. The raw report preserves complete rankings for successes too. Labels were not changed after inspecting these outcomes.

## Runtime

Measured on Intel Xeon W-3223 at 3.50 GHz, macOS 26.7.1, Python 3.11.13, two CPU threads. The evaluation pass took 5.401 seconds to load and hash the model, 1.313 seconds to embed 32 documents, 0.294 seconds to embed 46 queries, and 0.129 seconds to rank and score. These are single batched elapsed measurements, not latency percentiles.

The longest document was 53 tokens and the longest evaluation query was 22 tokens, below the 128-token limit. No measured input was truncated. Both saved runs recorded a clean tracked working tree. The evaluation source revision is `d89585a8e53aa43a0fa8bc6fe50d43635172c5de`; later changes add documentation, tests, and portable LF output handling, and a 1e-12 replay tolerance for Python summation roundoff without changing retrieval settings or measured results.
