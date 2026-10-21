# Evaluation design

This is a small, original, synthetic diagnostic set, not a representative benchmark of Spanish speakers. All settings and labels were committed before neural evaluation. The labels and scenarios were produced during assisted fixture authoring. No independent human review, annotator agreement, or externally blind evaluation is claimed.

## Units and splits

The corpus contains 32 monolingual documents: 12 scenario pairs and four distractor pairs. Each query searches only the 16 documents in its declared target language. Both language versions of an intent belong to the same family and split. Four families are development data; eight are evaluation data. Development and evaluation use a common candidate corpus but disjoint relevant scenario families. Documents are embedded whole, without chunking. Equivalent document pairs intentionally describe equivalent fictional facts.

There are 68 queries: 48 original language/direction combinations and 20 accent-omission variants. Development has 22 queries and evaluation has 46. Each original intent is searched in both document languages. Spanish query variants remove Unicode combining marks, including any tilde. This transformation is a mechanical stress test and can alter meaning; it is not a claim that accent removal is linguistically harmless. Entity presence is an authored binary tag, not automatic entity recognition or an isolated causal test.

Four evaluation families use contextual expressions associated with regional usage: shared yerba drink preparation, post-meal conversation, pooled gift money, and public transit. Their English counterparts express the intended meaning instead of translating each word literally. The corpus also includes distractors for bait, a tabletop object, a small cow, and financial bonds. Not every distractor is an equally plausible literal interpretation. In particular, the financial bond distractor is a lexical control, not an asserted translation of the transit expression. No regional coverage or dialect validation is claimed.

## Relevance and review protocol

Every query has an explicit judgment for every candidate document. Grade 2 directly answers the scenario intent; grade 1 would provide useful partial context; grade 0 does not answer it. The shipped fixtures use grades 0 and 2 only. The scorer supports all three grades, tested against a hand calculation. Positive judgments include an exact evidence substring and an English rationale. The dataset records the assessor and review status for every judgment. All current judgments have `review: authored` and assessor `fixture-authoring-process`.

A future manual review should proceed before changing labels or making human-validated claims:

1. Recruit at least two independent fluent Spanish-English reviewers, recording their relevant regional familiarity and consent to attribution.
2. Give reviewers the query, its intended scenario, and shuffled candidate documents without retrieval rankings or method names.
3. Ask each reviewer to assign grades, select supporting spans, flag unnatural phrasing, and explain cultural interpretation. Ask separately whether the English paraphrase preserves the Spanish intent.
4. Preserve both original judgment files. Measure raw agreement and an appropriate ordinal agreement statistic with the unit and denominator stated. Do not treat accent variants as independent intents.
5. Adjudicate disagreements with a documented rationale. Keep unresolved items marked pending and report their count. A third reviewer may adjudicate; identity and independence must be recorded accurately.
6. Publish reviewed labels as a new dataset version, then rerun all methods. Never silently revise the present labels after seeing a model failure.

The current `review` field records workflow state, not proof of independence. A reviewed claim requires the external reviewer records described above. The parser checks evidence containment and structure; it cannot establish semantic truth, language correctness, or human provenance.

## Retrieval

BM25 uses Unicode word tokens, case folding, no stopword list, `k1=1.2`, and `b=0.75`. Query terms are deduplicated. Document frequency and length statistics use the target-language candidate set. A second BM25 configuration removes combining marks from queries and documents. Scores tie-break by document ID. All documents, including zero-score matches, appear in the ranking.

Dense retrieval uses a pinned multilingual MiniLM encoder, attention-mask mean pooling, 128 tokens, batches of eight, CPU evaluation mode, and two CPU threads. Cosine ranking uses the saved vectors and the same target-language candidates. Token lengths before truncation are recorded. No translation service, hosted inference, generation model, fine-tuning, or paid API is used.

## Metrics and comparisons

At cutoff 3, recall is the fraction of all positive-grade candidates retrieved. Reciprocal rank is zero if no positive appears in the top three. NDCG uses gain `2^grade - 1` and discount `log2(rank + 1)`, divided by the ideal top-three DCG. Queries without any relevant document receive null metrics and are excluded from metric denominators, not assigned artificial zeros. The shipped fixture has no unanswerable queries.

Reports contain both all-variant breakdowns and original-only breakdowns. Original-only results are the headline comparison: each direction has eight evaluation queries. All-variant averages give Spanish extra weight, so they must not be compared as balanced language aggregates. Breakdowns include query language, direction, entity presence, accent state, and cultural versus parallel kind, with denominators.

Matched contrasts pair Spanish and English originals within family and target language. Accent contrasts pair an omitted-accent query with its Spanish original. Each record names the exact query IDs. Only uniquely matched pairs are included; ambiguous or absent pairs are omitted. These are descriptive paired differences, not proofs of causal language or accent effects.

Family intervals average query variants within each family, then weight families equally. The percentile bootstrap resamples families 2,000 times with seed 1729. Paired method differences use the same family grouping. These estimates differ from pooled query means when family sizes differ. Only eight evaluation families exist, and cultural subgroups are smaller still. Intervals describe this authored sample and do not imply population coverage.

## Evidence and timing

A report saves dataset SHA-256, source revision, dirty state, model revision and file hashes, installed package versions, actual CPU/OS/Python details, thread count, vectors, token lengths, complete rankings, and metrics. Replay checks the supplied dataset hash, fixed model identity, and recomputed rankings and metrics. Derived floating-point values allow an absolute tolerance of 1e-12 for cross-version summation roundoff; identifiers, ordering, integer counts, and structure remain exact. It does not rerun inference or authenticate file hashes, hardware metadata, authorship, or elapsed times. An attacker who changes vectors and all derived results can create an internally consistent report.

Timings separate model loading, batched document encoding, batched query encoding, and ranking plus scoring. Loading includes model-file hashing. These are one-pass elapsed durations, not request latency percentiles, concurrency measurements, or throughput benchmarks. Neural runs use the shared exclusive benchmark wrapper on the reference machine. Hardware and electricity costs are unmeasured.
