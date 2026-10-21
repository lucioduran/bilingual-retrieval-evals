# Bilingual Retrieval Evals

[![CI](https://github.com/lucioduran/bilingual-retrieval-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/lucioduran/bilingual-retrieval-evals/actions/workflows/ci.yml)

## Problem

A single retrieval average can hide failures across languages, local expressions, named entities, and missing accents. This project makes those differences inspectable with parallel English-Spanish queries, monolingual candidate documents, traceable relevance judgments, and saved local model evidence.

## Scope

32 original fictional documents, 68 query variants, four development families, and eight evaluation families. Compare BM25, accent-folded BM25, and pinned multilingual MiniLM cosine retrieval across English-to-English, English-to-Spanish, Spanish-to-English, and Spanish-to-Spanish search. Four evaluation families test contextual expressions with meaning-preserving English paraphrases and lexical distractors.

Labels are authored, not independently human-reviewed. This is a small diagnostic set, not a representative benchmark of Spanish speakers. See [fixture provenance](data/README.md) and [manual review methodology](docs/methodology.md).

## Quick start

Python 3.11 or later, no model dependencies needed for validation or replay:

```sh
git clone https://github.com/lucioduran/bilingual-retrieval-evals.git
cd bilingual-retrieval-evals
python -m bilingual_retrieval validate
python -m unittest discover -v
python -m bilingual_retrieval verify --report results/dev/report.json
```

## Usage

| Command | Purpose |
| --- | --- |
| `validate --dataset PATH` | Check identities, split isolation, all candidate labels, and evidence spans |
| `download --cache PATH` | Explicitly download the pinned public model |
| `run --split dev --output NEW_DIRECTORY` | Run local CPU inference and save evidence |
| `verify --report PATH` | Recompute rankings, metrics, paired contrasts, and breakdowns from saved vectors |

`run` also accepts `--dataset`, `--cache`, `--k` (default 3), and `--threads` (default 2, maximum 4). Each run requires a fresh destination. A failed run may leave a directory; it does not claim successful evidence. Exit code 0 means success and 2 means a command/input/environment failure. Inputs are trusted local files.

Reports contain full rankings, per-query recall/NDCG/reciprocal rank, language/direction/entity/accent/kind breakdowns, original-only breakdowns, matched contrasts, and family bootstrap intervals. Summary files contain the all-variant breakdowns. Use original-only results for balanced directional comparisons; Spanish accent variants otherwise receive extra weight.

## Architecture

```text
Original scenarios and distractors
              |
Complete judgments with exact supporting spans
              |
Strict schema and family split checks
              |
Target-language candidates (16 per query)
              |
BM25 / accent-folded BM25 / pinned local multilingual encoder
              |
Full rankings, metrics, matched contrasts, family intervals
              |
Hashed inputs + vectors + runtime evidence + portable replay
```

The portable evaluator uses only the Python standard library. Neural libraries are imported only for explicit download or inference. [Methodology](docs/methodology.md) defines formulas, pairing, sampling, review procedure, and verification boundaries.

## Reproduction

Reference environment: Intel macOS, Python 3.11.13, CPU execution with two threads. The model revision is `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`; see [model notices](docs/model-notices.md). The environment lock is specific to the measured platform.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-local.lock
python -m bilingual_retrieval download
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m bilingual_retrieval run --split dev --output build/dev
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m bilingual_retrieval run --split eval --output build/eval
python -m bilingual_retrieval verify --report build/dev/report.json
python -m bilingual_retrieval verify --report build/eval/report.json
```

On the series reference machine, every neural run is additionally wrapped in `python3 /Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py COMMAND ARGUMENTS`. This external coordinator wrapper serializes heavy jobs and is not needed on an unrelated machine. Limit BLAS/OpenMP threads to two when reproducing outside that wrapper. No paid services or rented compute are used.

Frozen settings: cutoff 3, BM25 `k1=1.2`, `b=0.75`, 128 embedding tokens, batch size eight, seed 1729, 2,000 family bootstrap resamples. No fitting or tuning on evaluation data. Evidence records actual hardware, package versions, input/model hashes, source revision, token lengths, and separate load/encoding/scoring timings.

## Results

The recorded development run is available in [raw evidence](results/dev/report.json) and [breakdowns](results/dev/summary.json). Evaluation measurements are being prepared with the same frozen settings.

## Testing

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m mypy bilingual_retrieval
python -m coverage run -m unittest discover -v
python -m coverage report
python scripts/build_dataset.py
git diff --exit-code -- data/dataset.json
```

CI checks Linux, macOS, and Windows with Python 3.11.9, pinned action commits, strict typing, formatting, unit tests, a 90% coverage gate, fixture byte reproduction, and saved-report replay. Adapter tests use mocks; recorded CPU runs provide actual neural execution evidence. CI does not rerun neural inference.

## Limitations

- Small synthetic and author-known data cannot establish production quality or dialect coverage. Labels and cultural interpretations await independent bilingual review.
- One encoder and two lexical settings are not a broad model comparison. All shipped positive judgments have grade 2; partial-grade scoring is tested separately.
- Entity and language groups differ in content. Their score gaps are descriptive, not causal. Mechanical accent removal may change meaning.
- Eight evaluation families provide weak uncertainty estimates. Parallel queries and accent variants are correlated and must not be counted as independent observations.
- Replay checks internal ranking/scoring consistency, not authenticity of saved embeddings, model hashes, hardware metadata, timing, or human review.
- Timings are single batched passes, not load tests or per-request latency. Inference uses a platform-specific pinned stack; other hardware may produce different vectors or rankings.
- The CLI loads the corpus and reports into memory. It provides no hosted service, relevance-learning pipeline, or downstream answer-quality evaluation.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Preserve old measurements, version label changes, keep language variants together, and add regression tests. [SECURITY.md](SECURITY.md) describes the local-file trust boundary.

## License

Code, original fixtures, labels, documentation, and recorded evidence use [MIT](LICENSE). The downloaded model is Apache-2.0 and is not redistributed. Third-party dependencies retain their own licenses.
