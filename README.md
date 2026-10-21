# Bilingual Retrieval Evals

## Problem

Aggregate retrieval scores can hide language, entity, accent, and cultural failures.

## Scope

English infrastructure is under development. The Spanish fixture exception is pending coordinator confirmation. No bilingual evaluation or final release is claimed.

## Quick start

Python 3.11 or later: `python -m unittest discover -v`.

## Usage

Strict datasets contain monolingual documents, query families, and complete candidate judgments.

## Architecture

Validated inputs feed deterministic rankings, per-query metrics, and stratified reports.

## Reproduction

Local CPU only. Neural evidence is not recorded yet.

## Results

No benchmark results yet.

## Testing

Install `requirements-dev.txt`, then run unittest, Ruff, and mypy.

## Limitations

Pending Spanish fixture authorization blocks bilingual evaluation and release. Authored labels do not constitute independent human review.

## Contributing

Use English prose, focused commits, and regression tests. Preserve evidence and record limitations.

## License

[MIT](LICENSE).
