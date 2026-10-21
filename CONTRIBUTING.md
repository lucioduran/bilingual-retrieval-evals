# Contributing

Use focused changes and regression tests. Run Ruff, mypy, the complete unit suite with coverage, and both saved-report replays. Keep the README heading structure. Use English prose and no em dash characters. Spanish is allowed only in evaluation query and document fixtures.

Do not tune on the evaluation families, overwrite measured evidence, or relabel a failure without a new dataset version and a documented rationale. Keep language variants in the same split. Independent human review claims require real reviewer records; authored labels are not independent review.

Model execution must stay local. On the series reference machine, use the shared benchmark lock described in the parent AGENTS.md. Do not bypass the lock, rent compute, call paid APIs, or modify sibling projects.
