# Docs map

Start here to find the right document. **Current** docs describe the code as it is now and are kept up to date. **History** docs are records of past work: useful for the "why", but their file names and numbers may be out of date.

## Current

| Doc | Read it when you want to… |
|---|---|
| [`../README.md`](../README.md) | Get the overview, install, run a query, see the headline results |
| [`guides/evaluation.md`](guides/evaluation.md) | Run the evals: which script, what it needs, cost, where results go |
| [`guides/aapl50-benchmark.md`](guides/aapl50-benchmark.md) | See what the 50 Apple questions test and how they were written |
| [`engineering/BASELINE.md`](engineering/BASELINE.md) | Check the repo still installs, tests and imports as before (exact commands) |
| [`engineering/REVIEW.md`](engineering/REVIEW.md) | See known bugs and improvements, ranked, with their status |
| [`engineering/LLM_CALL_SITES.md`](engineering/LLM_CALL_SITES.md) | Find every place an AI model is called (as of the 2026-10-23 review) |
| [`engineering/PRINCIPLES.md`](engineering/PRINCIPLES.md) | Know the house rules for changes, and how to check each one |
| [`../results/FINAL.md`](../results/FINAL.md) | Read the full v1.0 results write-up (GPT-4o, 2026-10-01) |
| [`../results/README.md`](../results/README.md) | Trace a published number back to its result file and commit |
| [`../data/README.md`](../data/README.md) | Check eval dataset sources and licences |
| [`../.env.example`](../.env.example) | See every setting and what it does |

Code areas with their own rules have a short `CLAUDE.md` next to the code:
[`src/orchestration/`](../src/orchestration/CLAUDE.md),
[`src/retrieval/`](../src/retrieval/CLAUDE.md),
[`scripts/eval/`](../scripts/eval/CLAUDE.md),
[`tests/`](../tests/CLAUDE.md).

## History (kept as written)

| Date | Doc | What it records |
|---|---|---|
| 2026-08-27 | [`history/2026-08-27-first-principles-analysis.md`](history/2026-08-27-first-principles-analysis.md) | The original design questions and decisions |
| 2026-08-27 | [`history/2026-08-27-implementation-plan.md`](history/2026-08-27-implementation-plan.md) | The original build plan (its folder tree is out of date) |
| 2026-09-25 | [`history/2026-09-25-ingestion-log.md`](history/2026-09-25-ingestion-log.md) | Fixing SEC table parsing and chunking, step by step |
| 2026-09-25 | [`history/2026-09-25-chunking-analysis.md`](history/2026-09-25-chunking-analysis.md) | Before/after report on those chunking changes |
| 2026-10-01 | [`history/2026-10-01-eval-progress-log.md`](history/2026-10-01-eval-progress-log.md) | The eval work that led to v1.0 (Part A, Part B, Phases A–C) |
| 2026-10-24 | [`history/2026-10-24-commit-id-map.md`](history/2026-10-24-commit-id-map.md) | Old → new commit IDs after removing the FinanceBench file from history |

## Where new docs go

- A how-to that should stay true → `docs/guides/`
- A dated log, report or plan → `docs/history/YYYY-MM-DD-short-name.md`
- Review, baseline and engineering rules → `docs/engineering/`
- Then add one line to this map.
