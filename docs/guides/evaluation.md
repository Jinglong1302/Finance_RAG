# Running the evaluations

How to rerun the benchmarks behind the README numbers: what each script measures, what it needs, what it costs, and how to tell a good run from a broken one.

## Before you start

You need all of these:

1. **Qdrant with the index loaded**: collection `sec_filings`, 6,971 points. Set `QDRANT_URL` (and `QDRANT_API_KEY` for Qdrant Cloud).
   For Qdrant Cloud, put the port in the URL: `https://<cluster>.cloud.qdrant.io:443`. Without a port, the Qdrant Python client uses 6333, which some networks block. (Seen in the cloud VM on 2026-10-23.)
2. **An API key for your `LLM_PROVIDER`**: `GEMINI_API_KEY` (default) or `OPENAI_API_KEY`. See "Choosing the AI provider" in the main README.
3. **The two Hugging Face models**: they download on first use, about 3.3 GB (BGE-M3 2.2 GB plus `bge-reranker-base` 1.1 GB).
4. **FinanceBench data** (FB27 scripts only): if `data/eval/financebench_in_corpus.jsonl` is missing, the scripts download it from Hugging Face. Never commit that file: its licence is CC BY-NC 4.0, see `data/README.md`.

Quick check that the pipeline works end to end before spending on a full run:

```bash
poetry run python scripts/query.py "What was Apple's total net sales in fiscal 2024?"
# expect: about $391,035 million, confidence HIGH, citations to the AAPL FY2024 10-K
```

## The scripts behind the published numbers

| Benchmark | Script | Questions | Uses the production graph? | Disk cache |
|---|---|---|---|---|
| FB27, production path (v1.0 headline) | `scripts/eval/eval_fb27_no_injection.py` | 27 | No, its own copy of the steps (same Part B grader) | yes |
| AAPL50 (clean held-out) | `scripts/eval/eval_part_b_aapl50.py` | 50 | No, its own copy of the steps | yes |
| OOC20 abstention | `scripts/eval/eval_abstention.py --no-baseline` | 20 | Yes | no |
| Part B tiers on FB27 (+ OOC20 with `--tier 2`) | `scripts/eval/eval_part_b.py --tier 1` / `--tier 2` | 16 / 22+20 | No | yes |
| Holdout-22, old per-chunk grader | `scripts/eval/eval_crag_holdout22.py` | 22 | No | yes |
| Part A ablations, oracle, dev-5 | `scripts/eval/eval_part_a_consolidated.py` | 93 runs (FINAL.md) | Partly (dev-5 uses the graph) | no |
| Ragas scores on answered FB27 + AAPL50 | `scripts/eval/eval_ragas_phase_c.py` | answered subsets | n/a (reads saved results) | yes |
| All four tracks on the graph, plus Ragas | `scripts/eval/eval_fullscale_baselines.py` | 27+50+50+20 | Yes | no |
| Plumbing test with a fake LLM (no API cost) | `scripts/eval/smoke_test_stub.py` | 1 | No | stub |

"Its own copy of the steps" means the script calls the pipeline steps in its own loop instead of `build_crag_graph`. These loops can drift from the real app (REVIEW.md item 19).

Every script writes a timestamped JSON to `results/eval/` with the git SHA and a "dirty" flag. `results/` is gitignored except `FINAL.md`, `README.md` and `archive/INDEX.md`, so result files stay on your machine. When a number goes into the README, add its file to `results/README.md`.

## Cost and time

| Provider | Cost per question (approx.) | Notes |
|---|---|---|
| OpenAI GPT-4o | ~$0.025–0.035 | From `results/FINAL.md`: FB27 run $0.92, AAPL50 run $1.25 |
| Gemini 2.5 Flash, paid tier | ~$0.003 | One measured CLI query on 2026-10-23; prices from third-party trackers |
| Gemini, **free tier** | $0 | **Only 20 requests per day per model** (gemini-2.5-flash, checked 2026-10-23). One question makes about 4–6 requests, so the free tier cannot run these evals. |

The disk cache (`results/llm_cache/`) saves each answer by its inputs, so rerunning unchanged code costs nothing. Delete the folder to force fresh calls.

Time: on a 4-CPU machine without a GPU, the first question also waits 1–2 minutes for the models to load. After that, expect roughly 10–90 seconds per question, depending on rewrite cycles and rate-limit retries.

## How to tell a broken run

The pipeline turns a failed API call into an answer, either an error message or a pass-through (REVIEW.md item 6). So a run can finish "successfully" with junk scores. After every run:

```bash
grep -ciE "429|RESOURCE_EXHAUSTED|quota|Error generating answer|failed:" <your-log-file>
```

Anything above 0 means some questions were scored on errors. Fix the cause, then rerun those questions. Other warning signs:
- coverage suddenly near 0% (every question refused);
- many answers starting "Error";
- the run finishing much faster than usual.

## Comparing against the published numbers

The README and `results/FINAL.md` numbers were measured with **OpenAI GPT-4o**, and before the reranker fix of 2026-10-23: at that time the cross-encoder was never actually called (REVIEW.md item 1). A new run with Gemini, or with the fix, is a new baseline, not a like-for-like comparison. Record the provider, model and git SHA alongside any new numbers.
