# Archived eval scripts

These scripts made results that were later replaced (see `results/archive/INDEX.md`). They are kept so old numbers can be traced, not for new runs. They still run: their project-root line was changed to `Path(__file__).parents[3]` when they moved here on 2026-10-23.

| Script | What it made | Replaced by |
|---|---|---|
| `eval_ablations.py` | `results/eval/ablations_eval.json`: Part A ablations (SHA 94fdd2f) | `scripts/eval/eval_part_a_consolidated.py` |
| `eval_oracle.py` | `results/eval/oracle_eval.json`: oracle-context test (SHA 94fdd2f) | `scripts/eval/eval_part_a_consolidated.py` |
| `run_all_evals.py` | `results/eval/summary_*.json` / `summary_latest.*`: early combined runs (2026-09-26/27) | the per-benchmark scripts in `docs/guides/evaluation.md` |
| `generate_3_way_report.py` | `results/eval/summary_3_pipelines.md`, built from the `summary_*.json` files above | `scripts/eval/eval_part_a_consolidated.py` |

Note: `run_all_evals.py` still calls `scripts/eval/eval_retrieval.py`, `eval_generation_financebench.py`, `eval_generation_tatqa.py` and `eval_abstention.py`, which were not archived.
