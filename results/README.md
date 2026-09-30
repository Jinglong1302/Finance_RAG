# Results Source Map

Each number in `FINAL.md` and `README.md` traces to a specific result file and git commit.

## Canonical Result Files

| File | Evaluation | Git SHA | Key Numbers |
|------|-----------|---------|-------------|
| `eval/fb27_no_injection_20261001_000652.json` | FB27 no-injection (Phase A v1.0) | b5f9c2e | coverage=59.3% (16/27), prec=93.8% (15/16) |
| `eval/ragas_phase_c_20261001_005227.json` | Phase C Ragas | b5f9c2e | FB27 (n=16): faith=0.797, ctx_prec=0.746, ans_rel=0.880, ctx_rec=0.344; AAPL50 (n=42): faith=0.900, ctx_prec=0.758, ans_rel=0.987, ctx_rec=0.956 |
| `eval/part_b_aapl50_20261001_004147.json` | AAPL50 Part B | b5f9c2e | coverage=84.0% (42/50), prec=95.2% (40/42) |
| `eval/crag_holdout22_20260930_154700.json` | Holdout-22 baseline (Cycle 1 manual loop) | b630a6b | baseline: 12/22 answered, 10 correct |
| `eval/part_b_tier1_20260930_161638.json` | Part B Tier1 Q01-Q16 (Cycle 1, shipped) | b630a6b | coverage=68.75%, prec=90.9% |
| `eval/part_b_tier2_20260930_162704.json` | Part B Tier2 Q06-Q27 (Cycle 1, shipped) | b630a6b | holdout-22 part |
| `eval/part_b_ooc_20260930_162704.json` | OOC20 abstention (Cycle 1) | b630a6b | 20/20 abstained |
| `eval/part_a_consolidated_20260929_171614.json` | Part A consolidated (ablations+oracle+dev5) | 78e05dd | dev-5 coverage=100%, oracle prec=77.3% |
| `eval/fullscale_baselines_eval.json` | TAT-QA50 + baseline Ragas reference | 94fdd2f | TAT-QA50: coverage=98%, prec=93.9%; Ragas baseline: faith=0.834, ctx_prec=0.856, ans_rel=0.892, ctx_rec=0.324 |

## Numbers with No Direct Single Source File

The **Holdout-22 Part B combined numbers** (14/22 answered, 13 correct, coverage=63.6%, prec=92.9%)
are derived by combining two files: `part_b_tier2_20260930_162704.json` (Q06-Q27) with
`part_b_tier1_20260930_161638.json` (Q01-Q16 overlap). The merged holdout-22 metrics are
reported in `docs/PROGRESS.md` and cited in `FINAL.md`.

The **injected Part B FB27 number** (17/27 answered, 15 correct, coverage=62.96%, prec=88.2%)
is derived from the same tier1+tier2 Cycle 1 files with ticker injection applied in the harness.

## Archived / Superseded Files

Superseded result files have been moved to `results/archive/`. See `results/archive/INDEX.md`
for a full list with reasons for archiving.

## Notes

- All canonical numbers use LLM disk cache (temp=0, seed=42) for reproducibility.
- Ragas metrics (Phase C) bypass the pipeline disk cache — real API spend at eval time.
- Git SHAs refer to commits on the `master` branch (merged from `eval/part-a-repair`).
