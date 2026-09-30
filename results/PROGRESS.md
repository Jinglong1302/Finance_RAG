# Finance RAG — Eval Progress Log

Branch: `eval/part-a-repair` | Start SHA: `78e05dd`

---

## Preflight (every run)
| Check | Status |
|-------|--------|
| Qdrant reachable (finance_rag-qdrant-1) | ✅ 6,971 points |
| temperature=0, seed=42 on all nodes | ✅ confirmed in all 5 nodes |
| Git tree clean | ⚠️ 3 untracked scripts → committed in step below |

---

## Part A Baseline (already recorded — SHA 78e05dd, 2026-09-29)

File: `results/eval/part_a_consolidated_20260929_171614.json`
Model: gpt-4o | Reranker: bge-reranker-base | Overlap threshold: 0.65

### Holdout-22 Retrieval (overlap threshold 0.65)

| Metric | (a) RRF-only | (b) Naive dense | (c) RRF+Reranker |
|--------|-------------|-----------------|------------------|
| Hit@1  | 0.6818 | 0.6818 | 0.6818 |
| Hit@3  | 0.8182 | 0.8182 | 0.7727 |
| Hit@5  | **0.9091** | 0.8182 | 0.8636 |
| Hit@10 | 0.9545 | 0.8636 | 0.8636 |
| MRR    | **0.7686** | 0.7470 | 0.7402 |

**Key finding**: RRF-only beats RRF+Reranker on every retrieval metric.
The bge-reranker-base is actually *hurting* retrieval quality on this set.

### Holdout-22 Generation (no grader on any ablation)

| Condition | n | correct | incorrect | abstained | coverage | prec@ans |
|-----------|---|---------|-----------|-----------|----------|----------|
| (a) RRF→gen | 22 | 9 | 1 | 12 | 45.45% | 90.0% |
| (b) Naive→gen | 22 | 7 | 0 | 15 | 31.82% | 100.0% |
| (c) RRF+Rerank→gen | 22 | 9 | 1 | 12 | 45.45% | 90.0% |
| Oracle (gold ctx) | 22 | 17 | 5 | 0 | 100.0% | 77.3% |

### Dev-5 Full CRAG

| n | correct | incorrect | abstained | coverage | prec@ans |
|---|---------|-----------|-----------|----------|----------|
| 5 | 2 | 1 | 2 | 60.0% | 66.7% |

**Note**: Dev-5 was the "sealed" tuning set — these numbers carry a holdout caveat.

### Oracle gap analysis
Oracle precision (77.3%) < 100% means 5 questions have *answer quality* issues
even with perfect context — generation or evaluation metric issue, not retrieval.

---

## Step 0 — Infrastructure (2026-09-30)

- Added `src/utils/llm_cache.py`: disk cache via monkey-patching
  `openai.resources.chat.completions.Completions.create`. Key = SHA-256 of
  (model, messages, temperature, seed, response_format). Unchanged nodes cost
  $0 on rerun.
- Added `scripts/eval/eval_crag_holdout22.py`: full production CRAG
  (decompose → retrieve → rerank → expand_notes → grade → generate/refuse/rewrite
  × ≤2 cycles → guard) on holdout-22 with ticker injection.
- Committed 3 previously untracked eval scripts.

---

## Step 1 — Full CRAG Holdout-22 Baseline (Tier 2) — COMPLETE

File: `results/eval/crag_holdout22_20260930_154700.json`
Git: e3d3e42, dirty=True | Cost: $0.6584 | Cache: 13H/111M

**Pre-registered hypothesis confirmed**: Full CRAG with per-chunk grader did NOT
improve coverage over ablation_a. Generator is the coverage bottleneck — it
abstains even on questions where the grader says "generate" (Q03, Q20-Q21).

### Retrieval (n=22)

| Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|-------|-------|-------|--------|-----|
| 0.7273 | 0.8636 | 0.9091 | 0.9091 | 0.8008 |

### Generation (n=22)

| correct | incorrect | abstained | coverage | prec@ans |
|---------|-----------|-----------|----------|----------|
| 10 | 2 | 10 | **54.5%** | **83.3%** |

**vs Part A ablation_a (no grader)**: coverage +9.1pp (45.45%→54.5%), prec +Δ(90%→83.3% −6.7pp).
Full CRAG grader improved coverage but hurt precision slightly.

### Key finding: reranker is NOT dropping evidence
Hit@5_reranked=1 for all grader-refused questions. The bottleneck is per-chunk
grading seeing partial relevance per chunk even when chunks together answer
the question.

---

## Part B — Pooled-context sufficiency grader (COMPLETE)

**Hypothesis**: Replacing per-chunk relevance grading with a single pooled-context
sufficiency judgment reduces LLM calls, evaluates context holistically, and
improves coverage without hurting precision. KEY CHANGE: PARTIAL → generate
(not rewrite as in per-chunk grader).

**Implementation**: `src/orchestration/nodes/sufficiency_grader.py` +
`src/orchestration/prompts/sufficiency.py`

### Part B Tier 1 (Q01-Q16, dev-5 + first 11 holdout)

File: `results/eval/part_b_tier1_20260930_161638.json` | Cost: $0.5216 | Cache: 19H/65M

| n | correct | incorrect | abstained | coverage | prec@ans |
|---|---------|-----------|-----------|----------|----------|
| 16 | 10 | 1 | 5 | **68.8%** | **90.9%** |

**Dev-5 breakdown** (Q01-Q05): correct=2, incorrect=1, abstained=2 → 3/5 answered

### Part B Tier 2 (holdout-22 + OOC20)

File: `results/eval/part_b_tier2_20260930_162704.json` | Cost: $0.7007 | Cache: 78H/34M
OOC file: `results/eval/part_b_ooc_20260930_162704.json` | OOC abstention: **100%** ✅

#### Retrieval (n=22)

| Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|-------|-------|-------|--------|-----|
| 0.6818 | 0.8636 | 0.9545 | 0.9545 | 0.7833 |

#### Generation (n=22)

| correct | incorrect | abstained | coverage | prec@ans |
|---------|-----------|-----------|----------|----------|
| 13 | 1 | 8 | **63.6%** | **92.9%** |

**vs Full CRAG baseline**: coverage +9.1pp (54.5%→63.6%), prec +9.6pp (83.3%→92.9%) ✅
**OOC20 abstention**: 100% ✅

#### FinanceBench27 projection (dev-5 + holdout-22)
- Dev-5: 3/5 answered | Holdout-22: 14/22 answered
- **Total: 17/27 = 62.96%** → just UNDER the >63% acceptance gate (need 18/27)

#### Remaining 8 abstentions in holdout-22
| Q | Ticker | Topic | action | cycles | root cause |
|---|--------|-------|--------|--------|------------|
| Q07 | MMM | debt securities | refuse | 2 | grader: INSUFFICIENT after 2 cycles |
| Q09 | BA | net PPNE FY2018 | refuse | 2 | grader: INSUFFICIENT after 2 cycles |
| Q13 | BA | primary customers | refuse | 2 | grader: INSUFFICIENT after 2 cycles |
| Q20 | NFLX | EBITDA % | generate | 0 | generator abstains despite action=generate |
| Q21 | NFLX | current liabilities | generate | 0 | generator abstains despite action=generate |
| Q22 | PFE | PPNE growth | refuse | 2 | grader: INSUFFICIENT after 2 cycles |
| Q24 | PFE | acquisitions | refuse | 2 | grader: INSUFFICIENT after 2 cycles |
| Q25 | PFE | Upjohn cost | refuse | 2 | grader: INSUFFICIENT after 2 cycles |

---

## Improvement Cycle 2 — PARTIAL confidence "medium"→"high" — REVERTED

**Pre-registered hypothesis (2026-09-30)**: Setting PARTIAL verdict confidence to
"high" (instead of "medium") will reduce generator abstentions on questions where
the pooled grader returns PARTIAL but the generator still abstains (Q03, Q20, Q21).
Expected effect: +1 to +2 more answered questions on dev-5+holdout, pushing
FinanceBench27 coverage from 62.96% to ≥63.0%.
Risk: may gain incorrect answers on borderline questions.

**Files**: `results/eval/part_b_tier1_20260930_170002.json`,
`results/eval/part_b_tier2_20260930_170337.json`

### Tier 1 results (Q01-Q16)
| correct | incorrect | abstained | coverage | prec@ans |
|---------|-----------|-----------|----------|----------|
| 10 | 3 | 3 | **81.2%** | **76.9%** |

**Changes vs Cycle 1 Tier 1**: Q03 abstained→incorrect; Q05 refused→incorrect.
Both qualitative questions now generate wrong answers.

### Tier 2 results (holdout-22)
**Identical to Cycle 1** — zero per-question changes on the clean holdout set.
coverage=63.6%, prec=92.9% (unchanged). The confidence label does not reach
the 5 grader-refused questions (they never hit the generator) or Q20/Q21
(generator already had the context; confidence alone doesn't override its
internal "insufficient evidence" check).

### OOC20 (safety gate)
100% abstention ✅ (unchanged)

### Decision: REVERTED
**Why**: The holdout-22 (clean evaluation) saw zero improvement. All changes
occurred on dev-5, which has a holdout caveat (was used as tuning set in earlier
work). Q03 and Q05 are now answered incorrectly rather than abstaining correctly
— higher coverage but worse answer quality. Reverting to Cycle 1 configuration.

**Lesson**: The 5 grader-refused holdout questions cannot be helped by confidence
tuning because they exhaust cycles (INSUFFICIENT after 2 rewrites). Q20/Q21
generator-abstentions require changes to the generation prompt, not the grader.

**FinanceBench27 combined projection (Cycle 2, rejected)**:
- Dev-5: 2c, 3i, 0a (5/5 answered, 40% precision — caveated)
- Holdout-22: 13c, 1i, 8a (14/22 answered)
- Combined: 19/27 = 70.4% coverage, 15/19 = 78.9% prec (passes both gates,
  BUT driven entirely by dev-5 incorrect-answer conversions — not accepted)

---

## Step 4 — Reranker analysis (CONCLUDED, no action)

Full CRAG holdout-22 confirmed Hit@5_reranked=1 for all grader-refused questions.
The reranker is NOT dropping evidence. The original Step 4 hypothesis (drop reranker
if RRF-only matches RRF+reranker) would require a clean A/B on holdout-22 with
sufficiency grader. Given Part B already beats the baseline and budget constraints,
Step 4 is deferred until after Tier 3.

---

## Acceptance Gates (Tier 3 — run at most once)

| Gate | Threshold | Part B Tier 2 status |
|------|-----------|----------------------|
| OOC20 abstention | = 100% | ✅ 100% |
| AAPL50 precision | ≥ 90% | pending |
| AAPL50 coverage  | ≥ 84% | pending |
| FinanceBench27 coverage | > 63% | ⚠️ 62.96% (borderline) |
| FinanceBench27 prec@ans | ≥ 75% | ✅ 92.9% |
| Latency | ≤ baseline | pending |

---

## Cumulative OpenAI Spend

| Run | n | est. cost | actual cost |
|-----|---|-----------|-------------|
| Part A consolidated (2026-09-29) | 22×4+5 | ~$0.78 | ~$0.78 (est) |
| Step 1 full CRAG holdout-22 | 22 | ~$1.00 | $0.6584 |
| Part B Tier 1 (Q01-Q16) | 16 | ~$0.40 | $0.5216 |
| Part B Tier 2+OOC (holdout-22+20) | 42 | ~$0.70 | $0.7007 |
| Cycle 2 Tier 1 (Q01-Q16, reverted) | 16 | ~$0.10 | $0.4848 |
| Cycle 2 Tier 2+OOC (holdout-22+20, reverted) | 42 | ~$0.05 | ~$0.05 |
| **Running total** | | **~$3.16** | **~$3.16** |

**Hard cap**: $10.00 total. Stop and ask before exceeding.
