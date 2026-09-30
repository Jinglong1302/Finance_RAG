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

## Step 1 — Full CRAG Holdout-22 Baseline (Tier 2) — PENDING

**Pre-registered hypothesis**: Full CRAG with grader will NOT improve coverage
over ablation_a because the generator itself is the bottleneck (it abstains even
when the grader says "generate"), but may improve precision by filtering bad
context before generation.

**Estimated cost**: ~$1.00 worst-case (no cache hits), ~$0.60 typical.
**Cumulative estimated spend**: ~$0.78 (Part A already spent) + ~$1.00 = ~$1.78

*Results will be filled in after the run.*

---

## Part B Plan — Pooled-context sufficiency grader

**Hypothesis**: Replacing per-chunk relevance grading with a single pooled-
context sufficiency judgment (one LLM call on the concatenated top-k vs the
original question) reduces LLM calls per query, uses context holistically
rather than chunk-by-chunk, and may improve coverage without hurting precision.

**References consulted before implementation**:
- LangGraph CRAG example (official corrective RAG tutorial)
- Google Research "Sufficient Context" autorater (EMNLP 2024-style)

**Tier 0 test**: replay on the 22 stored contexts (no new API calls).
**Tier 1 test**: dev-5 + dev-11 = 16 questions.
**Tier 2 test**: full holdout-22 + OOC20 safety check.

---

## Acceptance Gates (Tier 3 — run at most once)

| Gate | Threshold |
|------|-----------|
| OOC20 abstention | = 100% |
| AAPL50 precision | ≥ 90% |
| AAPL50 coverage  | ≥ 84% |
| FinanceBench27 coverage | > 63% |
| FinanceBench27 prec@ans | ≥ 75% |
| Latency | ≤ baseline |

---

## Cumulative OpenAI Spend

| Run | n | est. cost | actual cost |
|-----|---|-----------|-------------|
| Part A consolidated (2026-09-29) | 22×4+5 | ~$0.78 | not tracked |
| Step 1 full CRAG holdout-22 | 22 | ~$1.00 | pending |
| **Running total** | | **~$1.78** | |

**Hard cap**: $10.00 total. Stop and ask before exceeding.
