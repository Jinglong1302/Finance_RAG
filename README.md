# Finance RAG

Enterprise-grade financial intelligence engine for querying SEC 10-K/10-Q filings using **Corrective RAG (CRAG)**, hybrid dense-sparse search, cross-encoder reranking, and automated LLM-as-a-judge evaluation.

## Features

- **HTML-first SEC Filing Parsing** — Directly parses iXBRL/HTML from EDGAR with table-aware extraction
- **Hybrid Search** — BGE-M3 dense + sparse (lexical-weight) embeddings with RRF fusion
- **Cross-Encoder Reranking** — BAAI/bge-reranker-base (overridable via `RERANKER_MODEL` env var)
- **Corrective RAG (CRAG)** — LangGraph state machine with ≤2 rewrite cycles and explicit refusal
- **Pooled-Context Sufficiency Grading** — Single GPT-4o call on concatenated top-k context (Part B)
- **Parent-Child Chunk Hierarchy** — Retrieval on child chunks; parent text injected for context
- **Hallucination Guard** — Post-generation GPT-4o faithfulness check; triggers rewrite on fail
- **Citation Mapping** — Structured citation objects with company/section/year metadata per answer

## Quick Start

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- [Docker](https://docs.docker.com/get-docker/) (for local Qdrant)
- OpenAI API key

### Setup

```bash
# Clone and install
cd Finance_RAG
poetry install

# Copy and configure environment
cp .env.example .env
# Edit .env with your OpenAI API key and email

# Start Qdrant
docker compose up -d
```

### Ingest SEC Filings

```bash
poetry run python scripts/ingest.py --tickers AAPL MSFT NVDA AMZN --filing-type 10-K --limit 3
```

### Interactive Web Dashboard (UI)

Launch the visual inspection & monitoring dashboard:

```bash
poetry run python scripts/serve.py
```
Open [http://localhost:8000](http://localhost:8000) to inspect every CRAG step, view interactive citations, and inspect reranked evidence chunks in real time.

### CLI Query

```bash
poetry run python scripts/query.py "What was Apple's total revenue in FY2024?"
```

### Reproduce Evaluations

All eval scripts require Qdrant running and `OPENAI_API_KEY` set. Enable LLM disk cache to avoid re-spending on reruns (temp=0, seed=42 — deterministic).

```bash
# Part A ablations (retrieval + generation baselines, no grader)
python scripts/eval/eval_part_a_consolidated.py

# Full-CRAG holdout-22 baseline (per-chunk grader)
python scripts/eval/eval_crag_holdout22.py

# Part B: pooled sufficiency grader on FB27 (Tier 1 + Tier 2 + OOC20)
python scripts/eval/eval_part_b.py --tier 1
python scripts/eval/eval_part_b.py --tier 2   # also runs OOC20

# AAPL50 with Part B grader (clean held-out)
python scripts/eval/eval_part_b_aapl50.py

# Full-scale baselines (AAPL50 + TAT-QA50 + OOC20 + FB27) on production graph
python scripts/eval/eval_fullscale_baselines.py
```

Results are saved to `results/eval/` (gitignored except `results/PROGRESS.md` and `results/FINAL.md`).
LLM cache stored in `results/llm_cache/` (gitignored). Estimated costs: see `results/FINAL.md`.

## Architecture

```
User Query → Query Decomposition (GPT-4o)
           → Hybrid Search (Qdrant: BGE-M3 dense + sparse, RRF)
           → Cross-Encoder Reranking (bge-reranker-base, top-5)
           → Parent Expansion (child retrieval → parent context injection)
           → Pooled Sufficiency Grading (GPT-4o, 1 call on concatenated top-k)
           → [SUFFICIENT/PARTIAL] → Generation with Citations (GPT-4o)
                                  → Hallucination Guard (GPT-4o)
           → [INSUFFICIENT, cycles < 2] → Query Rewrite → Re-retrieve
           → [INSUFFICIENT, cycles = 2] → Explicit Refusal
```

**Current index**: MMM, BA, KO, NFLX, PFE, AAPL — 10-K filings — 6,971 Qdrant points

**Company and fiscal-year resolution (production)**: `query_decomposer_node` calls GPT-4o with the decomposition prompt; the model extracts `company_ticker` and `fiscal_year` from the query text alone — no fallback injection occurs. The Part B FB27 eval (SHA b630a6b) injected the ticker after decompose; that is NOT the production path.

## Evaluation Results

Final config: v1.0 (SHA b5f9c2e) — Part B pooled-context sufficiency grader, production path (no ticker injection).
Methodology: LLM disk cache (temp=0, seed=42).

### Held-Out Benchmarks (clean, grader never saw these)

| Benchmark | n | Coverage | Prec@Ans | Gate | Result |
|-----------|---|----------|----------|------|--------|
| AAPL50 (custom 10-K QA) | 50 | 84.0% (42/50) | 95.2% (40/42) | cov ≥84%, prec ≥90% | ✅ both pass |
| TAT-QA50 (context-injected)* | 50 | 98.0% (49/50) | 93.9% (46/49) | — | — |
| OOC20 abstention (out-of-corpus) | 20 | — | — | =100% | ✅ 20/20 abstained |

\* TAT-QA50 injects context directly to the generator; grader is not called.

### FinanceBench-27 (FB27 = dev-5 + holdout-22)

| Config | n | answered | correct | coverage | prec@ans |
|--------|---|----------|---------|----------|----------|
| Baseline (per-chunk grader, SHA 94fdd2f) | 27 | 17 | 13 | 62.96% | 76.5% |
| Part B (pooled sufficiency, SHA b630a6b) | 27 | 17 | 15 | 62.96% | **88.2%** |

**Coverage gate (>63%) not met** — 17/27 = 62.96% for both configs.
**Precision improved +11.7pp** (13→15 correct on the same 17 answered questions).

> Caveat: FB27 numbers are measured on two different harnesses (baseline via LangGraph graph without ticker injection; Part B via manual loop with ticker injection) and should not be compared directly. Holdout-22 numbers (Q06-Q27) are tuning-contaminated for Part B — the grader configuration was selected based on these metrics.

### Baseline vs Part B on Holdout-22 (same harness, contaminated)

| Config | n | answered | correct | coverage | prec@ans |
|--------|---|----------|---------|----------|----------|
| Full-CRAG baseline (grader_node) | 22 | 12 | 10 | 54.5% | 83.3% |
| Part B (sufficiency_grader_node) | 22 | 14 | 13 | **63.6%** | **92.9%** |

These numbers are contaminated (used for config selection). Treat as directional, not held-out.

### Retrieval (Holdout-22, overlap threshold 0.65)

| Hit@1 | Hit@5 | Hit@10 | MRR |
|-------|-------|--------|-----|
| 0.68 | **0.95** | 0.95 | 0.78 |

### FB27 Production Path (no ticker injection, v1.0)

The previously reported FB27 Part B numbers (SHA b630a6b) used ticker injection after decompose. The true production path relies on GPT-4o to extract the ticker from the query text.

| Config | n | answered | correct | coverage | prec@ans |
|--------|---|----------|---------|----------|----------|
| Injected (b630a6b, non-production) | 27 | 17 | 15 | 62.96% | 88.2% |
| **Production / no-injection (b5f9c2e)** | **27** | **16** | **15** | **59.3%** | **93.8%** |

Ticker resolution: 27/27 correct on production path. The coverage drop (−3.7pp) is entirely from one question (Q26 PFE) that the hallucination guard now correctly rejects instead of returning a wrong answer.

### Ragas Metrics (v1.0 final config)

Judge model: gpt-4o (ragas 0.2.15). Evaluated on answered subsets only.

| Metric | Baseline† | FB27 no-inj (n=16) | AAPL50 (n=42) |
|--------|----------|--------------------|---------------|
| faithfulness | 0.834 | 0.797 | **0.900** |
| context_precision | 0.892 | 0.746 | 0.758 |
| answer_relevancy | 0.856 | 0.880 | **0.987** |
| context_recall | 0.324 | 0.344 | **0.956** |

† Baseline = per-chunk-grader config (SHA 94fdd2f), measured on the same answered subsets.

### Key Limitations

- **Holdout-22 contamination**: Part B config selected on these 22 questions. AAPL50 is the clean estimate.
- **FB27 gate not met**: 17/27 = 62.96% coverage, below the >63% threshold.
- **Oracle ceiling 77.3%**: Even gold-retrieved context yields 77.3% precision, suggesting 5/22 questions have evaluation-metric or answer-quality issues independent of retrieval.
- **Eval metric narrowness**: `span_match` + `numeric_match` may miss correct paraphrases.
- **n=50 max**: All clean benchmarks cover ≤7 companies (AAPL, MMM, BA, KO, NFLX, PFE, OOC companies). Sector generalization is untested.

## Project Structure

```
├── config/          # Pydantic Settings configuration
├── src/
│   ├── ingestion/   # SEC EDGAR download, HTML parsing, table extraction
│   ├── chunking/    # Section-based chunking with parent-child hierarchy
│   ├── embedding/   # BGE-M3 dense+sparse, Qdrant indexing
│   ├── retrieval/   # Hybrid search, reranking, parent expansion
│   ├── orchestration/  # LangGraph CRAG state machine
│   └── evaluation/  # Ragas metrics, benchmarks, cost tracking
├── scripts/         # CLI entry points
├── notebooks/       # Exploration and analysis
└── tests/           # Unit and integration tests
```

## License

MIT
