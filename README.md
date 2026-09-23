# Finance RAG

Enterprise-grade financial intelligence engine for querying SEC 10-K/10-Q filings using **Corrective RAG (CRAG)**, hybrid dense-sparse search, cross-encoder reranking, and automated LLM-as-a-judge evaluation.

## Features

- **HTML-first SEC Filing Parsing** — Directly parses iXBRL/HTML from EDGAR with table-aware extraction
- **Hybrid Search** — BGE-M3 dense + sparse embeddings with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — BGE-reranker-large for precision refinement
- **Corrective RAG (CRAG)** — Self-correcting retrieval with LangGraph state machine
- **Zero-Hallucination Design** — Ternary relevance grading, faithfulness guardrails, and explicit refusal
- **Financial Table Intelligence** — Parent-child chunk hierarchy with header injection
- **Automated Evaluation** — Ragas metrics (Faithfulness, Context Precision) against FinanceBench & TAT-QA

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

### Evaluate

```bash
poetry run python scripts/evaluate.py --dataset financebench --split dev
```

## Architecture

```
User Query → Query Decomposition (GPT-4o)
           → Hybrid Search (Qdrant Dense + Sparse via RRF)
           → Cross-Encoder Reranking (BGE-reranker-large)
           → CRAG Relevance Grading (GPT-4o)
           → [Context Sufficient] → Generation with Citations
           → [Insufficient] → Query Rewrite → Re-retrieve (max 2 cycles)
           → Hallucination Guardrail Check
           → Final Answer + Document Citations
```

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
