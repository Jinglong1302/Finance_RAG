"""CLI script for running queries against the Naive RAG baseline.

Pipeline Architecture:
- Fixed-size chunking (~512 tokens) reusing already-ingested docs in Qdrant ('sec_filings')
- Single dense retrieval (BGE-M3 1024-dim, no sparse/BM25, no RRF)
- No cross-encoder reranker
- No query decomposition or sub-query rewriting
- One-shot direct LLM generation (no CRAG loops, no grading, no hallucination guard)

Usage:
    poetry run python scripts/query_naive.py "What was Apple's total revenue in FY2024?"
    poetry run python scripts/query_naive.py "What was 3M's CAPEX in FY2018?" --ticker MMM
    poetry run python scripts/query_naive.py --interactive
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.query_baseline import main

if __name__ == "__main__":
    main()
