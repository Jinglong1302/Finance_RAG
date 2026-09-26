"""CLI script for running queries against the Naive RAG baseline.

Pipeline Architecture:
- Fixed-size chunking (~512 tokens) reusing already-ingested docs in Qdrant ('sec_filings')
- Single dense retrieval (BGE-M3 1024-dim, no sparse/BM25, no RRF)
- No cross-encoder reranker
- No query decomposition or sub-query rewriting
- One-shot direct LLM generation (no CRAG loops, no grading, no hallucination guard)

Usage:
    poetry run python scripts/query_naive.py "What was Apple's total revenue in FY2024?"
    poetry run python scripts/query_baseline.py "What was 3M's CAPEX in FY2018?" --ticker MMM
    poetry run python scripts/query_naive.py --interactive
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import get_settings
from src.embedding.embedder import BGEEmbedder
from src.evaluation.baseline import NaiveRAG
from src.utils.logging import setup_logging

console = Console()


def run_baseline_query(baseline: NaiveRAG, query: str, ticker: str | None = None) -> None:
    console.print(Panel(query, title="[bold]Query (Naive RAG)[/bold]", border_style="yellow"))

    with console.status("[bold yellow]Retrieving via single dense vector search..."):
        res = baseline.run(query, ticker=ticker)

    console.print(Panel(res["answer"], title="[bold green]Naive RAG Answer (One-Shot)[/bold green]", border_style="green"))

    # Display retrieved chunks
    chunks = res.get("chunks", [])
    table = Table(title=f"Retrieved Chunks (Dense-Only Top-{len(chunks)})")
    table.add_column("Rank", justify="right", width=5)
    table.add_column("Ticker", style="cyan", width=8)
    table.add_column("FY", justify="right", width=6)
    table.add_column("Page", justify="right", width=6)
    table.add_column("Score", justify="right", style="green", width=8)
    table.add_column("Text Preview", style="dim")

    for i, c in enumerate(chunks):
        meta = c.get("metadata", {})
        table.add_row(
            str(i + 1),
            str(meta.get("company_ticker", "N/A")),
            str(meta.get("fiscal_year", "N/A")),
            str(meta.get("page_number", "N/A")),
            f"{c.get('score', 0.0):.4f}",
            c.get("text", "")[:100].replace("\n", " ") + "...",
        )

    console.print(table)
    tokens = res.get("token_detail", {})
    console.print(
        f"[dim]Latency: {res['latency_s']}s | Cost: ${res['cost_usd']:.5f} | "
        f"Prompt tokens: {tokens.get('prompt_tokens', 0)} | "
        f"Completion tokens: {tokens.get('completion_tokens', 0)}[/dim]\n"
    )


def run_interactive(baseline: NaiveRAG) -> None:
    console.print("[bold]Interactive Naive Baseline mode. Type 'quit' to exit.[/bold]\n")
    while True:
        query = console.input("[bold yellow]Ask a baseline question: [/bold yellow]")
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query.strip():
            continue
        run_baseline_query(baseline, query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query SEC filings using stripped-down Naive RAG baseline")
    parser.add_argument("query", nargs="?", help="Financial question to answer")
    parser.add_argument("--ticker", default=None, help="Optional company ticker filter (e.g. AAPL, MMM, BA)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--log-level", default="WARNING", help="Log level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    console.print("[bold cyan]Initializing Stripped-Down Baseline Pipeline...[/bold cyan]")
    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = BGEEmbedder(model_name=settings.embedding_model, use_fp16=settings.use_fp16)

    baseline = NaiveRAG(
        qdrant_client=qdrant,
        embedder=embedder,
        collection_name=settings.qdrant_collection,
        openai_model=settings.openai_model,
        top_k=args.top_k,
    )
    console.print("[bold green]✓ Baseline initialized (dense-only, no reranker, no decomposition, one-shot)[/bold green]\n")

    if args.query:
        run_baseline_query(baseline, args.query, ticker=args.ticker)
    elif args.interactive:
        run_interactive(baseline)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
