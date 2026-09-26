"""CLI script for interactive financial queries.

Runs queries through the full CRAG pipeline and displays results
with rich formatting.

Usage:
    poetry run python scripts/query.py "What was Apple's total revenue in FY2024?"
    poetry run python scripts/query.py "..." --report          # detailed pipeline trace
    poetry run python scripts/query.py --interactive --report  # interactive + trace
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from config.settings import get_settings
from src.embedding.embedder import BGEEmbedder
from src.orchestration.graph import build_crag_graph, run_query
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.parent_expander import ParentExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.logging import setup_logging
from src.utils.reporter import save_query_report

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query SEC filings")
    parser.add_argument("query", nargs="?", help="Financial question to answer")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--report", "-r", action="store_true",
                        help="Print detailed pipeline trace after each query")
    parser.add_argument("--naive", "--baseline", action="store_true", dest="naive",
                        help="Run Naive RAG baseline instead of CRAG")
    parser.add_argument("--ticker", default=None, help="Optional ticker filter (e.g. AAPL, MMM, BA, KO, NFLX, PFE)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve for Naive RAG (default: 5)")
    parser.add_argument("--log-level", default="WARNING", help="Log level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    embedder = BGEEmbedder(
        model_name=settings.embedding_model,
        use_fp16=settings.use_fp16,
    )

    if args.naive:
        console.print("[bold yellow]Running in Naive RAG Mode (Dense-only, no reranker, no loops)[/bold yellow]\n")
        from src.evaluation.baseline import NaiveRAG
        from scripts.query_baseline import run_baseline_query, run_interactive as run_baseline_interactive

        baseline = NaiveRAG(
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name=settings.qdrant_collection,
            openai_model=settings.openai_model,
            top_k=args.top_k,
        )
        if args.query:
            run_baseline_query(baseline, args.query, ticker=args.ticker)
        elif args.interactive:
            run_baseline_interactive(baseline)
        else:
            parser.print_help()
        return

    # Initialize components for CRAG
    console.print("[bold cyan]Initializing Finance RAG...[/bold cyan]")

    searcher = HybridSearcher(
        client=qdrant_client,
        embedder=embedder,
        collection_name=settings.qdrant_collection,
    )
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
    )
    expander = ParentExpander(
        client=qdrant_client,
        collection_name=settings.qdrant_collection,
    )

    # Build CRAG graph
    graph = build_crag_graph(
        searcher=searcher,
        reranker=reranker,
        expander=expander,
        qdrant_client=qdrant_client,
        collection_name=settings.qdrant_collection,
    )

    console.print("[bold green]✓ Finance RAG initialized[/bold green]\n")

    if args.query:
        _run_single_query(graph, args.query, args.report)
    elif args.interactive:
        _run_interactive(graph, args.report)
    else:
        parser.print_help()


def _run_single_query(graph: Any, query: str, report: bool = False) -> None:
    """Run a single query and display results."""
    console.print(Panel(query, title="[bold]Query[/bold]", border_style="blue"))

    with console.status("[bold cyan]Searching SEC filings..."):
        result = run_query(graph, query)

    # Save comprehensive report to disk
    try:
        payload = dict(result)
        payload["query"] = query
        report_info = save_query_report(payload)
    except Exception:
        report_info = None

    _display_result(result)
    if report_info:
        console.print(f"[dim]📁 Analysis report saved: {report_info['md_path']}[/dim]")
    if report:
        _display_pipeline_report(result)


def _run_interactive(graph: Any, report: bool = False) -> None:
    """Run interactive query loop."""
    console.print("[bold]Interactive mode. Type 'quit' to exit.[/bold]\n")

    while True:
        query = console.input("[bold cyan]Ask a question: [/bold cyan]")
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query.strip():
            continue

        with console.status("[bold cyan]Searching SEC filings..."):
            result = run_query(graph, query)

        try:
            payload = dict(result)
            payload["query"] = query
            report_info = save_query_report(payload)
        except Exception:
            report_info = None

        _display_result(result)
        if report_info:
            console.print(f"[dim]📁 Analysis report saved: {report_info['md_path']}[/dim]")
        if report:
            _display_pipeline_report(result)
        console.print()


def _display_result(result: dict) -> None:
    """Display query result with rich formatting."""
    answer = result.get("final_answer") or result.get("generation", "No answer generated")
    confidence = result.get("confidence", "unknown")
    cycles = result.get("cycle_count", 0)
    cost = result.get("cost_accumulated", 0.0)

    # Confidence badge
    conf_colors = {
        "high": "green",
        "medium": "yellow",
        "low": "red",
        "insufficient": "red",
    }
    conf_color = conf_colors.get(confidence, "white")

    # Answer panel
    console.print(
        Panel(
            Markdown(answer),
            title=f"[bold]Answer [{conf_color}]{confidence.upper()} CONFIDENCE[/{conf_color}][/bold]",
            border_style="green" if confidence in ("high", "medium") else "red",
        )
    )

    # Metadata table
    meta_table = Table(show_header=False, box=None)
    meta_table.add_row("Latency:", f"{result.get('latency_s', 0.0):.2f}s")
    meta_table.add_row("CRAG Cycles:", str(cycles))
    meta_table.add_row("API Cost:", f"${cost:.4f}")
    meta_table.add_row("Citations:", str(len(result.get("citations", []))))
    console.print(meta_table)

    # Citations
    citations = result.get("citations", [])
    if citations:
        console.print("\n[bold]Sources:[/bold]")
        for cit in citations:
            console.print(
                f"  [{cit.get('index', '?')}] "
                f"{cit.get('company', '')} {cit.get('filing_type', '')} "
                f"(FY{cit.get('fiscal_year', '')}), "
                f"{cit.get('section', '')}"
            )


def _display_pipeline_report(result: dict) -> None:
    """Print a detailed per-node pipeline trace report."""
    trace: list[dict] = result.get("pipeline_trace") or []
    if not trace:
        console.print("[dim]No pipeline trace available.[/dim]")
        return

    console.print()
    console.print(Rule("[bold cyan]Pipeline Trace Report[/bold cyan]"))

    for entry in trace:
        node = entry.get("node", "?")

        # ── Query Decomposer ─────────────────────────────────────────────────
        if node == "query_decomposer":
            console.print(f"\n[bold yellow]1. Query Decomposer[/bold yellow]")
            t = Table(show_header=True, header_style="bold", box=box.SIMPLE)
            t.add_column("Field", style="dim", width=22)
            t.add_column("Value")
            t.add_row("Query type", entry.get("query_type", "?"))
            t.add_row("Sub-queries", str(len(entry.get("sub_queries", []))))
            t.add_row("Filters", str(entry.get("filters", {})))
            t.add_row("Prompt tokens", str(entry.get("prompt_tokens", 0)))
            t.add_row("Cached tokens", str(entry.get("cached_tokens", 0)))
            t.add_row("Completion tokens", str(entry.get("completion_tokens", 0)))
            t.add_row("Cost", f"${entry.get('cost_usd', 0):.5f}")
            console.print(t)
            for i, sq in enumerate(entry.get("sub_queries", []), 1):
                console.print(f"  [{i}] {sq}")

        # ── Retriever ────────────────────────────────────────────────────────
        elif node == "retriever":
            console.print(f"\n[bold yellow]2. Hybrid Retriever[/bold yellow]")
            t = Table(show_header=True, header_style="bold", box=box.SIMPLE)
            t.add_column("Field", style="dim", width=22)
            t.add_column("Value")
            t.add_row("Queries issued", str(len(entry.get("queries", []))))
            t.add_row("Raw candidates", str(entry.get("raw_results", 0)))
            t.add_row("After dedup", str(entry.get("unique_results", 0)))
            top = entry.get("top_scores", [])
            t.add_row("Top-5 RRF scores", ", ".join(str(s) for s in top))
            console.print(t)

        # ── Reranker ─────────────────────────────────────────────────────────
        elif node == "reranker":
            console.print(f"\n[bold yellow]3. Cross-Encoder Reranker[/bold yellow]")
            # Summary table — scores and metadata only (no truncated preview)
            t = Table(show_header=True, header_style="bold", box=box.SIMPLE)
            t.add_column("Rank", justify="right", width=5)
            t.add_column("Score", justify="right", width=8)
            t.add_column("Company", width=8)
            t.add_column("FY", width=6)
            t.add_column("Section", width=45)
            for chunk in entry.get("chunks", []):
                score_val = chunk.get("rerank_score", 0)
                t.add_row(
                    str(chunk.get("rank", "?")),
                    f"{score_val:.3f}",
                    str(chunk.get("company", "?")),
                    str(chunk.get("fiscal_year", "?")),
                    str(chunk.get("section", ""))[:45],
                )
            console.print(t)
            console.print(
                f"  [dim]Candidates in: {entry.get('candidates_in',0)} → "
                f"Top-K selected: {entry.get('top_k',0)}[/dim]"
            )
            # Full-text panels for each retrieved chunk
            console.print("\n  [bold dim]Retrieved Chunk Contents:[/bold dim]")
            for chunk in entry.get("chunks", []):
                rank = chunk.get("rank", "?")
                score = chunk.get("rerank_score", 0)
                section = chunk.get("section", "?")
                fy = chunk.get("fiscal_year", "?")
                company = chunk.get("company", "?")
                full_text = chunk.get("text_full") or chunk.get("text_preview", "(no text)")
                title = (f"[bold]Rank {rank}[/bold]  score={score:.3f}  "
                         f"{company} FY{fy}  |  {section}")
                console.print(
                    Panel(
                        full_text,
                        title=title,
                        border_style="dim",
                        expand=True,
                    )
                )

        # ── Grader ───────────────────────────────────────────────────────────
        elif node == "grader":
            console.print(f"\n[bold yellow]4. Relevance Grader[/bold yellow]")
            t = Table(show_header=True, header_style="bold", box=box.SIMPLE)
            t.add_column("Field", style="dim", width=22)
            t.add_column("Value")
            t.add_row("Relevant", f"[green]{entry.get('relevant', 0)}[/green]")
            t.add_row("Partially relevant", f"[yellow]{entry.get('partial', 0)}[/yellow]")
            t.add_row("Irrelevant", f"[red]{entry.get('irrelevant', 0)}[/red]")
            t.add_row("Confidence", entry.get("confidence", "?"))
            t.add_row("Action", entry.get("action", "?"))
            t.add_row("Prompt tokens", str(entry.get("prompt_tokens", 0)))
            t.add_row("Cached tokens", str(entry.get("cached_tokens", 0)))
            t.add_row("Completion tokens", str(entry.get("completion_tokens", 0)))
            t.add_row("Cost", f"${entry.get('cost_usd', 0):.5f}")
            console.print(t)
            # Per-chunk grade detail
            grades = entry.get("grades", [])
            if grades:
                gt = Table(show_header=True, header_style="bold dim", box=box.SIMPLE)
                gt.add_column("#", justify="right", width=3)
                gt.add_column("Relevance", width=18)
                gt.add_column("Reason")
                for g in grades:
                    rel = g.get("relevance", "?")
                    color = {"relevant": "green", "partially_relevant": "yellow",
                             "irrelevant": "red"}.get(rel, "white")
                    gt.add_row(
                        str(g.get("chunk_index", "?")),
                        f"[{color}]{rel}[/{color}]",
                        str(g.get("reason", ""))[:80],
                    )
                console.print(gt)

        # ── Generator ────────────────────────────────────────────────────────
        elif node == "generator":
            console.print(f"\n[bold yellow]5. Answer Generator[/bold yellow]")
            t = Table(show_header=True, header_style="bold", box=box.SIMPLE)
            t.add_column("Field", style="dim", width=22)
            t.add_column("Value")
            t.add_row("Answer length (chars)", str(entry.get("answer_length", 0)))
            t.add_row("Citations", str(entry.get("citations_count", 0)))
            t.add_row("Prompt tokens", str(entry.get("prompt_tokens", 0)))
            t.add_row("Cached tokens", str(entry.get("cached_tokens", 0)))
            t.add_row("Completion tokens", str(entry.get("completion_tokens", 0)))
            t.add_row("Cost", f"${entry.get('cost_usd', 0):.5f}")
            console.print(t)

        # ── Hallucination Guard ───────────────────────────────────────────────
        elif node == "hallucination_guard":
            console.print(f"\n[bold yellow]6. Hallucination Guard[/bold yellow]")
            result_val = entry.get("result", "?")
            color = "green" if result_val == "pass" else "red"
            t = Table(show_header=True, header_style="bold", box=box.SIMPLE)
            t.add_column("Field", style="dim", width=22)
            t.add_column("Value")
            t.add_row("Result", f"[{color}]{result_val.upper()}[/{color}]")
            issues = entry.get("issues", [])
            t.add_row("Issues", str(len(issues)))
            t.add_row("Prompt tokens", str(entry.get("prompt_tokens", 0)))
            t.add_row("Cached tokens", str(entry.get("cached_tokens", 0)))
            t.add_row("Completion tokens", str(entry.get("completion_tokens", 0)))
            t.add_row("Cost", f"${entry.get('cost_usd', 0):.5f}")
            console.print(t)
            for issue in issues:
                console.print(f"  [red]- {issue}[/red]")

    # ── Cost Summary ─────────────────────────────────────────────────────────
    llm_nodes = [e for e in trace if "cost_usd" in e]
    if llm_nodes:
        console.print()
        console.print(Rule("[bold]Cost Summary[/bold]"))
        cost_t = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        cost_t.add_column("Node", width=20)
        cost_t.add_column("Prompt", justify="right", width=8)
        cost_t.add_column("Cached", justify="right", width=8)
        cost_t.add_column("Output", justify="right", width=8)
        cost_t.add_column("Cost (USD)", justify="right", width=12)
        total_cost = 0.0
        total_prompt = total_cached = total_output = 0
        for e in llm_nodes:
            cost_t.add_row(
                e.get("node", "?"),
                str(e.get("prompt_tokens", 0)),
                str(e.get("cached_tokens", 0)),
                str(e.get("completion_tokens", 0)),
                f"${e.get('cost_usd', 0):.5f}",
            )
            total_cost += e.get("cost_usd", 0)
            total_prompt += e.get("prompt_tokens", 0)
            total_cached += e.get("cached_tokens", 0)
            total_output += e.get("completion_tokens", 0)
        cost_t.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_prompt}[/bold]",
            f"[bold]{total_cached}[/bold]",
            f"[bold]{total_output}[/bold]",
            f"[bold]${total_cost:.5f}[/bold]",
        )
        console.print(cost_t)
        if total_cached > 0:
            saving = total_cached * (2.50 - 1.25) / 1_000_000
            console.print(
                f"  [dim]Cache hit: {total_cached} tokens saved ~${saving:.5f} "
                f"vs list price[/dim]"
            )

    console.print(Rule())


if __name__ == "__main__":
    main()
