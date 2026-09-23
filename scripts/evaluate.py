"""CLI script for running evaluation benchmarks.

Usage:
    poetry run python scripts/evaluate.py --dataset financebench --split dev
    poetry run python scripts/evaluate.py --dataset tatqa --split eval --output results/
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from src.evaluation.benchmarks import load_financebench, load_tatqa
from src.utils.logging import get_logger, setup_logging

console = Console()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation benchmarks")
    parser.add_argument(
        "--dataset",
        choices=["financebench", "tatqa", "both", "custom"],
        default="financebench",
        help="Evaluation dataset (use 'custom' for hand-curated eval set)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "eval", "all"],
        default="dev",
        help="Dataset split (default: dev)",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Filter to questions about a specific company (case-insensitive match)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of evaluation samples to run",
    )
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load dataset only, don't run pipeline",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    console.print(
        f"[bold cyan]Finance RAG Evaluation[/bold cyan]\n"
        f"Dataset: {args.dataset} | Split: {args.split}"
        + (f" | Company: {args.company}" if args.company else "")
        + (f" | Limit: {args.limit}" if args.limit else "")
    )

    # Load datasets
    samples = []
    if args.dataset == "custom":
        samples = _load_custom_eval()
    else:
        if args.dataset in ("financebench", "both"):
            fb_samples = load_financebench(split=args.split)
            console.print(f"FinanceBench: {len(fb_samples)} samples loaded")
            samples.extend(fb_samples)

        if args.dataset in ("tatqa", "both"):
            tq_samples = load_tatqa(split=args.split)
            console.print(f"TAT-QA: {len(tq_samples)} samples loaded")
            samples.extend(tq_samples)

    # Apply company filter
    if args.company:
        keyword = args.company.lower()
        before = len(samples)
        samples = [
            s for s in samples
            if keyword in s.question.lower()
            or keyword in s.ground_truth.lower()
            or keyword in (s.evidence or "").lower()
        ]
        console.print(f"Company filter '{args.company}': {before} → {len(samples)} samples")

    # Apply limit
    if args.limit and len(samples) > args.limit:
        samples = samples[:args.limit]
        console.print(f"Limited to {args.limit} samples")

    if not samples:
        console.print("[red]No evaluation samples loaded![/red]")
        return

    if args.dry_run:
        console.print(f"\n[yellow]Dry run: {len(samples)} samples loaded. Exiting.[/yellow]")
        _display_sample_preview(samples[:10])
        return

    # Initialize pipeline (same as query.py)
    console.print("\n[bold]Initializing pipeline...[/bold]")

    from qdrant_client import QdrantClient

    from src.embedding.embedder import BGEEmbedder
    from src.orchestration.graph import build_crag_graph, run_query
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_expander import ParentExpander
    from src.retrieval.reranker import CrossEncoderReranker

    qdrant_client = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key
    )
    embedder = BGEEmbedder(model_name=settings.embedding_model)
    searcher = HybridSearcher(
        client=qdrant_client,
        embedder=embedder,
        collection_name=settings.qdrant_collection,
    )
    reranker_model = CrossEncoderReranker(model_name=settings.reranker_model)
    expander = ParentExpander(
        client=qdrant_client, collection_name=settings.qdrant_collection
    )
    graph = build_crag_graph(
        searcher=searcher,
        reranker=reranker_model,
        expander=expander,
        qdrant_client=qdrant_client,
        collection_name=settings.qdrant_collection,
    )

    # Create pipeline function for Ragas
    def pipeline_fn(question: str) -> dict:
        return run_query(graph, question)

    # Run Ragas evaluation
    from src.evaluation.ragas_eval import run_ragas_evaluation

    console.print(f"\n[bold]Running evaluation on {len(samples)} samples...[/bold]")

    results = run_ragas_evaluation(
        eval_samples=samples,
        pipeline_fn=pipeline_fn,
        output_dir=args.output,
    )

    # Display results
    _display_results(results)


def _load_custom_eval() -> list:
    """Load hand-curated evaluation questions from data/eval/custom_eval.jsonl.

    Each line in the JSONL file should have:
        {"question": "...", "ground_truth": "...", "category": "...", "company": "..."}
    """
    import json
    from src.evaluation.benchmarks import EvalSample

    eval_path = Path(__file__).parent.parent / "data" / "eval" / "custom_eval.jsonl"
    if not eval_path.exists():
        console.print(f"[red]Custom eval file not found: {eval_path}[/red]")
        console.print("[dim]Create it with one JSON object per line:[/dim]")
        console.print('[dim]  {"question": "...", "ground_truth": "...", "category": "extraction"}[/dim]')
        return []

    samples = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            samples.append(EvalSample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                evidence=item.get("evidence", ""),
                source="custom",
                category=item.get("category", "extraction"),
                difficulty=item.get("difficulty", ""),
            ))

    console.print(f"Custom eval: {len(samples)} samples loaded from {eval_path}")
    return samples

def _display_sample_preview(samples: list) -> None:
    """Display a preview of loaded samples."""
    table = Table(title="Sample Preview")
    table.add_column("#", style="dim")
    table.add_column("Source")
    table.add_column("Question", max_width=60)
    table.add_column("Ground Truth", max_width=40)

    for i, s in enumerate(samples):
        table.add_row(
            str(i + 1),
            s.source,
            s.question[:60] + "..." if len(s.question) > 60 else s.question,
            s.ground_truth[:40] + "..." if len(s.ground_truth) > 40 else s.ground_truth,
        )
    console.print(table)


def _display_results(results: dict) -> None:
    """Display evaluation results in a rich table."""
    scores = results.get("scores", {})
    gate = results.get("gate_passed", False)

    console.print(f"\n{'='*50}")
    console.print("[bold]Evaluation Results[/bold]")
    console.print(f"{'='*50}")

    table = Table(title="Ragas Metrics")
    table.add_column("Metric")
    table.add_column("Score")
    table.add_column("Threshold")
    table.add_column("Status")

    thresholds = {
        "faithfulness": 0.95,
        "answer_relevancy": 0.80,
        "context_precision": 0.80,
        "context_recall": 0.75,
    }

    for metric, score in scores.items():
        if metric == "abstention_rate":
            table.add_row(
                metric,
                f"{score:.1%}",
                "-",
                "INFO",
            )
            continue
        threshold = thresholds.get(metric, 0.0)
        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        color = "green" if score >= threshold else "red"
        table.add_row(
            metric,
            f"[{color}]{score:.3f}[/{color}]",
            f"{threshold:.2f}",
            status,
        )

    console.print(table)

    gate_status = "[green]✅ PASSED[/green]" if gate else "[red]❌ FAILED[/red]"
    console.print(f"\nFaithfulness Gate: {gate_status}")
    console.print(f"Results saved to: {results.get('output_dir', 'results/')}")


if __name__ == "__main__":
    main()
