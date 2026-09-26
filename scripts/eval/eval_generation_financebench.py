"""Slice 2 — Generation Evaluation (FinanceBench in-corpus + Apple custom).

Runs the CRAG pipeline and naive baseline on:
  • 27 in-corpus FinanceBench questions (3M, Boeing, Coca-Cola, Netflix, Pfizer)
  • 50 Apple custom QA pairs (data/eval/custom_eval.jsonl)

Metrics: Ragas (faithfulness, answer_relevancy, context_precision,
context_recall) + numeric exact-match accuracy.

Usage:
    poetry run python scripts/eval/eval_generation_financebench.py
    poetry run python scripts/eval/eval_generation_financebench.py --limit 5 --no-baseline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import time
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from src.evaluation.benchmarks import EvalSample, load_financebench
from src.evaluation.numeric_eval import evaluate_numeric_accuracy
from src.utils.logging import setup_logging

console = Console()

IN_CORPUS_COMPANIES = {"3M", "Boeing", "Coca-Cola", "Netflix", "Pfizer"}
CORPUS_TICKER_MAP = {
    "3M": "MMM",
    "Boeing": "BA",
    "Coca-Cola": "KO",
    "Netflix": "NFLX",
    "Pfizer": "PFE",
}


def _load_custom_eval() -> list[EvalSample]:
    path = Path(__file__).parent.parent.parent / "data" / "eval" / "custom_eval.jsonl"
    if not path.exists():
        console.print(f"[yellow]custom_eval.jsonl not found at {path}[/yellow]")
        return []
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        samples.append(
            EvalSample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                evidence=item.get("evidence", ""),
                source="custom_aapl",
                category=item.get("category", "extraction"),
                difficulty=item.get("difficulty", ""),
            )
        )
    console.print(f"Loaded {len(samples)} Apple custom QA samples")
    return samples


def _load_in_corpus_fb() -> list[tuple[EvalSample, str]]:
    """Returns (EvalSample, ticker) pairs for in-corpus FinanceBench."""
    local_path = Path(__file__).parent.parent.parent / "data" / "eval" / "financebench_in_corpus.jsonl"
    if local_path.exists():
        rows = [json.loads(l) for l in open(local_path, encoding="utf-8") if l.strip()]
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="PatronusAI/financebench",
            filename="financebench_merged.jsonl",
            repo_type="dataset",
        )
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    pairs = []
    for row in rows:
        company = row.get("company", "")
        if company not in IN_CORPUS_COMPANIES:
            continue
        ev = row.get("evidence", [])
        evidence_text = (
            "\n".join(e.get("evidence_text", "") for e in ev if isinstance(e, dict))
            if isinstance(ev, list)
            else str(ev)
        )
        sample = EvalSample(
            question=row.get("question", ""),
            ground_truth=row.get("answer", ""),
            evidence=evidence_text,
            source="financebench",
            category=row.get("question_reasoning", "extraction"),
        )
        ticker = CORPUS_TICKER_MAP[company]
        pairs.append((sample, ticker))
    console.print(f"Loaded {len(pairs)} in-corpus FinanceBench samples")
    return pairs


def _run_crag(graph: Any, question: str) -> dict[str, Any]:
    from src.orchestration.graph import run_query
    return run_query(graph, question)


def _collect_crag_data(
    pairs: list[tuple[EvalSample, str]],
    graph: Any,
    limit: int | None,
) -> tuple[dict[str, list[Any]], list[str], list[str], list[str], float]:
    """Run CRAG pipeline, return ragas_data, raw_answers, predictions, ground_truths, total_latency."""
    from src.evaluation.ragas_eval import (
        _clean_answer_for_ragas,
        _format_contexts_for_ragas,
    )

    ragas_data: dict[str, list[Any]] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    raw_answers: list[str] = []
    predictions: list[str] = []
    ground_truths: list[str] = []
    total_latency = 0.0

    sample_pairs = pairs[:limit] if limit else pairs

    for i, (sample, _ticker) in enumerate(sample_pairs):
        try:
            t0 = time.perf_counter()
            result = _run_crag(graph, sample.question)
            latency = time.perf_counter() - t0
            total_latency += latency

            raw = result.get("final_answer") or result.get("generation", "")
            clean = _clean_answer_for_ragas(raw)
            ctxs = _format_contexts_for_ragas(result.get("enriched_contexts", []))

            ragas_data["question"].append(sample.question)
            ragas_data["answer"].append(clean or raw)
            ragas_data["contexts"].append(ctxs)
            ragas_data["ground_truth"].append(sample.ground_truth)
            raw_answers.append(raw)
            predictions.append(raw)
            ground_truths.append(sample.ground_truth)

            console.print(f"  CRAG Q{i+1:03d}: {latency:.1f}s")
        except Exception as e:
            console.print(f"  [red]CRAG Q{i+1:03d} error: {e}[/red]")
            ragas_data["question"].append(sample.question)
            ragas_data["answer"].append(f"Error: {e}")
            ragas_data["contexts"].append([""])
            ragas_data["ground_truth"].append(sample.ground_truth)
            raw_answers.append(f"Error: {e}")
            predictions.append("")
            ground_truths.append(sample.ground_truth)

    return ragas_data, raw_answers, predictions, ground_truths, total_latency


def _run_ragas(ragas_data: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, float | None]]]:
    import pandas as pd
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    ds = Dataset.from_dict(ragas_data)
    results = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    scores: dict[str, float] = {}
    df = results.to_pandas()
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for col in cols:
        if col in df.columns:
            valid_vals = df[col].dropna()
            scores[col] = float(valid_vals.mean()) if len(valid_vals) > 0 else 0.0

    per_sample: list[dict[str, float | None]] = []
    for _, row in df.iterrows():
        sample_scores: dict[str, float | None] = {}
        for col in cols:
            if col in df.columns:
                val = row[col]
                sample_scores[col] = float(val) if pd.notna(val) else None
            else:
                sample_scores[col] = None
        per_sample.append(sample_scores)

    return scores, per_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation eval — FinanceBench + Apple")
    parser.add_argument(
        "--source",
        choices=["all", "apple", "fb"],
        default="all",
        help="Which dataset to evaluate: 'fb' (27 in-corpus FB), 'apple' (50 AAPL custom), or 'all'",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip Ragas (latency)")
    parser.add_argument("--output", default="results/eval/generation_financebench")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    console.print(f"[bold cyan]Slice 2 — Generation ({args.source.upper()})[/bold cyan]")

    # Load data
    all_pairs: list[tuple[EvalSample, str]] = []
    if args.source in ("all", "fb"):
        all_pairs.extend(_load_in_corpus_fb())
    if args.source in ("all", "apple"):
        all_pairs.extend((s, "AAPL") for s in _load_custom_eval())

    console.print(f"Total: {len(all_pairs)} questions")

    # Init pipeline
    from qdrant_client import QdrantClient
    from src.embedding.embedder import BGEEmbedder
    from src.orchestration.graph import build_crag_graph
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_expander import ParentExpander
    from src.retrieval.reranker import CrossEncoderReranker
    from src.evaluation.baseline import NaiveRAG

    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = BGEEmbedder(model_name=settings.embedding_model)
    searcher = HybridSearcher(qdrant, embedder, settings.qdrant_collection)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)
    expander = ParentExpander(qdrant, settings.qdrant_collection)
    graph = build_crag_graph(
        searcher=searcher,
        reranker=reranker,
        expander=expander,
        qdrant_client=qdrant,
        collection_name=settings.qdrant_collection,
    )
    naive = (
        None
        if args.no_baseline
        else NaiveRAG(qdrant, embedder, settings.qdrant_collection, top_k=5)
    )

    # Run CRAG
    console.print("\n[bold]Running CRAG pipeline...[/bold]")
    crag_ragas, crag_raw, crag_preds, gts, crag_latency = _collect_crag_data(
        all_pairs, graph, args.limit
    )

    # Run naive baseline
    naive_preds: list[str] = []
    naive_latency = 0.0
    if naive:
        console.print("\n[bold]Running Naive baseline...[/bold]")
        limited_pairs = all_pairs[: args.limit] if args.limit else all_pairs
        for i, (sample, ticker) in enumerate(limited_pairs):
            t0 = time.perf_counter()
            res = naive.run(sample.question, ticker=ticker)
            naive_latency += time.perf_counter() - t0
            naive_preds.append(res["answer"])
            console.print(f"  Naive Q{i+1:03d}: {res['latency_s']}s")

    # Numeric accuracy
    crag_numeric = evaluate_numeric_accuracy(crag_preds, gts)
    naive_numeric = evaluate_numeric_accuracy(naive_preds, gts) if naive_preds else {}

    # Ragas
    crag_ragas_scores: dict[str, float] = {}
    crag_per_sample_ragas: list[dict[str, float | None]] = []
    if not args.skip_ragas:
        console.print("\n[bold]Computing Ragas metrics...[/bold]")
        try:
            crag_ragas_scores, crag_per_sample_ragas = _run_ragas(crag_ragas)
        except Exception as e:
            console.print(f"[red]Ragas failed: {e}[/red]")

    # Display
    table = Table(title="Generation Metrics (FinanceBench in-corpus + Apple, n≤77)")
    table.add_column("Metric")
    table.add_column("CRAG", style="green")
    if naive_preds:
        table.add_column("Naive", style="yellow")

    rows_data = [
        ("Numeric Accuracy", f"{crag_numeric.get('accuracy', 0):.3f}"),
        ("Avg Latency (s)", f"{crag_latency / max(len(crag_preds), 1):.1f}"),
    ]
    naive_row_data = [
        f"{naive_numeric.get('accuracy', 0):.3f}",
        f"{naive_latency / max(len(naive_preds), 1):.1f}",
    ]
    for metric, cval in crag_ragas_scores.items():
        rows_data.append((metric, f"{cval:.3f}"))
        naive_row_data.append("n/a")

    for i, (metric, cval) in enumerate(rows_data):
        row = [metric, cval]
        if naive_preds and i < len(naive_row_data):
            row.append(naive_row_data[i])
        table.add_row(*row)

    console.print(table)

    # Save
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_evaluated = args.limit or len(all_pairs)
    report = {
        "slice": "generation_financebench",
        "timestamp": ts,
        "n_questions": n_evaluated,
        "crag": {
            "ragas": crag_ragas_scores,
            "numeric_accuracy": crag_numeric.get("accuracy", 0.0),
            "avg_latency_s": crag_latency / max(n_evaluated, 1),
            "per_question": [
                {
                    "question": crag_ragas["question"][i],
                    "ground_truth": crag_ragas["ground_truth"][i],
                    "answer": crag_raw[i] if i < len(crag_raw) else "",
                    "faithfulness": crag_per_sample_ragas[i].get("faithfulness") if i < len(crag_per_sample_ragas) else None,
                    "answer_relevancy": crag_per_sample_ragas[i].get("answer_relevancy") if i < len(crag_per_sample_ragas) else None,
                    "context_precision": crag_per_sample_ragas[i].get("context_precision") if i < len(crag_per_sample_ragas) else None,
                    "context_recall": crag_per_sample_ragas[i].get("context_recall") if i < len(crag_per_sample_ragas) else None,
                }
                for i in range(len(crag_ragas["question"]))
            ],
        },
        "naive": {
            "numeric_accuracy": naive_numeric.get("accuracy", 0.0) if naive_numeric else None,
            "avg_latency_s": naive_latency / max(len(naive_preds), 1) if naive_preds else None,
            "per_question": [
                {
                    "question": crag_ragas["question"][i] if i < len(crag_ragas["question"]) else "",
                    "ground_truth": crag_ragas["ground_truth"][i] if i < len(crag_ragas["ground_truth"]) else "",
                    "answer": naive_preds[i] if i < len(naive_preds) else "",
                }
                for i in range(len(naive_preds))
            ],
        },
    }
    out_file = out / f"generation_fb_{ts}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"\n[dim]Results saved → {out_file}[/dim]")


if __name__ == "__main__":
    main()
