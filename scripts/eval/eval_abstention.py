"""Slice 4 — Abstention Evaluation.

Tests whether the CRAG pipeline correctly refuses to answer questions
about companies whose filings are NOT indexed in Qdrant.

Out-of-corpus companies: all FinanceBench companies except
{3M, Boeing, Coca-Cola, Netflix, Pfizer, Apple}.

Success criterion: the pipeline returns an abstention / refusal response
(not a hallucinated answer) when it cannot find supporting evidence.

Metrics:
  - abstention_rate: fraction of out-of-corpus questions that trigger a refusal
  - false_answer_rate: fraction that return a confident (non-refusal) answer
  - CRAG vs Naive comparison

Usage:
    poetry run python scripts/eval/eval_abstention.py
    poetry run python scripts/eval/eval_abstention.py --limit 20 --no-baseline
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

from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from src.utils.logging import setup_logging

console = Console()

# Companies whose 10-Ks are indexed in Qdrant
IN_CORPUS_COMPANIES = {"3M", "Boeing", "Coca-Cola", "Netflix", "Pfizer", "Apple"}

# Extended refusal phrases (mirrors ragas_eval._is_refusal + extras)
_REFUSAL_PHRASES = (
    "could not find sufficient evidence",
    "insufficient evidence",
    "not enough information",
    "no information found",
    "unable to locate",
    "not available in",
    "not indexed",
    "outside the scope",
    "not in the provided",
    "i don't have",
    "cannot find",
    "no relevant",
    "i was unable",
    "the filing is not",
)


def _is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(p in lower for p in _REFUSAL_PHRASES)


def load_out_of_corpus() -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="PatronusAI/financebench",
        filename="financebench_merged.jsonl",
        repo_type="dataset",
    )
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("company", "") not in IN_CORPUS_COMPANIES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Abstention evaluation slice")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--output", default="results/eval/abstention")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    console.print("[bold cyan]Slice 4 — Abstention Evaluation[/bold cyan]")

    rows = load_out_of_corpus()
    if args.limit:
        rows = rows[: args.limit]
    console.print(f"Loaded {len(rows)} out-of-corpus questions")

    # Init pipeline
    from qdrant_client import QdrantClient
    from src.embedding.embedder import BGEEmbedder
    from src.orchestration.graph import build_crag_graph, run_query
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

    crag_results: list[dict] = []
    naive_results: list[dict] = []

    for i, row in enumerate(rows):
        q = row.get("question", "")
        company = row.get("company", "")

        # CRAG
        t0 = time.perf_counter()
        try:
            result = run_query(graph, q)
            answer = result.get("final_answer") or result.get("generation", "")
        except Exception as e:
            answer = f"Error: {e}"
        latency = round(time.perf_counter() - t0, 3)

        refused = _is_refusal(answer)
        crag_results.append(
            {
                "question": q[:80],
                "company": company,
                "answer_preview": answer[:120],
                "refused": refused,
                "latency_s": latency,
            }
        )

        icon = "✅" if refused else "⚠️ "
        console.print(
            f"  {icon} Q{i+1:03d} [{company}] refused={refused} ({latency}s)"
        )

        # Naive baseline
        if naive:
            t0 = time.perf_counter()
            nres = naive.run(q)
            n_refused = _is_refusal(nres["answer"])
            naive_results.append(
                {
                    "question": q[:80],
                    "company": company,
                    "answer_preview": nres["answer"][:120],
                    "refused": n_refused,
                    "latency_s": round(time.perf_counter() - t0, 3),
                }
            )

    # Metrics
    n = len(crag_results)
    crag_abstention = sum(r["refused"] for r in crag_results) / n if n else 0.0
    crag_false_ans = 1.0 - crag_abstention

    naive_abstention = None
    naive_false_ans = None
    if naive_results:
        nn = len(naive_results)
        naive_abstention = sum(r["refused"] for r in naive_results) / nn
        naive_false_ans = 1.0 - naive_abstention

    # Display
    table = Table(title=f"Abstention Rate (out-of-corpus, n={n})")
    table.add_column("Metric")
    table.add_column("CRAG", style="green")
    if naive_results:
        table.add_column("Naive", style="yellow")

    rows_display = [
        ("Abstention Rate ↑", f"{crag_abstention:.3f}"),
        ("False Answer Rate ↓", f"{crag_false_ans:.3f}"),
    ]
    naive_display = [
        f"{naive_abstention:.3f}" if naive_abstention is not None else "n/a",
        f"{naive_false_ans:.3f}" if naive_false_ans is not None else "n/a",
    ]
    for i, (metric, cval) in enumerate(rows_display):
        row = [metric, cval]
        if naive_results:
            row.append(naive_display[i])
        table.add_row(*row)
    console.print(table)

    # Save
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "slice": "abstention",
        "timestamp": ts,
        "n_questions": n,
        "crag": {
            "abstention_rate": crag_abstention,
            "false_answer_rate": crag_false_ans,
            "per_question": crag_results,
        },
        "naive": {
            "abstention_rate": naive_abstention,
            "false_answer_rate": naive_false_ans,
            "per_question": naive_results,
        },
    }
    out_file = out / f"abstention_{ts}.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"\n[dim]Results saved → {out_file}[/dim]")


if __name__ == "__main__":
    main()
