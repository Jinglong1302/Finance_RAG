"""Full-scale baseline evaluations on frozen current code:
- Custom AAPL 50 (data/eval/custom_eval.jsonl)
- TAT-QA 50 (data/eval/tatqa_eval.jsonl)
- Out-of-Corpus (OOC) 20 (data/eval/abstention_eval.jsonl)
- FinanceBench 27 (data/eval/financebench_in_corpus.jsonl)

Reports coverage and precision-when-answered separately.
Ragas computed on the ANSWERED subset only for FinanceBench.
Records git SHA, dirty flag, and config.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import get_settings
from scripts.eval.eval_abstention import classify_response, load_out_of_corpus
from scripts.eval.eval_generation_financebench import (
    CORPUS_TICKER_MAP,
    _load_custom_eval,
    _load_in_corpus_fb,
    _run_crag,
    _run_ragas,
)
from scripts.eval.eval_generation_tatqa import (
    _generate_crag,
    load_tatqa_samples,
)
from src.evaluation.metrics import classify_qa_result, span_match
from src.evaluation.numeric_eval import evaluate_numeric_accuracy, numeric_match
from src.evaluation.ragas_eval import _clean_answer_for_ragas, _format_contexts_for_ragas, _is_refusal
from src.orchestration.graph import build_crag_graph

console = Console()


def get_git_info() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    return sha, bool(status)


def main() -> None:
    settings = get_settings()
    sha, dirty = get_git_info()
    console.print(f"[bold cyan]Running Full-Scale Baselines on Frozen Current Code (git {sha[:7]}, dirty={dirty})[/bold cyan]")

    from qdrant_client import QdrantClient
    from src.embedding.embedder import BGEEmbedder
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_expander import ParentExpander
    from src.retrieval.reranker import CrossEncoderReranker

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

    # ========================================================
    # 1. Out-of-Corpus (OOC) 20
    # ========================================================
    console.print("\n[bold yellow]1. Evaluating Out-of-Corpus (OOC 20)...[/bold yellow]")
    ooc_rows = load_out_of_corpus()[:20]
    ooc_results = []
    for i, row in enumerate(ooc_rows):
        q = row.get("question", "")
        t0 = time.perf_counter()
        state = _run_crag(graph, q)
        lat = time.perf_counter() - t0
        ans = state.get("final_answer") or state.get("generation", "")
        cls = classify_response(ans, state)
        ooc_results.append({
            "index": i + 1,
            "company": row.get("company", "Unknown"),
            "question": q,
            "answer": ans,
            "latency_s": round(lat, 2),
            "response_type": cls,
            "is_abstention": cls == "abstain",
        })
        console.print(f"  OOC Q{i+1:02d}: {cls} ({lat:.1f}s)")

    ooc_abstentions = sum(1 for r in ooc_results if r["is_abstention"])
    ooc_abstention_rate = ooc_abstentions / max(len(ooc_results), 1)

    # ========================================================
    # 2. TAT-QA 50 (Context-injected)
    # ========================================================
    console.print("\n[bold yellow]2. Evaluating TAT-QA 50 (Context-injected)...[/bold yellow]")
    tatqa_samples = load_tatqa_samples(50)
    tatqa_results = []
    for i, sample in enumerate(tatqa_samples):
        q = sample["question"]
        ctx = sample["context"]
        gt = sample["answer_str"]
        ans, lat = _generate_crag(q, ctx)
        refused = _is_refusal(ans)
        num_m = numeric_match(ans, gt) if any(c.isdigit() for c in gt) else None
        sp_m = span_match(ans, gt)
        is_correct = bool(num_m or sp_m)
        tatqa_results.append({
            "index": i + 1,
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "latency_s": round(lat, 2),
            "is_refusal": refused,
            "is_correct": is_correct,
            "numeric_match": num_m,
            "span_match": sp_m,
        })
        status = "✅" if is_correct else ("⚠️ Refused" if refused else "❌")
        console.print(f"  TAT-QA Q{i+1:02d}: {status} ({lat:.1f}s)")

    tatqa_answered = [r for r in tatqa_results if not r["is_refusal"]]
    tatqa_coverage = len(tatqa_answered) / max(len(tatqa_results), 1)
    tatqa_correct_answered = sum(1 for r in tatqa_answered if r["is_correct"])
    tatqa_pwa = tatqa_correct_answered / max(len(tatqa_answered), 1)

    # ========================================================
    # 3. Custom AAPL 50
    # ========================================================
    console.print("\n[bold yellow]3. Evaluating Custom AAPL 50...[/bold yellow]")
    aapl_samples = _load_custom_eval()[:50]
    aapl_results = []
    for i, sample in enumerate(aapl_samples):
        q = sample.question
        gt = sample.ground_truth
        t0 = time.perf_counter()
        state = _run_crag(graph, q)
        lat = time.perf_counter() - t0
        ans = state.get("final_answer") or state.get("generation", "")
        refused = bool(state.get("is_abstention") or _is_refusal(ans))
        num_m = numeric_match(ans, gt) if any(c.isdigit() for c in gt) else None
        sp_m = span_match(ans, gt)
        is_correct = bool(num_m or sp_m)
        aapl_results.append({
            "index": i + 1,
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "latency_s": round(lat, 2),
            "is_refusal": refused,
            "is_correct": is_correct,
            "numeric_match": num_m,
            "span_match": sp_m,
        })
        status = "✅" if is_correct else ("⚠️ Refused" if refused else "❌")
        console.print(f"  AAPL Q{i+1:02d}: {status} ({lat:.1f}s)")

    aapl_answered = [r for r in aapl_results if not r["is_refusal"]]
    aapl_coverage = len(aapl_answered) / max(len(aapl_results), 1)
    aapl_correct_answered = sum(1 for r in aapl_answered if r["is_correct"])
    aapl_pwa = aapl_correct_answered / max(len(aapl_answered), 1)

    # ========================================================
    # 4. FinanceBench 27 (In-Corpus) + Ragas on Answered Only
    # ========================================================
    console.print("\n[bold yellow]4. Evaluating FinanceBench 27 (In-Corpus)...[/bold yellow]")
    fb_pairs = _load_in_corpus_fb()[:27]
    fb_results = []
    ragas_data_answered = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, (sample, ticker) in enumerate(fb_pairs):
        q = sample.question
        gt = sample.ground_truth
        t0 = time.perf_counter()
        state = _run_crag(graph, q)
        lat = time.perf_counter() - t0
        ans = state.get("final_answer") or state.get("generation", "")
        refused = bool(state.get("is_abstention") or _is_refusal(ans))
        num_m = numeric_match(ans, gt) if any(c.isdigit() for c in gt) else None
        sp_m = span_match(ans, gt)
        is_correct = bool(num_m or sp_m)
        cls = classify_qa_result(ans, gt, is_indexed=True, is_refusal=refused)

        fb_results.append({
            "index": i + 1,
            "ticker": ticker,
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "latency_s": round(lat, 2),
            "is_refusal": refused,
            "classification": cls,
            "is_correct": is_correct,
            "numeric_match": num_m,
            "span_match": sp_m,
        })

        if not refused:
            clean = _clean_answer_for_ragas(ans)
            ctxs = _format_contexts_for_ragas(state.get("enriched_contexts", []))
            ragas_data_answered["question"].append(q)
            ragas_data_answered["answer"].append(clean or ans)
            ragas_data_answered["contexts"].append(ctxs)
            ragas_data_answered["ground_truth"].append(gt)

        console.print(f"  FB Q{i+1:02d} [{ticker}]: {cls} ({lat:.1f}s)")

    fb_answered = [r for r in fb_results if not r["is_refusal"]]
    fb_coverage = len(fb_answered) / max(len(fb_results), 1)
    fb_correct_answered = sum(1 for r in fb_answered if r["is_correct"])
    fb_pwa = fb_correct_answered / max(len(fb_answered), 1)

    fb_ragas_scores = {}
    if len(ragas_data_answered["question"]) > 0:
        console.print(f"\n[bold]Computing Ragas on Answered Subset (n={len(ragas_data_answered['question'])})...[/bold]")
        try:
            fb_ragas_scores, _ = _run_ragas(ragas_data_answered)
        except Exception as e:
            console.print(f"[red]Ragas evaluation failed: {e}[/red]")
            fb_ragas_scores = {"error": str(e)}

    # ========================================================
    # Summary Table
    # ========================================================
    table = Table(title="Full-Scale Baselines on Frozen Current Code")
    table.add_column("Slice")
    table.add_column("N", justify="right")
    table.add_column("Coverage (% Answered)", justify="right", style="cyan")
    table.add_column("Precision-When-Answered", justify="right", style="green")
    table.add_column("Abstention Rate", justify="right", style="yellow")
    table.add_column("Key Metric", justify="left")

    table.add_row(
        "Custom AAPL",
        str(len(aapl_results)),
        f"{aapl_coverage*100:.1f}%",
        f"{aapl_pwa*100:.1f}%",
        f"{(1-aapl_coverage)*100:.1f}%",
        f"Correct: {aapl_correct_answered}/{len(aapl_answered)}",
    )
    table.add_row(
        "TAT-QA (Context-Injected)",
        str(len(tatqa_results)),
        f"{tatqa_coverage*100:.1f}%",
        f"{tatqa_pwa*100:.1f}%",
        f"{(1-tatqa_coverage)*100:.1f}%",
        f"Correct: {tatqa_correct_answered}/{len(tatqa_answered)}",
    )
    table.add_row(
        "Out-of-Corpus (OOC)",
        str(len(ooc_results)),
        f"{(1-ooc_abstention_rate)*100:.1f}%",
        "N/A (Unindexed)",
        f"{ooc_abstention_rate*100:.1f}%",
        f"Abstained: {ooc_abstentions}/{len(ooc_results)}",
    )
    table.add_row(
        "FinanceBench 27 (In-Corpus)",
        str(len(fb_results)),
        f"{fb_coverage*100:.1f}%",
        f"{fb_pwa*100:.1f}%",
        f"{(1-fb_coverage)*100:.1f}%",
        f"Correct: {fb_correct_answered}/{len(fb_answered)}",
    )

    console.print(table)

    output_payload = {
        "evaluation": "fullscale_baselines",
        "git_commit": sha,
        "dirty": dirty,
        "config": {
            "model": settings.openai_model,
            "reranker_model": settings.reranker_model,
            "embedding_model": settings.embedding_model,
            "qdrant_collection": settings.qdrant_collection,
        },
        "custom_aapl_50": {
            "n": len(aapl_results),
            "coverage": round(aapl_coverage, 4),
            "precision_when_answered": round(aapl_pwa, 4),
            "n_answered": len(aapl_answered),
            "n_correct": aapl_correct_answered,
            "per_question": aapl_results,
        },
        "tatqa_50": {
            "n": len(tatqa_results),
            "coverage": round(tatqa_coverage, 4),
            "precision_when_answered": round(tatqa_pwa, 4),
            "n_answered": len(tatqa_answered),
            "n_correct": tatqa_correct_answered,
            "per_question": tatqa_results,
        },
        "ooc_20": {
            "n": len(ooc_results),
            "abstention_rate": round(ooc_abstention_rate, 4),
            "n_abstained": ooc_abstentions,
            "per_question": ooc_results,
        },
        "financebench_27": {
            "n": len(fb_results),
            "coverage": round(fb_coverage, 4),
            "precision_when_answered": round(fb_pwa, 4),
            "n_answered": len(fb_answered),
            "n_correct": fb_correct_answered,
            "ragas_answered_only": fb_ragas_scores,
            "per_question": fb_results,
        },
    }

    out_path = Path("results/eval/fullscale_baselines_eval.json")
    out_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    console.print(f"Saved full-scale baselines results to {out_path}")


if __name__ == "__main__":
    main()
