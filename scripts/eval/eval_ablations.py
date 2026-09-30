"""Ablations evaluation on all 22 holdout questions.

Ablation (a): RRF-only, reranker off (retrieval and generation).
Ablation (b): Naive retrieval (dense-only) + CRAG's generator prompt/guard, no grader loop.

Records git SHA, dirty flag, and full config metadata.
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

from openai import OpenAI
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import get_settings
from scripts.eval.eval_retrieval import (
    CORPUS_TICKER_MAP,
    _extract_evidence_entries,
    load_in_corpus_financebench,
)
from src.evaluation.baseline import NaiveRAG
from src.evaluation.metrics import check_chunk_evidence_detailed, classify_qa_result, hit_at_k, mrr, recall_at_k, span_match
from src.evaluation.numeric_eval import numeric_match
from src.evaluation.ragas_eval import _is_refusal
from src.orchestration.prompts.generation import (
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT,
    format_context_for_generation,
)
from src.utils.tokens import cost_from_usage

console = Console()


def get_git_info() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    return sha, bool(status)


def call_generator(client: OpenAI, model: str, query: str, contexts: list[dict], confidence: str = "medium") -> tuple[str, float, dict]:
    formatted_ctx = format_context_for_generation(contexts)
    user_prompt = GENERATION_USER_PROMPT.format(
        confidence=confidence,
        query=query,
        context=formatted_ctx,
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        seed=42,
    )
    latency = time.perf_counter() - t0
    ans = resp.choices[0].message.content or ""
    cost, tokens = cost_from_usage(resp.usage, model)
    return ans, latency, tokens


def main() -> None:
    settings = get_settings()
    sha, dirty = get_git_info()
    console.print(f"[bold cyan]Running Ablations on 22 Holdout Questions (git {sha[:7]}, dirty={dirty})[/bold cyan]")

    from qdrant_client import QdrantClient
    from src.embedding.embedder import BGEEmbedder
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_expander import ParentExpander
    from src.orchestration.nodes.query_decomposer import query_decomposer_node
    from src.orchestration.nodes.retriever import make_retriever_node

    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = BGEEmbedder(model_name=settings.embedding_model)
    searcher = HybridSearcher(qdrant, embedder, settings.qdrant_collection)
    expander = ParentExpander(qdrant, settings.qdrant_collection)
    ret_node = make_retriever_node(searcher)
    naive = NaiveRAG(qdrant, embedder, settings.qdrant_collection, top_k=10)
    openai_client = OpenAI(api_key=settings.openai_api_key)

    all_rows = load_in_corpus_financebench()
    holdout_rows = all_rows[5:]  # 22 holdout questions

    rrf_retrieval_results = []
    rrf_generation_results = []
    naive_retrieval_results = []
    naive_generation_results = []

    for idx, row in enumerate(holdout_rows):
        q_num = idx + 1
        ticker = CORPUS_TICKER_MAP[row["company"]]
        query = row["question"]
        gt = str(row["answer"])
        ev_entries = _extract_evidence_entries(row)

        console.print(f"Evaluating Q{q_num:02d} [{ticker}]: {query[:60]}...")

        # ----------------------------------------------------
        # Ablation (a): RRF-only, reranker off
        # ----------------------------------------------------
        state: dict[str, Any] = {
            "original_query": query,
            "cycle_count": 0,
            "cost_accumulated": 0.0,
            "pipeline_trace": [],
            "result_count": 5,
        }
        state.update(query_decomposer_node(state))
        filters = state.setdefault("structured_filters", {})
        if ticker and "company_ticker" not in filters:
            filters["company_ticker"] = ticker
        state.update(ret_node(state))

        rrf_candidates = state.get("search_results", [])
        # RRF top-5 directly (no cross-encoder)
        rrf_top5 = rrf_candidates[:5]
        # Expand parents for top-5 so context is enriched like pipeline
        from src.retrieval.hybrid_search import SearchResult
        search_objs = [
            SearchResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                metadata=r["metadata"],
                score=r.get("score", 0.0),
            )
            for r in rrf_top5
        ]
        rrf_enriched = [
            {
                "child_text": ec.child_text,
                "parent_text": ec.parent_text,
                "text": ec.child_text,
                "metadata": ec.metadata,
            }
            for ec in expander.expand(search_objs)
        ]

        # Retrieval metrics
        r_res: dict[str, Any] = {"question": query, "ticker": ticker}
        for k in (1, 3, 5, 10):
            r_res[f"Hit@{k}"] = int(hit_at_k(rrf_candidates, ev_entries, k))
        r_res["MRR"] = mrr(rrf_candidates, ev_entries)
        rrf_retrieval_results.append(r_res)

        # Generation
        ans_rrf, lat_rrf, tok_rrf = call_generator(openai_client, settings.openai_model, query, rrf_enriched, confidence="medium")
        refused_rrf = _is_refusal(ans_rrf)
        cls_rrf = classify_qa_result(ans_rrf, gt, is_indexed=True, is_refusal=refused_rrf)
        rrf_generation_results.append({
            "index": q_num,
            "ticker": ticker,
            "question": query,
            "ground_truth": gt,
            "answer": ans_rrf,
            "latency_s": round(lat_rrf, 2),
            "classification": cls_rrf,
            "is_refusal": refused_rrf,
            "span_match": span_match(ans_rrf, gt),
            "numeric_match": numeric_match(ans_rrf, gt) if any(c.isdigit() for c in gt) else None,
        })

        # ----------------------------------------------------
        # Ablation (b): Naive retrieval + CRAG generator prompt/guard, no grader loop
        # ----------------------------------------------------
        naive_chunks = naive.retrieve(query, ticker=ticker)
        # Retrieval metrics
        n_res: dict[str, Any] = {"question": query, "ticker": ticker}
        for k in (1, 3, 5, 10):
            n_res[f"Hit@{k}"] = int(hit_at_k(naive_chunks, ev_entries, k))
        n_res["MRR"] = mrr(naive_chunks, ev_entries)
        naive_retrieval_results.append(n_res)

        # Convert naive chunks to context format
        naive_ctx = []
        for c in naive_chunks[:5]:
            txt = c.get("text", "")
            meta = c.get("metadata", {})
            naive_ctx.append({
                "child_text": txt,
                "parent_text": None,
                "text": txt,
                "metadata": meta,
            })

        ans_naive, lat_naive, tok_naive = call_generator(openai_client, settings.openai_model, query, naive_ctx, confidence="medium")
        refused_naive = _is_refusal(ans_naive)
        cls_naive = classify_qa_result(ans_naive, gt, is_indexed=True, is_refusal=refused_naive)
        naive_generation_results.append({
            "index": q_num,
            "ticker": ticker,
            "question": query,
            "ground_truth": gt,
            "answer": ans_naive,
            "latency_s": round(lat_naive, 2),
            "classification": cls_naive,
            "is_refusal": refused_naive,
            "span_match": span_match(ans_naive, gt),
            "numeric_match": numeric_match(ans_naive, gt) if any(c.isdigit() for c in gt) else None,
        })

        console.print(f"  Q{q_num:02d}: RRF-Hit@5={r_res['Hit@5']} Gen={cls_rrf} | Naive-Hit@5={n_res['Hit@5']} Gen={cls_naive}")

    # Aggregates
    def agg_retrieval(res_list):
        n = len(res_list)
        return {
            "Hit@1": round(sum(r["Hit@1"] for r in res_list) / n, 4),
            "Hit@3": round(sum(r["Hit@3"] for r in res_list) / n, 4),
            "Hit@5": round(sum(r["Hit@5"] for r in res_list) / n, 4),
            "Hit@10": round(sum(r["Hit@10"] for r in res_list) / n, 4),
            "MRR": round(sum(r["MRR"] for r in res_list) / n, 4),
        }

    rrf_ret_agg = agg_retrieval(rrf_retrieval_results)
    naive_ret_agg = agg_retrieval(naive_retrieval_results)

    rrf_gen_breakdown = dict(Counter(r["classification"] for r in rrf_generation_results))
    naive_gen_breakdown = dict(Counter(r["classification"] for r in naive_generation_results))

    console.print("\n[bold]=== RETRIEVAL ABLATION RESULTS (n=22) ===[/bold]")
    ret_table = Table(title="Retrieval Ablations")
    ret_table.add_column("Metric")
    ret_table.add_column("RRF-only (No Reranker)", style="cyan")
    ret_table.add_column("Naive (Dense-only)", style="yellow")
    for k in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"):
        ret_table.add_row(k, f"{rrf_ret_agg[k]:.3f}", f"{naive_ret_agg[k]:.3f}")
    console.print(ret_table)

    console.print("\n[bold]=== GENERATION ABLATION RESULTS (n=22) ===[/bold]")
    gen_table = Table(title="Generation Ablations")
    gen_table.add_column("Classification")
    gen_table.add_column("RRF-only + CRAG Gen (no rerank)", style="cyan")
    gen_table.add_column("Naive Retrieval + CRAG Guard (no grader)", style="yellow")
    for cat in ("answered-correct", "answered-incorrect", "abstained-incorrectly"):
        gen_table.add_row(cat, str(rrf_gen_breakdown.get(cat, 0)), str(naive_gen_breakdown.get(cat, 0)))
    console.print(gen_table)

    output_payload = {
        "evaluation": "ablations",
        "git_commit": sha,
        "dirty": dirty,
        "config": {
            "model": settings.openai_model,
            "reranker_model": settings.reranker_model,
            "embedding_model": settings.embedding_model,
            "qdrant_collection": settings.qdrant_collection,
        },
        "ablation_a_rrf_only": {
            "retrieval_aggregate": rrf_ret_agg,
            "retrieval_per_question": rrf_retrieval_results,
            "generation_breakdown": rrf_gen_breakdown,
            "generation_per_question": rrf_generation_results,
        },
        "ablation_b_naive_retrieval_crag_guard": {
            "retrieval_aggregate": naive_ret_agg,
            "retrieval_per_question": naive_retrieval_results,
            "generation_breakdown": naive_gen_breakdown,
            "generation_per_question": naive_generation_results,
        },
    }

    out_path = Path("results/eval/ablations_eval.json")
    out_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    console.print(f"Saved ablations results to {out_path}")


if __name__ == "__main__":
    main()
