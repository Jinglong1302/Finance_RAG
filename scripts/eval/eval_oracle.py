"""Oracle evaluation for the 22 holdout questions.

Condition 1: context = gold chunk(s) only, bypassing grader/rewriter/refuse.
Condition 2: context = gold chunk(s) + top-4 from the pipeline.

Classified with the same classifier: classify_qa_result (span_match / numeric_match / refusal).
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
    check_filing_indexed,
    load_in_corpus_financebench,
)
from src.evaluation.metrics import classify_qa_result, span_match
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


def build_gold_context(evidence_entries: list[dict], ticker: str, doc_period: Any, doc_name: str) -> list[dict]:
    contexts = []
    for i, ev in enumerate(evidence_entries):
        text = ev.get("evidence_text", "")
        contexts.append({
            "child_text": text,
            "parent_text": None,
            "text": text,
            "metadata": {
                "company_ticker": ticker,
                "section_title": f"Gold Evidence {i+1}",
                "fiscal_year": doc_period or "Unknown",
                "filing_type": "10-Q" if ("10Q" in str(doc_name).upper() or "10-Q" in str(doc_name).upper()) else "10-K",
                "page_number": ev.get("evidence_page_num"),
            },
        })
    return contexts


def call_generator(client: OpenAI, model: str, query: str, contexts: list[dict]) -> tuple[str, float, dict]:
    formatted_ctx = format_context_for_generation(contexts)
    user_prompt = GENERATION_USER_PROMPT.format(
        confidence="high",
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
    console.print(f"[bold cyan]Running Oracle Evaluation on 22 Holdout Questions (git {sha[:7]}, dirty={dirty})[/bold cyan]")

    from qdrant_client import QdrantClient
    from src.embedding.embedder import BGEEmbedder
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_expander import ParentExpander
    from src.retrieval.reranker import CrossEncoderReranker
    from src.orchestration.nodes.query_decomposer import query_decomposer_node
    from src.orchestration.nodes.retriever import make_retriever_node
    from src.orchestration.nodes.reranker_node import make_reranker_node

    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = BGEEmbedder(model_name=settings.embedding_model)
    searcher = HybridSearcher(qdrant, embedder, settings.qdrant_collection)
    expander = ParentExpander(qdrant, settings.qdrant_collection)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)
    openai_client = OpenAI(api_key=settings.openai_api_key)
    ret_node = make_retriever_node(searcher)
    rerank_node = make_reranker_node(reranker, expander)

    all_rows = load_in_corpus_financebench()
    holdout_rows = all_rows[5:]  # Offset 5 -> 22 holdout questions

    results_cond1 = []  # Gold only
    results_cond2 = []  # Gold + top-4 pipeline

    for idx, row in enumerate(holdout_rows):
        q_num = idx + 1
        ticker = CORPUS_TICKER_MAP[row["company"]]
        query = row["question"]
        gt = str(row["answer"])
        doc_period = row.get("doc_period")
        doc_name = row.get("doc_name", "")
        ev_entries = _extract_evidence_entries(row)

        console.print(f"Processing Q{q_num:02d} [{ticker}]: {query[:60]}...")

        # 1. Gold only
        gold_ctx = build_gold_context(ev_entries, ticker, doc_period, doc_name)
        ans1, lat1, tok1 = call_generator(openai_client, settings.openai_model, query, gold_ctx)
        refused1 = _is_refusal(ans1)
        cls1 = classify_qa_result(ans1, gt, is_indexed=True, is_refusal=refused1)
        num_match1 = numeric_match(ans1, gt) if any(c.isdigit() for c in gt) else None
        sp_match1 = span_match(ans1, gt)

        results_cond1.append({
            "index": q_num,
            "ticker": ticker,
            "question": query,
            "ground_truth": gt,
            "answer": ans1,
            "latency_s": round(lat1, 2),
            "classification": cls1,
            "is_refusal": refused1,
            "numeric_match": num_match1,
            "span_match": sp_match1,
        })

        # 2. Pipeline retrieval to get top-4
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
        state.update(rerank_node(state))

        pipeline_chunks = (state.get("enriched_contexts") or state.get("reranked_results") or [])[:4]
        # Combined context: gold chunk(s) + top-4 pipeline chunks
        combined_ctx = gold_ctx + pipeline_chunks
        ans2, lat2, tok2 = call_generator(openai_client, settings.openai_model, query, combined_ctx)
        refused2 = _is_refusal(ans2)
        cls2 = classify_qa_result(ans2, gt, is_indexed=True, is_refusal=refused2)
        num_match2 = numeric_match(ans2, gt) if any(c.isdigit() for c in gt) else None
        sp_match2 = span_match(ans2, gt)

        results_cond2.append({
            "index": q_num,
            "ticker": ticker,
            "question": query,
            "ground_truth": gt,
            "answer": ans2,
            "latency_s": round(lat2, 2),
            "classification": cls2,
            "is_refusal": refused2,
            "numeric_match": num_match2,
            "span_match": sp_match2,
            "pipeline_chunks_count": len(pipeline_chunks),
        })

        console.print(f"  Q{q_num:02d}: Cond1={cls1} | Cond2={cls2}")

    breakdown1 = dict(Counter(r["classification"] for r in results_cond1))
    breakdown2 = dict(Counter(r["classification"] for r in results_cond2))

    table = Table(title="Oracle Test Results (n=22 holdout questions)")
    table.add_column("Condition")
    table.add_column("Answered Correct", style="green")
    table.add_column("Answered Incorrect", style="yellow")
    table.add_column("Abstained", style="red")
    table.add_column("Accuracy (%)", style="cyan")

    acc1 = round(100 * breakdown1.get("answered-correct", 0) / 22, 1)
    acc2 = round(100 * breakdown2.get("answered-correct", 0) / 22, 1)

    table.add_row(
        "Gold Context Only",
        str(breakdown1.get("answered-correct", 0)),
        str(breakdown1.get("answered-incorrect", 0)),
        str(breakdown1.get("abstained-incorrectly", 0)),
        f"{acc1}%",
    )
    table.add_row(
        "Gold + Top-4 Pipeline",
        str(breakdown2.get("answered-correct", 0)),
        str(breakdown2.get("answered-incorrect", 0)),
        str(breakdown2.get("abstained-incorrectly", 0)),
        f"{acc2}%",
    )
    console.print(table)

    output_payload = {
        "evaluation": "oracle_test",
        "git_commit": sha,
        "dirty": dirty,
        "config": {
            "model": settings.openai_model,
            "reranker_model": settings.reranker_model,
            "embedding_model": settings.embedding_model,
            "qdrant_collection": settings.qdrant_collection,
        },
        "condition_1_gold_only": {
            "breakdown": breakdown1,
            "per_question": results_cond1,
        },
        "condition_2_gold_plus_top4": {
            "breakdown": breakdown2,
            "per_question": results_cond2,
        },
    }

    out_path = Path("results/eval/oracle_eval.json")
    out_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    console.print(f"Saved oracle results to {out_path}")


if __name__ == "__main__":
    main()
