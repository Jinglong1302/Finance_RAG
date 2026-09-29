"""Consolidated evaluation — Part A completion.

Runs on the 22-question holdout AND the 5 dev questions in one pass:

Conditions on the 22-question HOLDOUT:
  (a) RRF-only, no reranker -> generator (no grader)
  (b) Naive dense retrieval -> generator (no grader, CRAG guard = prompt only)
  (c) Full retrieval (RRF + reranker) -> generator directly (no grader)  [NEW]
  (oracle) Gold context only -> generator

Condition on the 5 DEV questions (Q01-Q05):
  (full_crag) Full CRAG pipeline (retrieval + reranker + grader + generator)

All use:
  - Unified _is_refusal from ragas_eval.py (bb3321d)
  - overlap_match threshold 0.65 in metrics.py

Output:
  results/eval/part_a_consolidated_<timestamp>.json
  records git SHA + dirty flag
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from scripts.eval.eval_retrieval import (
    CORPUS_TICKER_MAP,
    _extract_evidence_entries,
    load_in_corpus_financebench,
)
from src.evaluation.metrics import (
    classify_qa_result,
    hit_at_k,
    mrr,
    recall_at_k,
    span_match,
)
from src.evaluation.numeric_eval import numeric_match
from src.evaluation.ragas_eval import _is_refusal
from src.orchestration.prompts.generation import (
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT,
    format_context_for_generation,
)
from src.retrieval.hybrid_search import SearchResult
from src.utils.tokens import cost_from_usage

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_git_info() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    return sha, bool(status)


def _call_generator(
    client: OpenAI,
    model: str,
    query: str,
    contexts: list[dict],
    confidence: str = "medium",
) -> tuple[str, float]:
    formatted = format_context_for_generation(contexts)
    user_prompt = GENERATION_USER_PROMPT.format(
        confidence=confidence, query=query, context=formatted
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
    return resp.choices[0].message.content or "", round(latency, 2)


def _classify(answer: str, gt: str) -> dict[str, Any]:
    refused = _is_refusal(answer)
    cls = classify_qa_result(answer, gt, is_indexed=True, is_refusal=refused)
    has_digit = any(c.isdigit() for c in gt)
    return {
        "classification": cls,
        "is_refusal": refused,
        "span_match": span_match(answer, gt),
        "numeric_match": numeric_match(answer, gt) if has_digit else None,
    }


def _retrieval_metrics(chunks: list, ev_entries: list) -> dict[str, Any]:
    r: dict[str, Any] = {}
    for k in (1, 3, 5, 10):
        r[f"Hit@{k}"] = int(hit_at_k(chunks, ev_entries, k))
    r["Recall@5"] = recall_at_k(chunks, ev_entries, 5)
    r["Recall@10"] = recall_at_k(chunks, ev_entries, 10)
    r["MRR"] = mrr(chunks, ev_entries)
    return r


def _agg_retrieval(results: list[dict]) -> dict[str, float]:
    n = len(results)
    if not n:
        return {}
    keys = ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR"]
    return {k: round(sum(r.get(k, 0) for r in results) / n, 4) for k in keys}


def _agg_generation(results: list[dict]) -> dict[str, Any]:
    breakdown = dict(Counter(r["classification"] for r in results))
    n = len(results)
    n_answered = breakdown.get("answered-correct", 0) + breakdown.get("answered-incorrect", 0)
    coverage = round(n_answered / n, 4) if n else 0.0
    pwa = (
        round(breakdown.get("answered-correct", 0) / n_answered, 4) if n_answered else None
    )
    return {"breakdown": breakdown, "coverage": coverage, "precision_when_answered": pwa}


def _enrich_to_ctx(chunks: list, expander: Any) -> list[dict]:
    """Expand raw search result dicts into generator-ready context dicts."""
    objs = [
        SearchResult(
            chunk_id=r["chunk_id"],
            text=r["text"],
            metadata=r["metadata"],
            score=r.get("score", 0.0),
        )
        for r in chunks
    ]
    return [
        {
            "child_text": ec.child_text,
            "parent_text": ec.parent_text,
            "text": ec.child_text,
            "metadata": ec.metadata,
        }
        for ec in expander.expand(objs)
    ]


def _reranked_to_ctx(reranked: list, n: int = 5) -> list[dict]:
    """Convert reranked result objects/dicts into generator-ready context dicts."""
    ctxs = []
    for r in reranked[:n]:
        if isinstance(r, dict):
            ctxs.append({
                "child_text": r.get("child_text", r.get("text", "")),
                "parent_text": r.get("parent_text"),
                "text": r.get("text", ""),
                "metadata": r.get("metadata", {}),
            })
        else:
            ctxs.append({
                "child_text": getattr(r, "child_text", "") or getattr(r, "text", ""),
                "parent_text": getattr(r, "parent_text", None),
                "text": getattr(r, "text", ""),
                "metadata": getattr(r, "metadata", {}),
            })
    return ctxs


def _build_gold_ctx(ev_entries: list[dict], ticker: str, doc_period: Any, doc_name: str) -> list[dict]:
    ctxs = []
    for i, ev in enumerate(ev_entries):
        text = ev.get("evidence_text", "")
        ctxs.append({
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
    return ctxs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()
    sha, dirty = get_git_info()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    console.print(f"[bold cyan]Part A Consolidated Eval (git {sha[:7]}, dirty={dirty})[/bold cyan]")

    # Init infrastructure
    from qdrant_client import QdrantClient
    from src.embedding.embedder import BGEEmbedder
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_expander import ParentExpander
    from src.retrieval.reranker import CrossEncoderReranker
    from src.evaluation.baseline import NaiveRAG
    from src.orchestration.nodes.query_decomposer import query_decomposer_node
    from src.orchestration.nodes.retriever import make_retriever_node
    from src.orchestration.nodes.reranker_node import make_reranker_node
    from src.orchestration.graph import build_crag_graph, run_query

    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = BGEEmbedder(model_name=settings.embedding_model)
    searcher = HybridSearcher(qdrant, embedder, settings.qdrant_collection)
    expander = ParentExpander(qdrant, settings.qdrant_collection)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)
    naive = NaiveRAG(qdrant, embedder, settings.qdrant_collection, top_k=10)
    openai_client = OpenAI(api_key=settings.openai_api_key)
    ret_node = make_retriever_node(searcher)
    rerank_node = make_reranker_node(reranker, expander)
    graph = build_crag_graph(
        searcher=searcher, reranker=reranker, expander=expander,
        qdrant_client=qdrant, collection_name=settings.qdrant_collection,
    )

    all_rows = load_in_corpus_financebench()
    dev_rows = all_rows[:5]       # Q01-Q05 (dev set)
    holdout_rows = all_rows[5:]   # Q06-Q27 (22 holdout)

    # ===================================================================
    # PART 1: 22 holdout — ablations (a), (b), (c) + oracle
    # ===================================================================
    console.print("\n[bold yellow]── 22-Question Holdout Eval ──[/bold yellow]")

    abl_a_ret, abl_a_gen = [], []
    abl_b_ret, abl_b_gen = [], []
    abl_c_ret, abl_c_gen = [], []
    oracle_gen = []

    for idx, row in enumerate(holdout_rows):
        q_num = idx + 1
        ticker = CORPUS_TICKER_MAP[row["company"]]
        query = row["question"]
        gt = str(row["answer"])
        doc_period = row.get("doc_period")
        doc_name = row.get("doc_name", "")
        ev_entries = _extract_evidence_entries(row)

        console.print(f"  Q{q_num:02d} [{ticker}] {query[:55]}...")

        # ----------------------------------------------------------------
        # Shared: RRF retrieval (used by ablation a and c)
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # Ablation (a): RRF-only, no reranker -> generator
        # ----------------------------------------------------------------
        rrf_top5 = rrf_candidates[:5]
        rrf_ctx = _enrich_to_ctx(rrf_top5, expander) if rrf_top5 else []

        r_a = _retrieval_metrics(rrf_candidates, ev_entries)
        r_a["question"] = query
        r_a["ticker"] = ticker
        abl_a_ret.append(r_a)

        ans_a, lat_a = _call_generator(openai_client, settings.openai_model, query, rrf_ctx)
        g_a = _classify(ans_a, gt)
        g_a.update({"index": q_num, "ticker": ticker, "question": query,
                    "ground_truth": gt, "answer": ans_a, "latency_s": lat_a})
        abl_a_gen.append(g_a)

        # ----------------------------------------------------------------
        # Ablation (b): Naive dense -> generator
        # ----------------------------------------------------------------
        naive_chunks = naive.retrieve(query, ticker=ticker)
        r_b = _retrieval_metrics(naive_chunks, ev_entries)
        r_b["question"] = query
        r_b["ticker"] = ticker
        abl_b_ret.append(r_b)

        naive_ctx = [
            {"child_text": c.get("text", ""), "parent_text": None,
             "text": c.get("text", ""), "metadata": c.get("metadata", {})}
            for c in naive_chunks[:5]
        ]
        ans_b, lat_b = _call_generator(openai_client, settings.openai_model, query, naive_ctx)
        g_b = _classify(ans_b, gt)
        g_b.update({"index": q_num, "ticker": ticker, "question": query,
                    "ground_truth": gt, "answer": ans_b, "latency_s": lat_b})
        abl_b_gen.append(g_b)

        # ----------------------------------------------------------------
        # Ablation (c): Full retrieval (RRF + reranker) -> generator (no grader)
        # ----------------------------------------------------------------
        state_c: dict[str, Any] = {
            "original_query": query,
            "cycle_count": 0,
            "cost_accumulated": 0.0,
            "pipeline_trace": [],
            "result_count": 5,
            "structured_filters": {"company_ticker": ticker} if ticker else {},
        }
        state_c.update(query_decomposer_node(state_c))
        state_c["structured_filters"].setdefault("company_ticker", ticker)
        state_c.update(ret_node(state_c))
        state_c.update(rerank_node(state_c))

        reranked = state_c.get("reranked_results", [])
        r_c = _retrieval_metrics(reranked if reranked else state_c.get("search_results", []), ev_entries)
        r_c["question"] = query
        r_c["ticker"] = ticker
        abl_c_ret.append(r_c)

        # Context: enriched_contexts set by reranker_node, or fall back to reranked
        c_ctx = state_c.get("enriched_contexts") or _reranked_to_ctx(reranked)
        if not c_ctx:
            c_ctx = [{"child_text": "", "parent_text": None, "text": "", "metadata": {}}]

        ans_c, lat_c = _call_generator(openai_client, settings.openai_model, query, c_ctx)
        g_c = _classify(ans_c, gt)
        g_c.update({"index": q_num, "ticker": ticker, "question": query,
                    "ground_truth": gt, "answer": ans_c, "latency_s": lat_c})
        abl_c_gen.append(g_c)

        # ----------------------------------------------------------------
        # Oracle: gold context only -> generator
        # ----------------------------------------------------------------
        gold_ctx = _build_gold_ctx(ev_entries, ticker, doc_period, doc_name)
        ans_o, lat_o = _call_generator(openai_client, settings.openai_model, query, gold_ctx, confidence="high")
        g_o = _classify(ans_o, gt)
        g_o.update({"index": q_num, "ticker": ticker, "question": query,
                    "ground_truth": gt, "answer": ans_o, "latency_s": lat_o})
        oracle_gen.append(g_o)

        console.print(
            f"    (a)RRF={g_a['classification'][:2]}  "
            f"(b)Naive={g_b['classification'][:2]}  "
            f"(c)RRF+R={g_c['classification'][:2]}  "
            f"Oracle={g_o['classification'][:2]}"
        )

    # ===================================================================
    # PART 2: 5 dev questions — full CRAG pipeline
    # ===================================================================
    console.print("\n[bold yellow]── 5 Dev Questions — Full CRAG Pipeline ──[/bold yellow]")
    dev_gen = []

    for idx, row in enumerate(dev_rows):
        q_num = idx + 1
        ticker = CORPUS_TICKER_MAP[row["company"]]
        query = row["question"]
        gt = str(row["answer"])
        ev_entries = _extract_evidence_entries(row)

        console.print(f"  Dev Q{q_num:02d} [{ticker}] {query[:55]}...")
        t0 = time.perf_counter()
        state = run_query(graph, query)
        lat = time.perf_counter() - t0
        ans = state.get("final_answer") or state.get("generation", "")
        g = _classify(ans, gt)
        g.update({
            "index": q_num, "ticker": ticker, "question": query,
            "ground_truth": gt, "answer": ans, "latency_s": round(lat, 2),
            "crag_action": state.get("crag_action", ""),
            "is_abstention": state.get("is_abstention", False),
            "abstention_reason": state.get("abstention_reason", ""),
        })
        dev_gen.append(g)
        console.print(f"    Full CRAG={g['classification']}  action={g['crag_action']}")

    # ===================================================================
    # Aggregate and display
    # ===================================================================
    console.print("\n[bold]══ CONSOLIDATED RESULTS ══[/bold]")

    def _show_gen_table(title: str, results: list[dict]) -> None:
        agg = _agg_generation(results)
        bd = agg["breakdown"]
        t = Table(title=title)
        t.add_column("correct")
        t.add_column("incorrect")
        t.add_column("abstained")
        t.add_column("coverage")
        t.add_column("prec-when-ans")
        t.add_row(
            str(bd.get("answered-correct", 0)),
            str(bd.get("answered-incorrect", 0)),
            str(bd.get("abstained-incorrectly", 0)),
            f"{agg['coverage']:.1%}",
            f"{agg['precision_when_answered']:.1%}" if agg["precision_when_answered"] is not None else "N/A",
        )
        console.print(t)

    def _show_ret_table(title: str, abl_a: list, abl_b: list, abl_c: list) -> None:
        agg_a = _agg_retrieval(abl_a)
        agg_b = _agg_retrieval(abl_b)
        agg_c = _agg_retrieval(abl_c)
        t = Table(title=title)
        t.add_column("Metric")
        t.add_column("(a) RRF-only", style="cyan")
        t.add_column("(b) Naive dense", style="yellow")
        t.add_column("(c) RRF+Reranker", style="green")
        for k in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR"]:
            t.add_row(k, f"{agg_a.get(k,0):.3f}", f"{agg_b.get(k,0):.3f}", f"{agg_c.get(k,0):.3f}")
        console.print(t)

    _show_ret_table("Retrieval — 22 Holdout (overlap@0.65)", abl_a_ret, abl_b_ret, abl_c_ret)
    _show_gen_table("Generation (a) RRF-only → gen (n=22)", abl_a_gen)
    _show_gen_table("Generation (b) Naive dense → gen (n=22)", abl_b_gen)
    _show_gen_table("Generation (c) RRF+Reranker → gen, NO GRADER (n=22)", abl_c_gen)
    _show_gen_table("Oracle gold context → gen (n=22)", oracle_gen)
    _show_gen_table("Full CRAG — 5 Dev Questions", dev_gen)

    console.print("\n[bold]── Per-question: 5 Dev Qs (Full CRAG) ──[/bold]")
    for r in dev_gen:
        console.print(
            f"  Dev Q{r['index']:02d} [{r['ticker']}] {r['classification']:30s}"
            f"  refusal={r['is_refusal']}  action={r['crag_action']}\n"
            f"    Q: {r['question'][:70]}\n"
            f"    GT: {r['ground_truth'][:60]}\n"
            f"    A:  {r['answer'][:80].replace(chr(10),' ')}"
        )

    # ===================================================================
    # Save
    # ===================================================================
    payload = {
        "evaluation": "part_a_consolidated",
        "git_commit": sha,
        "dirty": dirty,
        "timestamp": ts,
        "config": {
            "model": settings.openai_model,
            "reranker_model": settings.reranker_model,
            "embedding_model": settings.embedding_model,
            "qdrant_collection": settings.qdrant_collection,
            "overlap_threshold": 0.65,
            "refusal_detector": "bb3321d",
        },
        "holdout_22": {
            "ablation_a_rrf_only": {
                "retrieval_aggregate": _agg_retrieval(abl_a_ret),
                "retrieval_per_question": abl_a_ret,
                "generation": _agg_generation(abl_a_gen),
                "generation_per_question": abl_a_gen,
            },
            "ablation_b_naive": {
                "retrieval_aggregate": _agg_retrieval(abl_b_ret),
                "retrieval_per_question": abl_b_ret,
                "generation": _agg_generation(abl_b_gen),
                "generation_per_question": abl_b_gen,
            },
            "ablation_c_full_retrieval_no_grader": {
                "retrieval_aggregate": _agg_retrieval(abl_c_ret),
                "retrieval_per_question": abl_c_ret,
                "generation": _agg_generation(abl_c_gen),
                "generation_per_question": abl_c_gen,
            },
            "oracle_gold_only": {
                "generation": _agg_generation(oracle_gen),
                "generation_per_question": oracle_gen,
            },
        },
        "dev_5_full_crag": {
            "generation": _agg_generation(dev_gen),
            "generation_per_question": dev_gen,
        },
    }

    out = Path("results/eval")
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"part_a_consolidated_{ts}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]Saved → {out_file}[/dim]")


if __name__ == "__main__":
    main()
