"""Full production CRAG baseline on the 22-question holdout set.

Runs the complete CRAG pipeline (decompose → retrieve → rerank → expand_notes
→ grade → [generate | rewrite loop | refuse] → guard) on every holdout
question with the company ticker injected after decomposition (matching the
approach in eval_part_a_consolidated.py).

This is the "true current baseline" referenced in PROGRESS.md Step 1.

Key design choices
------------------
* Disk cache enabled first: unchanged prompts cost nothing on rerun.
* Ticker injected after decompose (safety, not override — decomposer usually
  gets it right but injection ensures consistency with ablation measurements).
* Retrieval metrics computed on ``search_results`` (top-25 RRF) to match the
  ablation_a baseline.  Reranked-top-5 metrics also reported.
* CRAG loop manually driven (≤ 2 cycles) to enable per-question state capture
  and ticker injection on cycle 0.

Cost estimate (before cache):
  22 × decompose  ~$0.005 each → $0.11
  22 × grade      ~$0.014 each → $0.31
  ~8  × rewrite   ~$0.003 each → $0.02
  ~8  × re-grade  ~$0.014 each → $0.11
  ~14 × generate  ~$0.020 each → $0.28
  ~14 × guard     ~$0.012 each → $0.17
  ─────────────────────────────────────
  Total estimate  ≈ $1.00  (worst-case, no cache hits)
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

# ── Enable disk cache BEFORE any OpenAI import ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.utils.llm_cache import enable_disk_cache
enable_disk_cache()

# ── Now safe to import pipeline ──────────────────────────────────────────────
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
from src.utils.llm_cache import cache_stats

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_git_info() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    return sha, dirty


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
    bd = dict(Counter(r["classification"] for r in results))
    n = len(results)
    n_ans = bd.get("answered-correct", 0) + bd.get("answered-incorrect", 0)
    cov = round(n_ans / n, 4) if n else 0.0
    pwa = round(bd.get("answered-correct", 0) / n_ans, 4) if n_ans else None
    return {"breakdown": bd, "coverage": cov, "precision_when_answered": pwa}


def run_crag_on_question(
    query: str,
    ticker: str,
    *,
    decompose_fn,
    ret_node,
    rerank_node,
    note_expander_fn,
    grade_fn,
    generate_fn,
    guard_fn,
    rewrite_fn,
    refuse_fn,
) -> dict[str, Any]:
    """Drive the CRAG pipeline manually to allow ticker injection.

    Mirrors the LangGraph topology from graph.py:
      decompose → [inject ticker] → retrieve → rerank → expand_notes →
      grade → generate → guard  (or rewrite loop, or refuse)
    """
    state: dict[str, Any] = {
        "original_query": query,
        "cycle_count": 0,
        "cost_accumulated": 0.0,
        "pipeline_trace": [],
        "result_count": 5,
    }

    # Decompose
    state.update(decompose_fn(state))

    # Inject ticker (safety: only if decomposer missed it)
    filters = state.setdefault("structured_filters", {})
    if ticker and not filters.get("company_ticker"):
        filters["company_ticker"] = ticker

    max_cycles = 2

    for cycle in range(max_cycles + 1):
        state["cycle_count"] = cycle

        # Retrieve → Rerank → Notes
        state.update(ret_node(state))
        state.update(rerank_node(state))
        state.update(note_expander_fn(state))

        # Grade
        state.update(grade_fn(state))
        action = state.get("crag_action", "generate")

        if action == "generate":
            state.update(generate_fn(state))
            state.update(guard_fn(state))
            guard_result = state.get("hallucination_check", "pass")
            if guard_result not in ("pass", "error") and cycle < max_cycles:
                # Guard triggered rewrite
                state["cycle_count"] = cycle + 1
                state.update(rewrite_fn(state))
                continue
            break  # done

        elif action == "rewrite" and cycle < max_cycles:
            state["cycle_count"] = cycle + 1
            state.update(rewrite_fn(state))
            continue

        else:
            # refuse (or rewrite but cycles exhausted)
            state.update(refuse_fn(state))
            break

    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()
    sha, dirty = get_git_info()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(
        f"[bold cyan]Full CRAG Holdout-22 Eval — git {sha[:7]}, dirty={dirty}[/bold cyan]"
    )
    console.print("[dim]LLM disk cache: results/llm_cache/[/dim]")

    # ── Infrastructure ────────────────────────────────────────────────────
    from qdrant_client import QdrantClient
    from src.embedding.embedder import BGEEmbedder
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_expander import ParentExpander
    from src.retrieval.reranker import CrossEncoderReranker
    from src.orchestration.nodes.query_decomposer import query_decomposer_node
    from src.orchestration.nodes.generator import generator_node
    from src.orchestration.nodes.grader import grader_node
    from src.orchestration.nodes.hallucination_guard import hallucination_guard_node
    from src.orchestration.nodes.query_rewriter import query_rewriter_node
    from src.orchestration.nodes.note_expander import make_note_expander_node
    from src.orchestration.graph import refuse_node
    from src.orchestration.nodes.retriever import make_retriever_node
    from src.orchestration.nodes.reranker_node import make_reranker_node

    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = BGEEmbedder(model_name=settings.embedding_model)
    searcher = HybridSearcher(qdrant, embedder, settings.qdrant_collection)
    expander = ParentExpander(qdrant, settings.qdrant_collection)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)

    ret_node = make_retriever_node(searcher)
    rerank_node = make_reranker_node(reranker, expander)
    note_expander_fn = make_note_expander_node(qdrant, settings.qdrant_collection)

    # ── Dataset ───────────────────────────────────────────────────────────
    all_rows = load_in_corpus_financebench()
    holdout_rows = all_rows[5:]  # Q06-Q27
    assert len(holdout_rows) == 22, f"Expected 22 holdout rows, got {len(holdout_rows)}"

    console.print(
        f"\n[yellow]Running full CRAG on {len(holdout_rows)} holdout questions...[/yellow]"
    )

    ret_results: list[dict] = []
    gen_results: list[dict] = []
    t_wall_start = time.perf_counter()

    for idx, row in enumerate(holdout_rows):
        q_num = idx + 6  # holdout starts at Q06
        ticker = CORPUS_TICKER_MAP[row["company"]]
        query = row["question"]
        gt = str(row["answer"])
        ev_entries = _extract_evidence_entries(row)

        console.print(f"  Q{q_num:02d} [{ticker}] {query[:60]}...")

        t0 = time.perf_counter()
        state = run_crag_on_question(
            query=query,
            ticker=ticker,
            decompose_fn=query_decomposer_node,
            ret_node=ret_node,
            rerank_node=rerank_node,
            note_expander_fn=note_expander_fn,
            grade_fn=grader_node,
            generate_fn=generator_node,
            guard_fn=hallucination_guard_node,
            rewrite_fn=query_rewriter_node,
            refuse_fn=refuse_node,
        )
        latency = round(time.perf_counter() - t0, 2)

        # Retrieval metrics: use full search_results (top-25 RRF) — matches ablation_a
        raw_chunks = state.get("search_results", [])
        r = _retrieval_metrics(raw_chunks, ev_entries)
        r["question"] = query
        r["ticker"] = ticker
        # Also capture reranked top-5 metrics
        reranked = state.get("reranked_results", [])
        r["Hit@5_reranked"] = int(hit_at_k(reranked, ev_entries, 5)) if reranked else 0
        r["MRR_reranked"] = mrr(reranked, ev_entries) if reranked else 0.0
        ret_results.append(r)

        # Generation metrics
        answer = state.get("final_answer") or state.get("generation", "")
        g = _classify(answer, gt)
        g.update({
            "index": q_num,
            "ticker": ticker,
            "question": query,
            "ground_truth": gt,
            "answer": answer,
            "latency_s": latency,
            "crag_action": state.get("crag_action", ""),
            "cycle_count": state.get("cycle_count", 0),
            "is_abstention": state.get("is_abstention", False),
            "abstention_stage": state.get("abstention_stage"),
            "abstention_reason": state.get("abstention_reason", ""),
            "cost_accumulated": state.get("cost_accumulated", 0.0),
            "hallucination_check": state.get("hallucination_check", ""),
        })
        gen_results.append(g)

        console.print(
            f"    cls={g['classification']:30s} action={g['crag_action']}"
            f" cycles={g['cycle_count']} lat={latency}s"
        )

    # ── Aggregates ────────────────────────────────────────────────────────
    wall_time = round(time.perf_counter() - t_wall_start, 1)
    stats = cache_stats()
    total_cost = sum(g.get("cost_accumulated", 0.0) for g in gen_results)

    agg_ret = _agg_retrieval(ret_results)
    agg_gen = _agg_generation(gen_results)

    console.print("\n[bold]── Retrieval Aggregates (n=22) ──[/bold]")
    t = Table()
    for k, v in agg_ret.items():
        t.add_column(k)
    t.add_row(*[f"{v:.4f}" for v in agg_ret.values()])
    console.print(t)

    console.print("\n[bold]── Generation Aggregates (n=22) ──[/bold]")
    bd = agg_gen["breakdown"]
    console.print(
        f"  correct={bd.get('answered-correct',0)}  "
        f"incorrect={bd.get('answered-incorrect',0)}  "
        f"abstained={bd.get('abstained-incorrectly',0)}  "
        f"coverage={agg_gen['coverage']:.1%}  "
        f"precision-when-answered={agg_gen['precision_when_answered']}"
    )

    console.print(
        f"\n[dim]Wall time: {wall_time}s | "
        f"Cache hits/misses: {stats['hits']}/{stats['misses']} | "
        f"Total cost: ${total_cost:.4f}[/dim]"
    )

    # ── Save ─────────────────────────────────────────────────────────────
    payload: dict[str, Any] = {
        "evaluation": "crag_holdout22",
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
            "retrieval_top_k": settings.retrieval_top_k,
            "rerank_top_k": settings.rerank_top_k,
            "max_crag_cycles": settings.max_crag_cycles,
            "reranker_coarse_threshold": settings.reranker_coarse_threshold,
        },
        "retrieval_aggregate": agg_ret,
        "retrieval_per_question": ret_results,
        "generation_aggregate": agg_gen,
        "generation_per_question": gen_results,
        "meta": {
            "wall_time_s": wall_time,
            "cache_hits": stats["hits"],
            "cache_misses": stats["misses"],
            "total_cost_usd": round(total_cost, 4),
        },
    }

    out_dir = Path("results/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"crag_holdout22_{ts}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]Saved → {out_file}[/dim]")

    return payload


if __name__ == "__main__":
    main()
