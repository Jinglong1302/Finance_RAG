"""Phase A: FB27 production-path eval — NO ticker injection.

Runs all 27 FinanceBench in-corpus questions through the full Part B CRAG loop
exactly as production does: GPT-4o extracts company_ticker from the query text;
there is NO fallback ticker injection after decompose.

Key difference vs. previous Part B runs (b630a6b):
  REMOVED: if ticker and not filters.get("company_ticker"):
               filters["company_ticker"] = ticker

Stores enriched_contexts in the result file for downstream Phase B/C analysis.
Records decomposer-extracted ticker and fiscal_year for diagnostic comparison.

Cost estimate (with disk cache):
  - Decompose: 27 calls; ~19 already cached (same queries, same temp/seed) → ~8 misses
  - Sufficiency grade: 27+ calls; ~0 cached (different context w/o ticker filter) → ~27+ misses
  - Generate/guard: ~17 if same answer rate; mostly uncached → ~17 misses each
  - Rewrite: ~9 expected; mostly uncached → ~9 misses
  Total est: 8 + 27 + 17 + 17 + 9 = ~78 new calls × ~$0.015 avg = ~$1.20
  Upper bound: $1.50 (all 27 answer, 2 rewrite cycles each)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.utils.llm_cache import enable_disk_cache
enable_disk_cache()

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


def get_git_info() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    return sha, dirty


def run_crag_no_injection(
    query: str,
    *,
    decompose_fn,
    ret_node,
    rerank_node,
    note_expander_fn,
    sufficiency_grade_fn,
    generate_fn,
    guard_fn,
    rewrite_fn,
    refuse_fn,
) -> dict[str, Any]:
    """CRAG loop with NO ticker injection — pure production path."""
    state: dict[str, Any] = {
        "original_query": query,
        "cycle_count": 0,
        "cost_accumulated": 0.0,
        "pipeline_trace": [],
        "result_count": 5,
    }
    state.update(decompose_fn(state))
    # PRODUCTION: do NOT inject ticker — decomposer must extract it from query text

    max_cycles = 2
    for cycle in range(max_cycles + 1):
        state["cycle_count"] = cycle
        state.update(ret_node(state))
        state.update(rerank_node(state))
        state.update(note_expander_fn(state))
        state.update(sufficiency_grade_fn(state))
        action = state.get("crag_action", "generate")

        if action == "generate":
            state.update(generate_fn(state))
            state.update(guard_fn(state))
            guard_result = state.get("hallucination_check", "pass")
            if guard_result not in ("pass", "error") and cycle < max_cycles:
                state["cycle_count"] = cycle + 1
                state.update(rewrite_fn(state))
                continue
            break

        elif action == "rewrite" and cycle < max_cycles:
            state["cycle_count"] = cycle + 1
            state.update(rewrite_fn(state))
            continue

        else:
            state.update(refuse_fn(state))
            break

    return state


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


def main() -> None:
    settings = get_settings()
    sha, dirty = get_git_info()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(
        f"[bold cyan]Phase A: FB27 NO-INJECTION eval — git {sha[:7]}, dirty={dirty}[/bold cyan]"
    )
    console.print("[yellow]Production path: decomposer extracts ticker from query text only[/yellow]")
    console.print("[dim]LLM disk cache: results/llm_cache/[/dim]\n")

    from qdrant_client import QdrantClient
    from src.embedding.embedder import BGEEmbedder
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_expander import ParentExpander
    from src.retrieval.reranker import CrossEncoderReranker
    from src.orchestration.nodes.query_decomposer import query_decomposer_node
    from src.orchestration.nodes.generator import generator_node
    from src.orchestration.nodes.sufficiency_grader import sufficiency_grader_node
    from src.orchestration.nodes.hallucination_guard import hallucination_guard_node
    from src.orchestration.nodes.query_rewriter import query_rewriter_node
    from src.orchestration.nodes.note_expander import make_note_expander_node
    from src.orchestration.graph import refuse_node as refuse_fn
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

    all_rows = load_in_corpus_financebench()
    assert len(all_rows) == 27, f"Expected 27 FB27 rows, got {len(all_rows)}"

    console.print(f"[yellow]Running Phase A on all {len(all_rows)} FB27 questions (NO injection)...[/yellow]\n")

    ret_results: list[dict] = []
    gen_results: list[dict] = []
    t_wall_start = time.perf_counter()

    for idx, row in enumerate(all_rows):
        q_num = idx + 1
        true_ticker = CORPUS_TICKER_MAP[row["company"]]
        query = row["question"]
        gt = str(row["answer"])
        ev_entries = _extract_evidence_entries(row)

        console.print(f"  Q{q_num:02d} [{true_ticker}] {query[:65]}...")

        t0 = time.perf_counter()
        state = run_crag_no_injection(
            query=query,
            decompose_fn=query_decomposer_node,
            ret_node=ret_node,
            rerank_node=rerank_node,
            note_expander_fn=note_expander_fn,
            sufficiency_grade_fn=sufficiency_grader_node,
            generate_fn=generator_node,
            guard_fn=hallucination_guard_node,
            rewrite_fn=query_rewriter_node,
            refuse_fn=refuse_fn,
        )
        latency = round(time.perf_counter() - t0, 2)

        # What ticker/year did the decomposer extract?
        decomp_filters = state.get("structured_filters", {})
        extracted_ticker = decomp_filters.get("company_ticker", "")
        extracted_year = decomp_filters.get("fiscal_year", "")
        ticker_ok = (extracted_ticker == true_ticker)

        raw_chunks = state.get("search_results", [])
        r = _retrieval_metrics(raw_chunks, ev_entries)
        r["question"] = query
        r["ticker"] = true_ticker
        r["extracted_ticker"] = extracted_ticker
        r["ticker_ok"] = ticker_ok
        ret_results.append(r)

        answer = state.get("final_answer") or state.get("generation", "")
        g = _classify(answer, gt)
        g.update({
            "index": q_num,
            "ticker": true_ticker,
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
            # Decomposer resolution diagnostics
            "extracted_ticker": extracted_ticker,
            "extracted_year": extracted_year,
            "ticker_resolved_correctly": ticker_ok,
            # Enriched contexts for downstream Ragas / Phase B
            "enriched_contexts": state.get("enriched_contexts", []),
        })
        gen_results.append(g)

        trace = state.get("pipeline_trace", [])
        verdict = next(
            (t.get("verdict", "") for t in reversed(trace) if t.get("node") == "sufficiency_grader"),
            "?",
        )
        ticker_flag = "" if ticker_ok else f" [red]TICKER:{extracted_ticker or 'NONE'}!={true_ticker}[/red]"
        console.print(
            f"    cls={g['classification']:30s} verdict={verdict}"
            f" action={g['crag_action']} cyc={g['cycle_count']} lat={latency}s{ticker_flag}"
        )

    wall_time = round(time.perf_counter() - t_wall_start, 1)
    stats = cache_stats()
    total_cost = sum(g.get("cost_accumulated", 0.0) for g in gen_results)

    # Aggregates
    n = len(gen_results)
    answered = [g for g in gen_results if not g["is_refusal"]]
    n_correct = sum(1 for g in answered if g["classification"] == "answered-correct")
    n_incorrect = sum(1 for g in answered if g["classification"] == "answered-incorrect")
    n_abstained = n - len(answered)
    coverage = len(answered) / n if n else 0.0
    pwa = n_correct / len(answered) if answered else 0.0

    # Ticker resolution stats
    ticker_ok_count = sum(1 for g in gen_results if g["ticker_resolved_correctly"])
    ticker_fail_count = n - ticker_ok_count
    ticker_fail_qs = [
        f"Q{g['index']:02d}[{g['ticker']}→{g['extracted_ticker'] or 'NONE'}]"
        for g in gen_results if not g["ticker_resolved_correctly"]
    ]

    agg_ret = _agg_retrieval(ret_results)

    console.print(f"\n[bold]── Phase A: FB27 (no injection) Results ──[/bold]")
    t = Table()
    t.add_column("n"); t.add_column("answered"); t.add_column("correct")
    t.add_column("incorrect"); t.add_column("abstained")
    t.add_column("coverage"); t.add_column("prec@ans")
    t.add_row(
        str(n), str(len(answered)), str(n_correct), str(n_incorrect), str(n_abstained),
        f"{coverage:.1%}", f"{pwa:.1%}",
    )
    console.print(t)

    console.print(f"\n[bold]── vs Injected Part B (b630a6b) ──[/bold]")
    console.print(f"  Injected (b630a6b):  answered=17/27 (62.96%), prec=88.2% (15/17), correct=15, wrong=2, abstained=10")
    console.print(f"  No-injection (this): answered={len(answered)}/27 ({coverage:.1%}), prec={pwa:.1%} ({n_correct}/{len(answered)}), correct={n_correct}, wrong={n_incorrect}, abstained={n_abstained}")

    console.print(f"\n[bold]── Ticker Resolution ──[/bold]")
    console.print(f"  Resolved correctly: {ticker_ok_count}/{n}")
    if ticker_fail_qs:
        console.print(f"  [red]Failures: {', '.join(ticker_fail_qs)}[/red]")

    console.print(f"\n[bold]── Retrieval (n=27) ──[/bold]")
    t2 = Table()
    for k, v in agg_ret.items():
        t2.add_column(k)
    t2.add_row(*[f"{v:.4f}" for v in agg_ret.values()])
    console.print(t2)

    console.print(
        f"\n[dim]Wall: {wall_time}s | Cache {stats['hits']}H/{stats['misses']}M | Cost: ${total_cost:.4f}[/dim]"
    )

    payload: dict[str, Any] = {
        "evaluation": "fb27_no_injection",
        "phase": "A",
        "git_commit": sha,
        "dirty": dirty,
        "timestamp": ts,
        "config": {
            "ticker_injection": False,
            "grader": "pooled_sufficiency (Part B)",
            "model": settings.openai_model,
            "reranker_model": settings.reranker_model,
            "embedding_model": settings.embedding_model,
            "qdrant_collection": settings.qdrant_collection,
            "overlap_threshold": 0.65,
        },
        "n": n,
        "n_answered": len(answered),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_abstained": n_abstained,
        "coverage": round(coverage, 4),
        "precision_when_answered": round(pwa, 4),
        "ticker_resolution": {
            "n_correct": ticker_ok_count,
            "n_failed": ticker_fail_count,
            "failures": ticker_fail_qs,
        },
        "comparison_injected_b630a6b": {
            "n_answered": 17,
            "coverage": 0.6296,
            "n_correct": 15,
            "precision_when_answered": 0.882,
        },
        "retrieval_aggregate": agg_ret,
        "retrieval_per_question": ret_results,
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
    out_file = out_dir / f"fb27_no_injection_{ts}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim]Saved → {out_file}[/dim]")


if __name__ == "__main__":
    main()
