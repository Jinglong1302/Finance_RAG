"""Part B evaluation: pooled-context sufficiency grader.

Compares the new pooled sufficiency grader against the per-chunk baseline.

Tier 0 (--tier 0, free/cheap):
  Load stored enriched_contexts from the crag_holdout22 JSON.
  Run ONLY the new pooled grader on each (no generator).
  Report: decision comparison (old vs new) and predicted coverage shift.
  Cost: ~22 × $0.002 = $0.04.

Tier 1 (--tier 1, small):
  Full pipeline with new grader on Q01-Q16 (dev-5 + first 11 holdout).
  Estimated cost: ~16 × $0.03 = $0.48.

Tier 2 (--tier 2, medium):
  Full pipeline with new grader on all 22 holdout + OOC-20 safety check.
  Estimated cost: ~22 × $0.03 + 20 × $0.01 = $0.86.
"""
from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------------------
# CRAG loop with new grader (mirrors eval_crag_holdout22.py but uses
# sufficiency_grader_node instead of grader_node)
# ---------------------------------------------------------------------------

def run_crag_part_b(
    query: str,
    ticker: str,
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
    state: dict[str, Any] = {
        "original_query": query,
        "cycle_count": 0,
        "cost_accumulated": 0.0,
        "pipeline_trace": [],
        "result_count": 5,
    }
    state.update(decompose_fn(state))
    filters = state.setdefault("structured_filters", {})
    if ticker and not filters.get("company_ticker"):
        filters["company_ticker"] = ticker

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


# ---------------------------------------------------------------------------
# Tier 0: shadow grader analysis on stored contexts (no generator)
# ---------------------------------------------------------------------------

def run_tier0(baseline_file: Path) -> dict[str, Any]:
    """Run pooled grader on stored contexts; compare decisions vs baseline."""
    from src.orchestration.nodes.sufficiency_grader import sufficiency_grader_node

    console.print(f"\n[yellow]Tier 0: shadow grader on stored contexts[/yellow]")
    console.print(f"[dim]Baseline: {baseline_file.name}[/dim]")

    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    gen_per_q = baseline.get("generation_per_question", [])

    results = []
    for orig in gen_per_q:
        q_idx = orig["index"]
        q = orig["question"]
        gt = orig["ground_truth"]
        # Reconstruct a minimal state from stored data
        # We need enriched_contexts and reranked_results for Stage 1+2
        # These are NOT stored in the baseline JSON (only generation output is).
        # So we record the original action and will compare with new grader decision
        # using a SYNTHETIC state based on stored metadata.
        #
        # Since enriched_contexts are not serialised, we cannot run the real grader
        # on stored state. Instead, we use the retrieval per-question data:
        ret_idx = q_idx - 6  # holdout starts at Q06=idx 0
        ret_data = baseline["retrieval_per_question"][ret_idx] if ret_idx >= 0 else {}

        results.append({
            "index": q_idx,
            "ticker": orig["ticker"],
            "question": q,
            "ground_truth": gt,
            "orig_classification": orig["classification"],
            "orig_action": orig.get("crag_action", ""),
            "orig_cycles": orig.get("cycle_count", 0),
            "Hit@5": ret_data.get("Hit@5", "?"),
        })

    # Print summary
    console.print("\n[bold]Tier 0: Need enriched contexts for full replay.[/bold]")
    console.print(
        "Enriched contexts are not stored in the baseline JSON.\n"
        "Proceeding to Tier 1 (live run on Q01-Q16) instead.\n"
        "Tier 0 shadow analysis will be done via Tier 1 comparison."
    )

    refused_by_grader = [r for r in results if r["orig_action"] == "refuse"]
    gen_abstained = [r for r in results
                     if r["orig_action"] == "generate"
                     and r["orig_classification"] == "abstained-incorrectly"]

    console.print(
        f"Baseline refused (action=refuse): {len(refused_by_grader)} questions"
    )
    console.print(
        f"Baseline abstained despite action=generate: {len(gen_abstained)} questions"
    )
    console.print("\n[bold]Questions refused by grader:[/bold]")
    for r in refused_by_grader:
        console.print(
            f"  Q{r['index']:02d} [{r['ticker']}] Hit@5={r['Hit@5']} | {r['question'][:70]}"
        )
    console.print("\n[bold]Questions where generator abstained (grader approved):[/bold]")
    for r in gen_abstained:
        console.print(
            f"  Q{r['index']:02d} [{r['ticker']}] Hit@5={r['Hit@5']} cycles={r['orig_cycles']}"
        )

    return {"tier": 0, "refused_by_grader": len(refused_by_grader),
            "gen_abstained": len(gen_abstained)}


# ---------------------------------------------------------------------------
# Tier 1: full pipeline on Q01-Q16 (dev-5 + first-11 holdout)
# ---------------------------------------------------------------------------

def run_tier1_or_2(
    tier: int,
    sha: str,
    dirty: bool,
    settings: Any,
    ts: str,
) -> dict[str, Any]:
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

    if tier == 1:
        # Q01-Q16: dev-5 + first 11 holdout
        rows = all_rows[:16]
        console.print(f"\n[yellow]Tier 1: Full pipeline (Part B grader) on Q01-Q16 (n={len(rows)})[/yellow]")
    else:
        # Tier 2: Q06-Q27 holdout only (OOC handled separately)
        rows = all_rows[5:]   # 22 holdout
        console.print(f"\n[yellow]Tier 2: Full pipeline (Part B grader) on Q06-Q27 (n={len(rows)})[/yellow]")

    ret_results: list[dict] = []
    gen_results: list[dict] = []
    t_wall_start = time.perf_counter()

    for idx, row in enumerate(rows):
        q_num = idx + 1 if tier == 1 else idx + 6
        ticker = CORPUS_TICKER_MAP[row["company"]]
        query = row["question"]
        gt = str(row["answer"])
        ev_entries = _extract_evidence_entries(row)

        console.print(f"  Q{q_num:02d} [{ticker}] {query[:60]}...")

        t0 = time.perf_counter()
        state = run_crag_part_b(
            query=query,
            ticker=ticker,
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

        raw_chunks = state.get("search_results", [])
        r = _retrieval_metrics(raw_chunks, ev_entries)
        r["question"] = query
        r["ticker"] = ticker
        ret_results.append(r)

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

        # Get grader verdict from trace
        trace = state.get("pipeline_trace", [])
        verdict = next(
            (t.get("verdict", "") for t in reversed(trace) if t.get("node") == "sufficiency_grader"),
            "?",
        )
        console.print(
            f"    cls={g['classification']:30s} verdict={verdict}"
            f" action={g['crag_action']} cyc={g['cycle_count']} lat={latency}s"
        )

    wall_time = round(time.perf_counter() - t_wall_start, 1)
    stats = cache_stats()
    total_cost = sum(g.get("cost_accumulated", 0.0) for g in gen_results)

    agg_ret = _agg_retrieval(ret_results)
    agg_gen = _agg_generation(gen_results)

    console.print(f"\n[bold]── Retrieval Aggregates (n={len(rows)}) ──[/bold]")
    t = Table()
    for k, v in agg_ret.items():
        t.add_column(k)
    t.add_row(*[f"{v:.4f}" for v in agg_ret.values()])
    console.print(t)

    console.print(f"\n[bold]── Generation Aggregates (n={len(rows)}) ──[/bold]")
    bd = agg_gen["breakdown"]
    console.print(
        f"  correct={bd.get('answered-correct',0)}  "
        f"incorrect={bd.get('answered-incorrect',0)}  "
        f"abstained={bd.get('abstained-incorrectly',0)}  "
        f"coverage={agg_gen['coverage']:.1%}  "
        f"precision={agg_gen['precision_when_answered']}"
    )
    console.print(
        f"\n[dim]Wall time: {wall_time}s | Cache {stats['hits']}H/{stats['misses']}M | "
        f"Cost: ${total_cost:.4f}[/dim]"
    )

    payload: dict[str, Any] = {
        "evaluation": f"part_b_tier{tier}",
        "git_commit": sha,
        "dirty": dirty,
        "timestamp": ts,
        "tier": tier,
        "grader": "pooled_sufficiency",
        "config": {
            "model": settings.openai_model,
            "reranker_model": settings.reranker_model,
            "embedding_model": settings.embedding_model,
            "qdrant_collection": settings.qdrant_collection,
            "overlap_threshold": 0.65,
            "refusal_detector": "bb3321d",
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
    out_file = out_dir / f"part_b_tier{tier}_{ts}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]Saved → {out_file}[/dim]")

    return payload


# ---------------------------------------------------------------------------
# OOC safety check (called from Tier 2)
# ---------------------------------------------------------------------------

def run_ooc_check(
    sha: str,
    dirty: bool,
    settings: Any,
    ts: str,
) -> dict[str, Any]:
    """Run pooled grader pipeline on all 20 OOC questions."""
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

    ooc_rows = [
        json.loads(l) for l in
        Path("data/eval/abstention_eval.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    console.print(f"\n[yellow]OOC safety check: {len(ooc_rows)} questions[/yellow]")

    results = []
    t_wall_start = time.perf_counter()

    for idx, row in enumerate(ooc_rows):
        query = row["question"]
        company = row.get("company", "OOC")
        console.print(f"  OOC{idx+1:02d} [{company[:15]}] {query[:55]}...")

        t0 = time.perf_counter()
        state = run_crag_part_b(
            query=query,
            ticker="",  # no ticker for OOC (they should be caught by Stage 1 coarse gate)
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
        answer = state.get("final_answer") or state.get("generation", "")
        refused = _is_refusal(answer)
        is_abstain = state.get("is_abstention", refused)

        console.print(
            f"    abstained={is_abstain} action={state.get('crag_action','')} lat={latency}s"
        )
        results.append({
            "company": company,
            "question": query,
            "answer_preview": answer[:100],
            "is_abstention": is_abstain,
            "crag_action": state.get("crag_action", ""),
            "abstention_stage": state.get("abstention_stage"),
            "latency_s": latency,
            "cost_accumulated": state.get("cost_accumulated", 0.0),
        })

    n = len(results)
    n_abstained = sum(1 for r in results if r["is_abstention"])
    abstention_rate = round(n_abstained / n, 4) if n else 0.0
    total_cost = sum(r.get("cost_accumulated", 0.0) for r in results)
    wall_time = round(time.perf_counter() - t_wall_start, 1)

    console.print(
        f"\n[bold]OOC Abstention: {n_abstained}/{n} = {abstention_rate:.1%}[/bold]"
    )
    if n_abstained < n:
        console.print("[red]WARNING: NOT all OOC questions abstained![/red]")
        for r in results:
            if not r["is_abstention"]:
                console.print(f"  LEAK: {r['company']} - {r['question'][:60]}")

    payload = {
        "evaluation": "part_b_ooc_check",
        "git_commit": sha,
        "dirty": dirty,
        "timestamp": ts,
        "n_questions": n,
        "n_abstained": n_abstained,
        "abstention_rate": abstention_rate,
        "per_question": results,
        "total_cost_usd": round(total_cost, 4),
        "wall_time_s": wall_time,
    }

    out_dir = Path("results/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"part_b_ooc_{ts}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim]Saved → {out_file}[/dim]")
    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Part B eval: pooled sufficiency grader")
    parser.add_argument(
        "--tier", type=int, default=1, choices=[0, 1, 2],
        help="0=shadow analysis, 1=Q01-Q16, 2=Q06-Q27 + OOC20"
    )
    args = parser.parse_args()

    settings = get_settings()
    sha, dirty = get_git_info()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(
        f"[bold cyan]Part B Eval — Pooled Sufficiency Grader "
        f"(Tier {args.tier}) — git {sha[:7]}, dirty={dirty}[/bold cyan]"
    )

    if args.tier == 0:
        import glob
        baseline_files = sorted(glob.glob("results/eval/crag_holdout22_*.json"))
        if not baseline_files:
            console.print("[red]No crag_holdout22_*.json found. Run eval_crag_holdout22.py first.[/red]")
            sys.exit(1)
        baseline_file = Path(baseline_files[-1])
        run_tier0(baseline_file)

    elif args.tier in (1, 2):
        result = run_tier1_or_2(args.tier, sha, dirty, settings, ts)

        if args.tier == 2:
            # Also run OOC safety check
            ooc_result = run_ooc_check(sha, dirty, settings, ts)

            abstention_rate = ooc_result.get("abstention_rate", 0.0)
            if abstention_rate < 1.0:
                console.print(
                    f"\n[bold red]OOC SAFETY GATE FAILED: "
                    f"abstention_rate={abstention_rate:.1%} (expected 100%)[/bold red]"
                )
                console.print("[bold red]STOP: Reverting to baseline grader.[/bold red]")
                sys.exit(2)
            else:
                console.print(
                    "\n[bold green]OOC safety gate PASSED: 100% abstention[/bold green]"
                )

if __name__ == "__main__":
    main()
