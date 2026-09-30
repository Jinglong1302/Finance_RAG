"""AAPL50 evaluation with the Part B pooled-context sufficiency grader.

Runs the same manual CRAG loop used in eval_part_b.py (with ticker injection)
on the custom 50-question Apple eval set.  TAT-QA50 is skipped here because
it uses direct context injection that bypasses the grader entirely — results
are identical to baseline.

Cost estimate: ~$0.30–0.40
  ~50 sufficiency grade calls × $0.003   = ~$0.15
  ~10 new generate/guard calls (routing changes) × $0.025 = ~$0.25
  Decompose/rewrite/guard: mostly cached
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
from scripts.eval.eval_generation_financebench import _load_custom_eval
from src.evaluation.numeric_eval import numeric_match
from src.evaluation.ragas_eval import _is_refusal
from src.evaluation.metrics import span_match
from src.utils.llm_cache import cache_stats

console = Console()


def get_git_info() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    return sha, dirty


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


def main() -> None:
    settings = get_settings()
    sha, dirty = get_git_info()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(f"[bold cyan]Part B AAPL50 eval — git {sha[:7]}, dirty={dirty}[/bold cyan]")
    console.print("[dim]LLM disk cache: results/llm_cache/[/dim]")

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

    samples = _load_custom_eval()[:50]
    assert len(samples) == 50, f"Expected 50 AAPL samples, got {len(samples)}"

    console.print(f"\n[yellow]Running Part B CRAG on {len(samples)} AAPL questions...[/yellow]")

    results = []
    t_wall_start = time.perf_counter()

    for idx, sample in enumerate(samples):
        q = sample.question
        gt = str(sample.ground_truth)
        ticker = "AAPL"

        t0 = time.perf_counter()
        state = run_crag_part_b(
            query=q,
            ticker=ticker,
            decompose_fn=query_decomposer_node,
            ret_node=ret_node,
            rerank_node=rerank_node,
            note_expander_fn=note_expander_fn,
            sufficiency_grade_fn=sufficiency_grader_node,
            generate_fn=generator_node,
            guard_fn=hallucination_guard_node,
            rewrite_fn=query_rewriter_node,
            refuse_fn=refuse_node,
        )
        latency = round(time.perf_counter() - t0, 2)

        answer = state.get("final_answer") or state.get("generation", "")
        refused = bool(state.get("is_abstention") or _is_refusal(answer))
        has_digit = any(c.isdigit() for c in gt)
        num_m = numeric_match(answer, gt) if has_digit else None
        sp_m = span_match(answer, gt)
        is_correct = bool(num_m or sp_m)

        results.append({
            "index": idx + 1,
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "latency_s": latency,
            "is_refusal": refused,
            "is_correct": is_correct,
            "numeric_match": num_m,
            "span_match": sp_m,
            "crag_action": state.get("crag_action", ""),
            "cycle_count": state.get("cycle_count", 0),
            "cost_accumulated": state.get("cost_accumulated", 0.0),
        })

        status = "c" if is_correct else ("A" if refused else "i")
        console.print(f"  A{idx+1:02d}: {status}  lat={latency}s  {q[:55]}...")

    wall_time = round(time.perf_counter() - t_wall_start, 1)
    stats = cache_stats()
    total_cost = sum(r.get("cost_accumulated", 0.0) for r in results)

    answered = [r for r in results if not r["is_refusal"]]
    n_correct = sum(1 for r in answered if r["is_correct"])
    coverage = len(answered) / len(results)
    pwa = n_correct / len(answered) if answered else 0.0

    console.print(f"\n[bold]── AAPL50 Part B Results ──[/bold]")
    t = Table()
    t.add_column("n"); t.add_column("n_answered"); t.add_column("n_correct"); t.add_column("coverage"); t.add_column("prec@ans")
    t.add_row(str(len(results)), str(len(answered)), str(n_correct), f"{coverage:.1%}", f"{pwa:.1%}")
    console.print(t)

    console.print(f"\n[bold]── vs Baseline (94fdd2f, per-chunk grader) ──[/bold]")
    console.print(f"  Baseline: n_answered=42/50 (84.0%), prec=95.2% (40/42)")
    console.print(f"  Part B:   n_answered={len(answered)}/50 ({coverage:.1%}), prec={pwa:.1%} ({n_correct}/{len(answered)})")

    gate_cov = coverage >= 0.84
    gate_prec = pwa >= 0.90
    console.print(f"\n  Gate coverage ≥84%: {'PASS' if gate_cov else 'FAIL'} ({coverage:.1%})")
    console.print(f"  Gate precision ≥90%: {'PASS' if gate_prec else 'FAIL'} ({pwa:.1%})")

    console.print(
        f"\n[dim]Wall: {wall_time}s | Cache {stats['hits']}H/{stats['misses']}M | Cost: ${total_cost:.4f}[/dim]"
    )

    payload: dict[str, Any] = {
        "evaluation": "part_b_aapl50",
        "git_commit": sha,
        "dirty": dirty,
        "timestamp": ts,
        "config": {
            "model": settings.openai_model,
            "reranker_model": settings.reranker_model,
            "grader": "sufficiency_grader_node",
        },
        "n": len(results),
        "n_answered": len(answered),
        "n_correct": n_correct,
        "coverage": round(coverage, 4),
        "precision_when_answered": round(pwa, 4),
        "gate_coverage_pass": gate_cov,
        "gate_precision_pass": gate_prec,
        "per_question": results,
        "meta": {
            "wall_time_s": wall_time,
            "cache_hits": stats["hits"],
            "cache_misses": stats["misses"],
            "total_cost_usd": round(total_cost, 4),
        },
    }

    out_dir = Path("results/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"part_b_aapl50_{ts}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim]Saved → {out_file}[/dim]")


if __name__ == "__main__":
    main()
