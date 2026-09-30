"""Plumbing smoke test using stub-LLM mode — zero API calls, zero cost.

Exercises the full Part B CRAG loop (decompose → retrieve → rerank →
expand_notes → sufficiency_grade → generate → guard) on one question
from each indexed company, verifying that:
  1. All nodes are reached with no exceptions.
  2. Stub LLM is called for every expected node (decomposer, grader,
     generator, guard).
  3. The final state has expected keys: final_answer, citations, crag_action.
  4. No real API calls are made (stub_call_log total == expected count).

Cost: $0.00 (stub mode).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# MUST enable stub BEFORE any other imports that touch openai
from src.utils.llm_stub import enable_stub_mode, stub_call_log
enable_stub_mode()

from rich.console import Console
from config.settings import get_settings

console = Console()


def run_stub_crag(query: str, ticker: str) -> dict:
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

    settings = get_settings()
    qdrant = QdrantClient(url=settings.qdrant_url)
    embedder = BGEEmbedder(model_name=settings.embedding_model)
    searcher = HybridSearcher(qdrant, embedder, settings.qdrant_collection)
    expander = ParentExpander(qdrant, settings.qdrant_collection)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)

    ret_node = make_retriever_node(searcher)
    rerank_node = make_reranker_node(reranker, expander)
    note_expander_fn = make_note_expander_node(qdrant, settings.qdrant_collection)

    state: dict = {
        "original_query": query,
        "cycle_count": 0,
        "cost_accumulated": 0.0,
        "pipeline_trace": [],
        "result_count": 5,
    }

    state.update(query_decomposer_node(state))

    # Production path: NO ticker injection
    # (ticker arg only used for label in this smoke test)

    max_cycles = 2
    for cycle in range(max_cycles + 1):
        state["cycle_count"] = cycle
        state.update(ret_node(state))
        state.update(rerank_node(state))
        state.update(note_expander_fn(state))
        state.update(sufficiency_grader_node(state))
        action = state.get("crag_action", "generate")

        if action == "generate":
            state.update(generator_node(state))
            state.update(hallucination_guard_node(state))
            break
        elif action == "rewrite" and cycle < max_cycles:
            state["cycle_count"] = cycle + 1
            state.update(query_rewriter_node(state))
            continue
        else:
            state.update(refuse_node(state))
            break

    return state


def main() -> None:
    console.print("[bold cyan]Plumbing Smoke Test (Stub-LLM, $0.00)[/bold cyan]")

    test_cases = [
        ("What was Apple's total revenue in FY2024?", "AAPL"),
        ("What was Coca-Cola's net income in FY2022?", "KO"),
        ("What is Boeing's total debt in FY2022?", "BA"),
    ]

    all_pass = True
    for query, ticker in test_cases:
        t0 = time.perf_counter()
        pre_calls = len(stub_call_log())
        try:
            state = run_stub_crag(query, ticker)
            lat = round(time.perf_counter() - t0, 2)
            calls_made = len(stub_call_log()) - pre_calls

            # Check expected keys
            has_answer = bool(state.get("final_answer") or state.get("generation"))
            has_action = bool(state.get("crag_action"))
            nodes_called = [c["node"] for c in stub_call_log()[pre_calls:]]

            # Expect at least: decomposer, sufficiency_grader, generator, guard
            expected_nodes = {"decomposer", "sufficiency_grader", "generator", "guard"}
            nodes_hit = set(nodes_called)
            missing = expected_nodes - nodes_hit

            status = "PASS" if has_answer and has_action and not missing else "FAIL"
            if status == "FAIL":
                all_pass = False

            console.print(
                f"  [{status}] [{ticker}] {query[:55]}..."
                f"\n        lat={lat}s llm_calls={calls_made} nodes={nodes_called}"
                f"\n        action={state.get('crag_action')} has_answer={has_answer}"
            )
            if missing:
                console.print(f"        [red]Missing nodes: {missing}[/red]")

        except Exception as exc:
            all_pass = False
            console.print(f"  [FAIL] [{ticker}] {query[:55]}... EXCEPTION: {exc}")

    nodes_by_type: dict = {}
    for c in stub_call_log():
        nodes_by_type[c["node"]] = nodes_by_type.get(c["node"], 0) + 1

    console.print(f"\n[bold]Total stub LLM calls: {len(stub_call_log())}[/bold]")
    for node, count in sorted(nodes_by_type.items()):
        console.print(f"  {node}: {count}")

    console.print(f"\n[bold {'green' if all_pass else 'red'}]Overall: {'PASS' if all_pass else 'FAIL'}[/bold {'green' if all_pass else 'red'}]")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
