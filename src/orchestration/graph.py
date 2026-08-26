"""LangGraph StateGraph assembly for the CRAG pipeline.

Wires all nodes together with conditional routing to implement
the Corrective RAG state machine with self-correction loops.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.embedding.embedder import BGEEmbedder
from src.orchestration.nodes.generator import generator_node
from src.orchestration.nodes.grader import grader_node
from src.orchestration.nodes.hallucination_guard import hallucination_guard_node
from src.orchestration.nodes.note_expander import make_note_expander_node
from src.orchestration.nodes.query_decomposer import query_decomposer_node
from src.orchestration.nodes.query_rewriter import query_rewriter_node
from src.orchestration.nodes.reranker_node import make_reranker_node
from src.orchestration.nodes.retriever import make_retriever_node
from src.orchestration.state import CRAGState
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.parent_expander import ParentExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.logging import get_logger

logger = get_logger(__name__)


def route_after_grading(state: CRAGState) -> str:
    """Route based on CRAG grading results.

    Routes to:
    - "generate": if context is sufficient
    - "rewrite": if context is insufficient and cycles remain
    - "refuse": if context is insufficient and cycles exhausted

    Args:
        state: Current pipeline state.

    Returns:
        Next node name.
    """
    action = state.get("crag_action", "generate")

    if action == "generate":
        return "generate"
    elif action == "rewrite":
        return "rewrite"
    else:
        return "refuse"


def route_after_guard(state: CRAGState) -> str:
    """Route based on hallucination guard results.

    Routes to:
    - END: if hallucination check passed
    - "rewrite": if hallucination detected and cycles remain

    Args:
        state: Current pipeline state.

    Returns:
        Next node name or END.
    """
    check = state.get("hallucination_check", "pass")
    cycle_count = state.get("cycle_count", 0)

    if check == "pass" or check == "error":
        return "end"
    elif cycle_count < 2:
        return "rewrite"
    else:
        # Max cycles reached, accept with warnings
        return "end"


def refuse_node(state: CRAGState) -> dict[str, Any]:
    """Generate a graceful refusal response.

    Args:
        state: Current pipeline state.

    Returns:
        State update with refusal answer.
    """
    query = state.get("original_query", "")
    enriched = state.get("enriched_contexts", [])
    filters = state.get("structured_filters", {})

    # Build partial evidence summary
    partial_evidence = []
    for ctx in enriched[:3]:
        text = ctx.get("child_text", "")
        if text:
            partial_evidence.append(text[:200] + "...")

    ticker = filters.get("company_ticker", "the company")

    answer = (
        f"I could not find sufficient evidence in the available SEC filings "
        f"to answer this question.\n\n"
    )

    if partial_evidence:
        answer += "**Partial evidence found:**\n"
        for i, evidence in enumerate(partial_evidence):
            answer += f"- {evidence}\n"
        answer += "\n"

    answer += (
        f"**Suggested actions:**\n"
        f"- Check the original filing on SEC EDGAR: "
        f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
        f"&company={ticker}&type=10-K\n"
        f"- The data may be in a filing not included in the current corpus.\n"
    )

    return {
        "final_answer": answer,
        "confidence": "insufficient",
        "generation": answer,
    }


def build_crag_graph(
    searcher: HybridSearcher,
    reranker: CrossEncoderReranker,
    expander: ParentExpander,
    qdrant_client: Any,
    collection_name: str = "sec_filings",
) -> Any:
    """Build and compile the CRAG LangGraph pipeline.

    Creates a StateGraph with all nodes and conditional edges
    implementing the Corrective RAG pattern with self-correction.

    Graph topology:
    ```
    START → decompose → retrieve → rerank → expand_notes → grade
        grade →[generate]→ generate → guard →[pass]→ END
        grade →[rewrite]→ rewrite → retrieve (loop)
        grade →[refuse]→ refuse → END
        guard →[fail]→ rewrite → retrieve (loop)
    ```

    Args:
        searcher: HybridSearcher instance.
        reranker: CrossEncoderReranker instance.
        expander: ParentExpander instance.
        qdrant_client: QdrantClient for note expansion.
        collection_name: Qdrant collection name.

    Returns:
        Compiled LangGraph runnable.
    """
    # Create node functions with injected dependencies
    retriever = make_retriever_node(searcher)
    reranker_node = make_reranker_node(reranker, expander)
    note_expander = make_note_expander_node(qdrant_client, collection_name)

    # Build the graph
    workflow = StateGraph(CRAGState)

    # Add nodes
    workflow.add_node("decompose", query_decomposer_node)
    workflow.add_node("retrieve", retriever)
    workflow.add_node("rerank", reranker_node)
    workflow.add_node("expand_notes", note_expander)
    workflow.add_node("grade", grader_node)
    workflow.add_node("generate", generator_node)
    workflow.add_node("guard", hallucination_guard_node)
    workflow.add_node("rewrite", query_rewriter_node)
    workflow.add_node("refuse", refuse_node)

    # Wire edges — linear flow
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "expand_notes")
    workflow.add_edge("expand_notes", "grade")

    # Conditional: after grading
    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "refuse": "refuse",
        },
    )

    # Generation → Guard
    workflow.add_edge("generate", "guard")

    # Conditional: after guard
    workflow.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "end": END,
            "rewrite": "rewrite",
        },
    )

    # Rewrite → Retrieve (CRAG loop)
    workflow.add_edge("rewrite", "retrieve")

    # Refuse → END
    workflow.add_edge("refuse", END)

    # Compile
    compiled = workflow.compile()
    logger.info("CRAG graph compiled successfully")

    return compiled


def run_query(graph: Any, query: str) -> dict[str, Any]:
    """Run a query through the compiled CRAG graph.

    Args:
        graph: Compiled LangGraph runnable.
        query: The user's financial question.

    Returns:
        Final state dict with answer, citations, and metadata.
    """
    initial_state: CRAGState = {
        "original_query": query,
        "cycle_count": 0,
        "cost_accumulated": 0.0,
        "pipeline_trace": [],
    }

    logger.info(f"Running CRAG pipeline: '{query[:80]}...'")

    # Execute the graph
    result = graph.invoke(initial_state)

    # Log summary
    final_answer = result.get("final_answer") or result.get("generation", "")
    confidence = result.get("confidence", "unknown")
    cycles = result.get("cycle_count", 0)
    cost = result.get("cost_accumulated", 0.0)

    logger.info(
        f"Pipeline complete: confidence={confidence}, "
        f"cycles={cycles}, cost=${cost:.4f}, "
        f"answer_length={len(final_answer)}"
    )

    return result
