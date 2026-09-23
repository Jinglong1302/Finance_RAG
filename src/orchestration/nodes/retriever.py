"""Retriever node for the CRAG pipeline.

Performs hybrid search using the HybridSearcher, handling both
single-query and multi-query (decomposed) retrieval.
"""

from __future__ import annotations

from typing import Any

from src.orchestration.state import CRAGState
from src.retrieval.hybrid_search import HybridSearcher
from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_retriever_node(searcher: HybridSearcher):
    """Factory function to create a retriever node with injected dependencies.

    Args:
        searcher: HybridSearcher instance.

    Returns:
        A node function compatible with LangGraph.
    """

    def retriever_node(state: CRAGState) -> dict[str, Any]:
        """Perform hybrid search for all sub-queries.

        Uses the rewritten query if available (CRAG retry),
        otherwise uses the decomposed sub-queries.

        Args:
            state: Current CRAG pipeline state.

        Returns:
            State update with search_results.
        """
        # Use rewritten query if this is a CRAG retry
        rewritten = state.get("rewritten_query")
        if rewritten:
            queries = [rewritten]
        else:
            queries = state.get("sub_queries", [state.get("original_query", "")])

        filters = state.get("structured_filters", {})
        # Always fetch a broad candidate pool (top-25) for the reranker to refine
        top_k = 25

        # Perform search for each sub-query
        all_results = []
        for query in queries:
            results = searcher.search(
                query=query,
                filters=filters if filters else None,
                top_k=top_k,
            )
            # Fallback: if filtered search yields 0 results and fiscal_year was filtered,
            # retry without fiscal_year to avoid blocking relevant comparative disclosures.
            if not results and filters and "fiscal_year" in filters:
                relaxed_filters = {k: v for k, v in filters.items() if k != "fiscal_year"}
                logger.info(
                    f"0 results with fiscal_year filter. Retrying with relaxed filters: {relaxed_filters}"
                )
                results = searcher.search(
                    query=query,
                    filters=relaxed_filters if relaxed_filters else None,
                    top_k=top_k,
                )
            for r in results:
                all_results.append({
                    "chunk_id": r.chunk_id,
                    "text": r.text,
                    "metadata": r.metadata,
                    "score": r.score,
                })

        # Deduplicate by chunk_id (keep highest score)
        seen: dict[str, dict] = {}
        for result in all_results:
            cid = result["chunk_id"]
            if cid not in seen or result["score"] > seen[cid]["score"]:
                seen[cid] = result

        deduped = sorted(seen.values(), key=lambda r: r["score"], reverse=True)

        logger.info(
            f"Retrieved {len(all_results)} raw results → "
            f"{len(deduped)} unique chunks"
        )

        trace = {
            "node": "retriever",
            "queries": queries,
            "filters": filters,
            "raw_results": len(all_results),
            "unique_results": len(deduped),
            "top_scores": [round(r["score"], 4) for r in deduped[:5]],
        }
        prev_trace = state.get("pipeline_trace") or []

        return {"search_results": deduped, "pipeline_trace": prev_trace + [trace]}

    return retriever_node
