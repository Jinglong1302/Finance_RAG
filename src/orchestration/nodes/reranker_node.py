"""Reranker node for the CRAG pipeline.

Applies cross-encoder reranking to search results and expands
parent context.
"""

from __future__ import annotations

import time
from typing import Any

from src.orchestration.state import CRAGState
from src.retrieval.hybrid_search import SearchResult
from src.retrieval.parent_expander import ParentExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_reranker_node(
    reranker: CrossEncoderReranker | None,
    expander: ParentExpander,
):
    """Factory to create a reranker node with injected dependencies.

    Args:
        reranker: CrossEncoderReranker instance, or None to skip reranking.
        expander: ParentExpander instance.

    Returns:
        A node function compatible with LangGraph.
    """

    def reranker_node(state: CRAGState) -> dict[str, Any]:
        """Rerank search results and expand parent context.

        Args:
            state: Current CRAG pipeline state.

        Returns:
            State update with reranked_results and enriched_contexts.
        """
        search_results = state.get("search_results", [])
        if not search_results:
            return {
                "reranked_results": [],
                "enriched_contexts": [],
            }

        start_t = time.perf_counter()

        # Convert dicts back to SearchResult objects
        candidates = [
            SearchResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                metadata=r["metadata"],
                score=r["score"],
            )
            for r in search_results
        ]

        # Determine top-K
        top_k = state.get("result_count", 5)
        query = state.get("rewritten_query") or state.get("original_query", "")

        # Rerank if model provided, otherwise preserve hybrid rank order
        if reranker is not None:
            reranked = reranker.rerank(query, candidates, top_k=top_k)
        else:
            reranked = candidates[:top_k]

        # Convert to dicts for state
        reranked_dicts = [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "metadata": r.metadata,
                "score": r.score,
            }
            for r in reranked
        ]

        # Expand parent context
        enriched = expander.expand(reranked)
        enriched_dicts = [
            {
                "child_text": e.child_text,
                "parent_text": e.parent_text,
                "metadata": e.metadata,
                "rerank_score": e.rerank_score,
                "chunk_id": e.chunk_id,
            }
            for e in enriched
        ]

        logger.info(
            f"Reranked to top-{len(reranked)}, "
            f"expanded {sum(1 for e in enriched if e.parent_text)} parents"
        )

        duration_s = round(time.perf_counter() - start_t, 3)

        # Build trace: top chunks with scores, section labels, full text, and parent text
        chunk_trace = [
            {
                "rank": i + 1,
                "chunk_id": r.chunk_id,
                "section": r.metadata.get("section_title", r.metadata.get("section_id", "?")),
                "fiscal_year": r.metadata.get("fiscal_year", "?"),
                "company": r.metadata.get("company_ticker", "?"),
                "rerank_score": round(r.score, 4),
                "text_preview": r.text[:140].replace("\n", " "),
                "text_full": r.text,
                "parent_text": enriched_dicts[i].get("parent_text") if i < len(enriched_dicts) else None,
            }
            for i, r in enumerate(reranked)
        ]
        trace = {
            "node": "reranker",
            "candidates_in": len(candidates),
            "top_k": len(reranked),
            "chunks": chunk_trace,
            "duration_s": duration_s,
        }
        prev_trace = state.get("pipeline_trace") or []

        return {
            "reranked_results": reranked_dicts,
            "enriched_contexts": enriched_dicts,
            "pipeline_trace": prev_trace + [trace],
        }

    return reranker_node
