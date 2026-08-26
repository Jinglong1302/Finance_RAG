"""Cross-encoder reranker using BGE-reranker-large.

Scores and reranks search candidates using a cross-encoder model.
Supports per-sub-query reranking with merge and deduplication.
"""

from __future__ import annotations

from src.retrieval.hybrid_search import SearchResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model.

    Uses FlagEmbedding's FlagReranker for cross-encoder scoring.
    Scores query-document pairs and returns the top-K by score.

    Args:
        model_name: HuggingFace model identifier.
                   Default: "BAAI/bge-reranker-large".
        use_fp16: Whether to use FP16 inference. Default: True.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None

    @property
    def model(self):
        """Lazy-load the reranker model."""
        if self._model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self.model_name, use_fp16=self.use_fp16
            )
            logger.info("Reranker model loaded successfully")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank candidates against a single query.

        Args:
            query: The query text.
            candidates: List of SearchResult from hybrid search.
            top_k: Number of top results to return.

        Returns:
            Top-K SearchResults re-scored and sorted by cross-encoder score.
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        # Create query-document pairs for cross-encoder scoring
        pairs = [[query, result.text] for result in candidates]

        # Score all pairs
        scores = self.model.compute_score(pairs)

        # Handle single result case (compute_score returns float, not list)
        if isinstance(scores, (int, float)):
            scores = [scores]

        # Attach scores and sort
        scored_results: list[tuple[float, SearchResult]] = []
        for score, result in zip(scores, candidates):
            # Create a new SearchResult with the reranker score
            reranked = SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                metadata=result.metadata,
                score=float(score),
            )
            scored_results.append((float(score), reranked))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        top_results = [result for _, result in scored_results[:top_k]]

        logger.info(
            f"Reranked {len(candidates)} → top {len(top_results)} "
            f"(score range: {scored_results[0][0]:.3f} to "
            f"{scored_results[-1][0]:.3f})"
        )
        return top_results

    def rerank_multi_query(
        self,
        sub_queries: list[str],
        candidates_per_query: list[list[SearchResult]],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank per sub-query, then merge and deduplicate.

        Each sub-query's candidates are reranked independently.
        Results are merged, deduplicated by chunk_id (keeping highest
        score), and the global top-K is returned.

        Args:
            sub_queries: List of sub-query strings.
            candidates_per_query: List of candidate lists, one per sub-query.
            top_k: Total number of results to return after merging.

        Returns:
            Merged and deduplicated top-K SearchResults.
        """
        if len(sub_queries) != len(candidates_per_query):
            raise ValueError(
                f"Number of sub-queries ({len(sub_queries)}) must match "
                f"number of candidate lists ({len(candidates_per_query)})"
            )

        # Rerank each sub-query independently
        all_reranked: list[SearchResult] = []
        for query, candidates in zip(sub_queries, candidates_per_query):
            # For multi-query, get more per-query to ensure good coverage
            per_query_k = max(top_k, 5)
            reranked = self.rerank(query, candidates, top_k=per_query_k)
            all_reranked.extend(reranked)

        # Deduplicate by chunk_id, keeping highest score
        best_by_id: dict[str, SearchResult] = {}
        for result in all_reranked:
            existing = best_by_id.get(result.chunk_id)
            if existing is None or result.score > existing.score:
                best_by_id[result.chunk_id] = result

        # Sort by score and take top-K
        deduped = sorted(
            best_by_id.values(), key=lambda r: r.score, reverse=True
        )

        top_results = deduped[:top_k]

        logger.info(
            f"Multi-query rerank: {len(sub_queries)} queries, "
            f"{len(all_reranked)} total candidates → "
            f"{len(best_by_id)} unique → top {len(top_results)}"
        )
        return top_results
