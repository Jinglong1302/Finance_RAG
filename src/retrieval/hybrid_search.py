"""Hybrid search module using Qdrant.

Performs hybrid dense+sparse search with Reciprocal Rank Fusion (RRF)
and metadata pre-filtering. Uses Qdrant's query_points API with
prefetch for server-side fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    SparseVector,
)

from src.embedding.embedder import BGEEmbedder
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result from hybrid retrieval."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float  # RRF fusion score


class HybridSearcher:
    """Performs hybrid dense+sparse search on Qdrant.

    Uses Qdrant's query_points API with prefetch to run dense and
    sparse searches independently, then fuses results using
    Reciprocal Rank Fusion (RRF) server-side.

    Supports metadata pre-filtering to scope searches to specific
    companies, fiscal years, or filing sections.

    Args:
        client: QdrantClient instance.
        embedder: BGEEmbedder for query encoding.
        collection_name: Qdrant collection name.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: BGEEmbedder,
        collection_name: str = "sec_filings",
        dense_weight: float = 1.0,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 25,
    ) -> list[SearchResult]:
        """Perform hybrid search with optional pre-filtering.

        Args:
            query: The search query text.
            filters: Optional metadata filters to apply. Supported keys:
                - company_ticker: str
                - fiscal_year: int
                - filing_type: str
                - section: str
            top_k: Number of results to return. Default: 25.

        Returns:
            List of SearchResult objects, sorted by RRF score descending.
        """
        # Generate query embeddings
        dense_vector, sparse_weights = self.embedder.embed_query(query)

        # Build Qdrant filter
        qdrant_filter = self._build_qdrant_filter(filters)

        # Always exclude parent chunks from search results
        parent_exclusion = FieldCondition(
            key="chunk_type",
            match=MatchValue(value="child"),
        )
        if qdrant_filter is None:
            qdrant_filter = Filter(must=[parent_exclusion])
        else:
            qdrant_filter.must = qdrant_filter.must or []
            qdrant_filter.must.append(parent_exclusion)

        # Build sparse vector
        sparse_indices = list(sparse_weights.keys())
        sparse_values = list(sparse_weights.values())

        # Retrieve candidates independently from dense and sparse streams
        fetch_limit = max(top_k * 2, 50)
        try:
            dense_points = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                limit=fetch_limit,
                query_filter=qdrant_filter,
                with_payload=True,
            ).points

            sparse_points = []
            if sparse_indices:
                sparse_points = self.client.query_points(
                    collection_name=self.collection_name,
                    query=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=fetch_limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                ).points
        except Exception as e:
            logger.error(f"Hybrid search queries failed: {e}")
            return []

        # Client-side weighted RRF fusion (default: dense=1.0, sparse=0.3, k=60)
        scores: dict[Any, float] = {}
        point_map: dict[Any, Any] = {}

        for rank, p in enumerate(dense_points):
            pid = p.id
            point_map[pid] = p
            scores[pid] = scores.get(pid, 0.0) + self.dense_weight * (1.0 / (self.rrf_k + rank + 1))

        for rank, p in enumerate(sparse_points):
            pid = p.id
            point_map[pid] = p
            scores[pid] = scores.get(pid, 0.0) + self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))

        sorted_pids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)
        top_pids = sorted_pids[:top_k]

        # Convert to SearchResult objects
        search_results: list[SearchResult] = []
        for pid in top_pids:
            point = point_map[pid]
            payload = point.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    text=payload.get("text", ""),
                    metadata={
                        k: v for k, v in payload.items() if k != "text"
                    },
                    score=scores[pid],
                )
            )

        logger.info(
            f"Hybrid search: query='{query[:50]}...' "
            f"filters={filters} "
            f"results={len(search_results)}"
        )
        return search_results

    def search_multi_query(
        self,
        queries: list[str],
        filters_per_query: list[dict[str, Any] | None] | None = None,
        top_k: int = 25,
    ) -> list[list[SearchResult]]:
        """Perform hybrid search for multiple sub-queries.

        Args:
            queries: List of query strings.
            filters_per_query: Optional list of filter dicts, one per query.
                             If None, uses the same (no) filter for all.
            top_k: Number of results per query.

        Returns:
            List of SearchResult lists, one per query.
        """
        if filters_per_query is None:
            filters_per_query = [None] * len(queries)

        all_results: list[list[SearchResult]] = []
        for query, filters in zip(queries, filters_per_query):
            results = self.search(query, filters, top_k)
            all_results.append(results)

        return all_results

    def _build_qdrant_filter(
        self, filters: dict[str, Any] | None
    ) -> Filter | None:
        """Build a Qdrant Filter from a metadata filter dict.

        Args:
            filters: Dict of field_name -> value pairs.

        Returns:
            Qdrant Filter object or None if no filters specified.
        """
        if not filters:
            return None

        conditions: list[FieldCondition] = []

        for field, value in filters.items():
            if value is None:
                continue

            if field in (
                "company_ticker",
                "filing_type",
                "section",
                "chunk_type",
            ):
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=str(value)),
                    )
                )
            elif field == "fiscal_year":
                if isinstance(value, (list, tuple)):
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchAny(any=[int(v) for v in value]),
                        )
                    )
                else:
                    y = int(value)
                    # SEC 10-K filings contain 3-year comparative financial statements
                    # (Y, Y-1, Y-2), so year Y data can appear in filing Y, Y+1, or Y+2.
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchAny(any=[y, y + 1, y + 2]),
                        )
                    )

        if not conditions:
            return None

        return Filter(must=conditions)
