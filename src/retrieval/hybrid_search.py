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
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name

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

        # Perform hybrid search with RRF fusion
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=top_k,
                        filter=qdrant_filter,
                    ),
                    models.Prefetch(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                        using="sparse",
                        limit=top_k,
                        filter=qdrant_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

        # Convert to SearchResult objects
        search_results: list[SearchResult] = []
        for point in results.points:
            payload = point.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    text=payload.get("text", ""),
                    metadata={
                        k: v for k, v in payload.items() if k != "text"
                    },
                    score=point.score if point.score is not None else 0.0,
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
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=int(value)),
                    )
                )

        if not conditions:
            return None

        return Filter(must=conditions)
