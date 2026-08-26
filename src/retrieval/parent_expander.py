"""Parent chunk expander.

Expands child chunks to their parent context for richer generation input.
Fetches parent chunks from Qdrant by ID (deterministic lookup, no vector search).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.retrieval.hybrid_search import SearchResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnrichedContext:
    """A search result enriched with parent chunk context."""

    child_text: str
    parent_text: str | None
    metadata: dict[str, Any]
    rerank_score: float
    chunk_id: str


class ParentExpander:
    """Expands child chunks to include parent context.

    For each reranked child chunk, fetches its parent chunk text from
    Qdrant using the parent_chunk_id metadata field. Deduplicates
    parents when multiple children share the same parent.

    Args:
        client: QdrantClient instance.
        collection_name: Qdrant collection name.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "sec_filings",
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        # Cache to avoid duplicate fetches
        self._parent_cache: dict[str, str | None] = {}

    def expand(
        self, reranked_results: list[SearchResult]
    ) -> list[EnrichedContext]:
        """Expand reranked results with parent chunk context.

        Args:
            reranked_results: List of reranked SearchResult objects.

        Returns:
            List of EnrichedContext objects with parent text attached.
        """
        self._parent_cache.clear()
        enriched: list[EnrichedContext] = []

        for result in reranked_results:
            parent_id = result.metadata.get("parent_chunk_id")
            parent_text = None

            if parent_id:
                parent_text = self._fetch_parent(parent_id)

            enriched.append(
                EnrichedContext(
                    child_text=result.text,
                    parent_text=parent_text,
                    metadata=result.metadata,
                    rerank_score=result.score,
                    chunk_id=result.chunk_id,
                )
            )

        # Log stats
        parents_found = sum(1 for e in enriched if e.parent_text is not None)
        unique_parents = len(self._parent_cache)
        logger.info(
            f"Parent expansion: {len(enriched)} children, "
            f"{parents_found} with parents, "
            f"{unique_parents} unique parents fetched"
        )

        return enriched

    def _fetch_parent(self, parent_id: str) -> str | None:
        """Fetch parent chunk text from Qdrant by chunk_id.

        Uses a cache to avoid duplicate fetches when multiple children
        share the same parent.

        Args:
            parent_id: The parent_chunk_id to look up.

        Returns:
            Parent chunk text or None if not found.
        """
        # Check cache first
        if parent_id in self._parent_cache:
            return self._parent_cache[parent_id]

        # Fetch from Qdrant
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=MatchValue(value=parent_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )

            if results:
                text = results[0].payload.get("text", "")
                self._parent_cache[parent_id] = text
                return text
            else:
                logger.debug(f"Parent chunk not found: {parent_id}")
                self._parent_cache[parent_id] = None
                return None

        except Exception as e:
            logger.warning(f"Failed to fetch parent {parent_id}: {e}")
            self._parent_cache[parent_id] = None
            return None
