"""Qdrant indexer for storing embedded chunks.

Creates the hybrid collection (dense + sparse vectors), sets up
payload indexes for pre-filtering, and handles batched upserts
of embedded chunks and parent chunks.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.chunking.metadata import Chunk
from src.embedding.embedder import EmbeddedChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Fields to create Qdrant payload indexes on (for pre-filtering)
_INDEXED_FIELDS = {
    "company_ticker": PayloadSchemaType.KEYWORD,
    "filing_type": PayloadSchemaType.KEYWORD,
    "fiscal_year": PayloadSchemaType.INTEGER,
    "section": PayloadSchemaType.KEYWORD,
    "chunk_type": PayloadSchemaType.KEYWORD,
}


class QdrantIndexer:
    """Manages Qdrant collection creation, indexing, and upserts.

    Creates a hybrid collection with:
    - Dense vectors: "dense" (1024-dim, Cosine distance)
    - Sparse vectors: "sparse" (learned lexical weights from BGE-M3)
    - Payload indexes on 5 filter fields

    Args:
        url: Qdrant server URL.
        collection_name: Name of the collection.
        api_key: Optional Qdrant Cloud API key.
        embedding_dim: Dense vector dimension. Default: 1024.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "sec_filings",
        api_key: str | None = None,
        embedding_dim: int = 1024,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=60,
        )

    def create_collection(self, recreate: bool = False) -> None:
        """Create the hybrid vector collection in Qdrant.

        Args:
            recreate: If True, delete and recreate the collection.
                     Default: False.
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and not recreate:
            logger.info(
                f"Collection '{self.collection_name}' already exists. Skipping creation."
            )
            return

        if exists and recreate:
            logger.warning(
                f"Recreating collection '{self.collection_name}'"
            )
            self.client.delete_collection(self.collection_name)

        # Create collection with dense + sparse vectors
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                ),
            },
        )

        # Create payload indexes for pre-filtering
        self._create_payload_indexes()

        logger.info(
            f"Created collection '{self.collection_name}' "
            f"(dense={self.embedding_dim}d, sparse, "
            f"{len(_INDEXED_FIELDS)} payload indexes)"
        )

    def _create_payload_indexes(self) -> None:
        """Create payload indexes on filter fields for efficient pre-filtering."""
        for field_name, schema_type in _INDEXED_FIELDS.items():
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            logger.debug(f"Created payload index: {field_name} ({schema_type})")

    def upsert_chunks(
        self,
        embedded_chunks: list[EmbeddedChunk],
        batch_size: int = 100,
    ) -> None:
        """Upsert embedded child chunks with dense and sparse vectors.

        Args:
            embedded_chunks: List of EmbeddedChunk objects to upsert.
            batch_size: Number of points per upsert batch. Default: 100.
        """
        if not embedded_chunks:
            return

        points: list[PointStruct] = []
        for ec in embedded_chunks:
            # Build payload from chunk metadata
            payload = ec.chunk.metadata.model_dump()
            payload["text"] = ec.chunk.text  # Store full text in payload

            point = PointStruct(
                id=self._generate_point_id(ec.chunk.chunk_id),
                vector={
                    "dense": ec.dense_vector,
                    "sparse": SparseVector(
                        indices=ec.sparse_indices,
                        values=ec.sparse_values,
                    ),
                },
                payload=payload,
            )
            points.append(point)

        # Batch upsert
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start : batch_start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            logger.debug(
                f"Upserted batch {batch_start // batch_size + 1}: "
                f"{len(batch)} points"
            )

        logger.info(f"Upserted {len(points)} embedded chunks")

    def upsert_parent_chunks(
        self,
        parent_chunks: list[Chunk],
        batch_size: int = 100,
    ) -> None:
        """Upsert parent chunks as payload-only points (no vectors).

        Parent chunks are stored for ID-based retrieval during parent
        expansion, not for vector search. They use zero vectors to
        satisfy Qdrant's schema requirements.

        Args:
            parent_chunks: List of parent Chunk objects.
            batch_size: Batch size for upserts.
        """
        if not parent_chunks:
            return

        points: list[PointStruct] = []
        for chunk in parent_chunks:
            payload = chunk.metadata.model_dump()
            payload["text"] = chunk.text

            # Use zero vectors (parents are not searchable)
            point = PointStruct(
                id=self._generate_point_id(chunk.chunk_id),
                vector={
                    "dense": [0.0] * self.embedding_dim,
                    "sparse": SparseVector(indices=[], values=[]),
                },
                payload=payload,
            )
            points.append(point)

        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start : batch_start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Upserted {len(points)} parent chunks (payload-only)")

    def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a chunk by its chunk_id from Qdrant.

        Uses payload filtering to find the chunk by its chunk_id field.

        Args:
            chunk_id: The unique chunk identifier.

        Returns:
            Dict with 'text' and metadata fields, or None if not found.
        """
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="chunk_id",
                        match=MatchValue(value=chunk_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )

        points, _ = results
        if points:
            return points[0].payload
        return None

    def get_collection_info(self) -> dict[str, Any]:
        """Get collection information and statistics.

        Returns:
            Dict with collection stats.
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count or 0,
            # vectors_count removed in Qdrant client v1.9+; use indexed_vectors_count
            "vectors_count": getattr(info, "vectors_count", None)
                or getattr(info, "indexed_vectors_count", 0),
            "status": str(info.status),
        }

    def _generate_point_id(self, chunk_id: str) -> str:
        """Generate a deterministic UUID from a chunk_id string.

        Uses UUID5 with a namespace to ensure the same chunk_id
        always maps to the same point ID.

        Args:
            chunk_id: The chunk identifier string.

        Returns:
            UUID string for use as Qdrant point ID.
        """
        namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        return str(uuid.uuid5(namespace, chunk_id))
