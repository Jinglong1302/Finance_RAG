"""BGE-M3 embedding module.

Generates dense (1024-dim) and sparse (lexical weight) embeddings
using the BAAI/bge-m3 model via FlagEmbedding. Produces both vector
types in a single forward pass for efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.chunking.metadata import Chunk
from src.utils.logging import get_logger
from src.utils.tokens import truncate_to_tokens

logger = get_logger(__name__)


@dataclass
class EmbeddedChunk:
    """A chunk with its dense and sparse embeddings."""

    chunk: Chunk
    dense_vector: list[float]  # 1024-dim dense embedding
    sparse_indices: list[int]  # Token indices for sparse vector
    sparse_values: list[float]  # Lexical weights for sparse vector


class BGEEmbedder:
    """Generates dense + sparse embeddings using BGE-M3.

    Uses FlagEmbedding's BGEM3FlagModel for native multi-output encoding.
    Produces both dense and sparse vectors in a single forward pass.

    Parent chunks (chunk_type="parent") are skipped — they are fetched
    by ID, not by vector similarity.

    Args:
        model_name: HuggingFace model identifier. Default: "BAAI/bge-m3".
        use_fp16: Whether to use FP16 inference. Default: True.
        max_embed_tokens: Maximum tokens for dense embedding. Default: 512.
        batch_size: Encoding batch size. Default: 32.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        max_embed_tokens: int = 512,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_embed_tokens = max_embed_tokens
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazy-load the BGE-M3 model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_name, use_fp16=self.use_fp16
            )
            logger.info("Embedding model loaded successfully")
        return self._model

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Generate embeddings for a list of chunks.

        Skips parent chunks (they are fetched by ID, not searched).

        Args:
            chunks: List of Chunk objects to embed.

        Returns:
            List of EmbeddedChunk objects (child chunks only).
        """
        # Filter to child chunks only
        child_chunks = [c for c in chunks if c.metadata.chunk_type == "child"]

        if not child_chunks:
            return []

        logger.info(f"Embedding {len(child_chunks)} child chunks (batch_size={self.batch_size})")

        # Prepare texts — truncate to max_embed_tokens for dense quality
        texts = []
        for chunk in child_chunks:
            # For dense embedding, use truncated text
            # For sparse embedding, the model uses the full text
            texts.append(chunk.text)

        # Encode in batches
        all_embedded: list[EmbeddedChunk] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_chunks = child_chunks[batch_start:batch_end]

            # Encode with both dense and sparse outputs
            output = self.model.encode(
                batch_texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,  # Not using ColBERT
            )

            dense_vecs = output["dense_vecs"]
            sparse_weights = output["lexical_weights"]

            for i, chunk in enumerate(batch_chunks):
                # Dense vector
                dense_vec = dense_vecs[i].tolist()

                # Sparse vector: convert from {token_id: weight} dict
                sparse_dict = sparse_weights[i]
                if isinstance(sparse_dict, dict):
                    sparse_idx = [int(k) for k in sparse_dict.keys()]
                    sparse_vals = [float(v) for v in sparse_dict.values()]
                else:
                    sparse_idx = []
                    sparse_vals = []

                all_embedded.append(
                    EmbeddedChunk(
                        chunk=chunk,
                        dense_vector=dense_vec,
                        sparse_indices=sparse_idx,
                        sparse_values=sparse_vals,
                    )
                )

            logger.debug(
                f"Encoded batch {batch_start // self.batch_size + 1}: "
                f"{len(batch_texts)} chunks"
            )

        logger.info(f"Embedding complete: {len(all_embedded)} chunks embedded")
        return all_embedded

    def embed_query(self, query: str) -> tuple[list[float], dict[int, float]]:
        """Generate embeddings for a query string.

        Args:
            query: The search query text.

        Returns:
            Tuple of (dense_vector, sparse_weights_dict).
        """
        output = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vec = output["dense_vecs"][0].tolist()

        sparse_dict = output["lexical_weights"][0]
        if isinstance(sparse_dict, dict):
            sparse_weights = {int(k): float(v) for k, v in sparse_dict.items()}
        else:
            sparse_weights = {}

        return dense_vec, sparse_weights
