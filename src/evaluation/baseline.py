"""Naive RAG baseline for comparison against the CRAG pipeline.

Dense-only Qdrant search (no sparse/RRF), no reranking, no query
decomposition, no CRAG correction loops — a single vector search
followed by a direct GPT call.
"""
from __future__ import annotations

import time
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.embedding.embedder import BGEEmbedder
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the question based ONLY on "
    "the provided context from SEC 10-K filings. If the context does not "
    "contain enough information, say 'Insufficient information in context.'"
)


class NaiveRAG:
    """Single-step dense-only retrieval + LLM generation baseline.

    Args:
        qdrant_client: Active QdrantClient instance.
        embedder: BGEEmbedder (uses only dense vector).
        collection_name: Qdrant collection to search.
        openai_model: OpenAI model for generation.
        top_k: Number of chunks to retrieve (no reranking applied).
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedder: BGEEmbedder,
        collection_name: str,
        openai_model: str = "gpt-4o",
        top_k: int = 5,
    ) -> None:
        self.client = qdrant_client
        self.embedder = embedder
        self.collection_name = collection_name
        self.openai_model = openai_model
        self.top_k = top_k
        self._llm: OpenAI | None = None

    @property
    def llm(self) -> OpenAI:
        if self._llm is None:
            self._llm = OpenAI()
        return self._llm

    def retrieve(
        self, query: str, ticker: str | None = None
    ) -> list[dict[str, Any]]:
        """Dense-only retrieval — no sparse, no reranking."""
        dense_vector, _ = self.embedder.embed_query(query)

        # Optionally filter by ticker (child chunks only)
        must_conditions: list[FieldCondition] = [
            FieldCondition(key="chunk_type", match=MatchValue(value="child"))
        ]
        if ticker:
            must_conditions.append(
                FieldCondition(
                    key="company_ticker", match=MatchValue(value=ticker)
                )
            )
        qdrant_filter = Filter(must=must_conditions)

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                query_filter=qdrant_filter,
                limit=self.top_k,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Naive dense search failed: {e}")
            return []

        return [
            {
                "text": (p.payload or {}).get("text", ""),
                "metadata": {k: v for k, v in (p.payload or {}).items() if k != "text"},
                "score": p.score,
            }
            for p in results.points
        ]

    def generate(self, query: str, chunks: list[dict[str, Any]]) -> str:
        """Direct LLM call — no CRAG loop, no hallucination guard."""
        context = "\n\n---\n\n".join(
            f"[Chunk {i+1}]\n{c['text']}" for i, c in enumerate(chunks)
        )
        try:
            resp = self.llm.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    },
                ],
                temperature=0,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Naive generation failed: {e}")
            return f"Error: {e}"

    def run(
        self, query: str, ticker: str | None = None
    ) -> dict[str, Any]:
        """Full naive pipeline: retrieve then generate.

        Returns:
            Dict with keys: answer, chunks, latency_s.
        """
        t0 = time.perf_counter()
        chunks = self.retrieve(query, ticker=ticker)
        answer = self.generate(query, chunks) if chunks else "No results retrieved."
        return {
            "answer": answer,
            "chunks": chunks,
            "latency_s": round(time.perf_counter() - t0, 3),
        }
