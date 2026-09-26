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

    def generate(self, query: str, chunks: list[dict[str, Any]]) -> tuple[str, float, dict[str, int]]:
        """Direct one-shot LLM call — no CRAG loops, no grading, no hallucination guard."""
        from src.utils.tokens import cost_from_usage

        context = "\n\n---\n\n".join(
            f"[Chunk {i+1} | {c.get('metadata', {}).get('company_ticker', '')} "
            f"FY{c.get('metadata', {}).get('fiscal_year', '')} "
            f"Page {c.get('metadata', {}).get('page_number', 'N/A')}]\n{c['text']}"
            for i, c in enumerate(chunks)
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
            answer = resp.choices[0].message.content or ""
            cost, token_detail = cost_from_usage(resp.usage, self.openai_model)
            return answer, cost, token_detail
        except Exception as e:
            logger.error(f"Naive generation failed: {e}")
            return f"Error: {e}", 0.0, {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    def run(
        self, query: str, ticker: str | None = None
    ) -> dict[str, Any]:
        """Full naive baseline pipeline: single dense retrieval + one-shot generation.

        Returns:
            Dict with keys: answer, chunks, latency_s, cost_usd, token_detail, pipeline.
        """
        t0 = time.perf_counter()
        chunks = self.retrieve(query, ticker=ticker)
        if chunks:
            answer, cost, token_detail = self.generate(query, chunks)
        else:
            answer, cost, token_detail = (
                "Insufficient information in context (no chunks retrieved).",
                0.0,
                {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
            )

        return {
            "answer": answer,
            "chunks": chunks,
            "latency_s": round(time.perf_counter() - t0, 3),
            "cost_usd": cost,
            "token_detail": token_detail,
            "pipeline": "naive_rag",
        }


def main() -> None:
    import argparse
    from config.settings import get_settings
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    parser = argparse.ArgumentParser(description="Run Naive RAG baseline")
    parser.add_argument("query", help="Financial query to answer")
    parser.add_argument("--ticker", default=None, help="Optional ticker filter (e.g. AAPL)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    args = parser.parse_args()

    console = Console()
    settings = get_settings()

    console.print(f"[bold cyan]Initializing Naive RAG Pipeline...[/bold cyan]")
    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = BGEEmbedder(model_name=settings.embedding_model, use_fp16=settings.use_fp16)

    baseline = NaiveRAG(
        qdrant_client=qdrant,
        embedder=embedder,
        collection_name=settings.qdrant_collection,
        openai_model=settings.openai_model,
        top_k=args.top_k,
    )

    console.print(Panel(args.query, title="Baseline Query", border_style="yellow"))
    res = baseline.run(args.query, ticker=args.ticker)

    console.print(Panel(res["answer"], title="Baseline Answer (One-Shot)", border_style="green"))

    # Chunks table
    table = Table(title=f"Retrieved Chunks (Dense-only Top-{len(res['chunks'])})")
    table.add_column("Rank", justify="right")
    table.add_column("Ticker", style="cyan")
    table.add_column("FY", justify="right")
    table.add_column("Page", justify="right")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Preview", style="dim")

    for i, c in enumerate(res["chunks"]):
        meta = c.get("metadata", {})
        table.add_row(
            str(i + 1),
            str(meta.get("company_ticker", "N/A")),
            str(meta.get("fiscal_year", "N/A")),
            str(meta.get("page_number", "N/A")),
            f"{c.get('score', 0.0):.4f}",
            c.get("text", "")[:90].replace("\n", " ") + "...",
        )
    console.print(table)
    console.print(
        f"[dim]Latency: {res['latency_s']}s | Cost: ${res['cost_usd']:.5f} | "
        f"Prompt tokens: {res['token_detail']['prompt_tokens']} | "
        f"Completion tokens: {res['token_detail']['completion_tokens']}[/dim]\n"
    )


if __name__ == "__main__":
    main()
