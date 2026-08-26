"""LangGraph state definition for the CRAG pipeline.

Defines the TypedDict that flows through all nodes in the
LangGraph state machine, carrying query data, retrieval results,
grading decisions, and generation output.
"""

from __future__ import annotations

from typing import Any, TypedDict


class CRAGState(TypedDict, total=False):
    """State object for the Corrective RAG pipeline.

    Flows through all LangGraph nodes. Each node reads and writes
    specific fields. total=False allows partial initialization.
    """

    # === Input ===
    original_query: str

    # === Query Processing ===
    sub_queries: list[str]
    structured_filters: dict[str, Any]
    query_type: str  # "factual_numeric" | "conceptual" | "comparative"
    result_count: int  # Adaptive: 5 or 8

    # === Retrieval ===
    search_results: list[dict[str, Any]]  # Raw hybrid search results
    reranked_results: list[dict[str, Any]]  # After cross-encoder
    enriched_contexts: list[dict[str, Any]]  # After parent + note expansion

    # === CRAG Grading ===
    grading_results: list[dict[str, Any]]  # Per-chunk relevance grades
    cycle_count: int  # 0, 1, or 2 (max)
    confidence: str  # "high" | "medium" | "low" | "insufficient"
    crag_action: str  # "generate" | "rewrite" | "fallback" | "refuse"
    rewritten_query: str | None

    # === Generation ===
    generation: str | None  # Final answer text
    citations: list[dict[str, Any]]  # Structured citations
    structured_metrics: list[dict[str, Any]] | None  # Optional structured output

    # === Guardrails ===
    hallucination_check: str  # "pass" | "fail"
    final_answer: str | None  # Post-guardrail answer

    # === Cost ===
    cost_accumulated: float  # Running cost in USD

    # === Pipeline Trace (per-node diagnostics for reporting) ===
    pipeline_trace: list[dict]  # List of per-node trace entries

    # === Error ===
    error: str | None  # Error message if pipeline fails
