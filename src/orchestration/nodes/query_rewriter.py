"""Query rewriter node for CRAG correction cycles."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.orchestration.prompts.rewriting import (
    REWRITING_SYSTEM_PROMPT,
    REWRITING_USER_PROMPT,
)
from src.orchestration.state import CRAGState
from src.utils.logging import get_logger
from src.utils.tokens import estimate_cost

logger = get_logger(__name__)


def query_rewriter_node(state: CRAGState) -> dict[str, Any]:
    """Rewrite the query for improved retrieval in CRAG correction cycles.

    On Cycle 1: Rewrites the query with alternative terminology.
    On Cycle 2: Also relaxes metadata filters (removes fiscal_year, broadens section).

    Args:
        state: Current CRAG pipeline state.

    Returns:
        State update with rewritten_query, incremented cycle_count,
        and potentially relaxed structured_filters.
    """
    query = state.get("original_query", "")
    cycle_count = state.get("cycle_count", 0)
    confidence = state.get("confidence", "low")

    # Build failure reason from grading results
    grades = state.get("grading_results", [])
    failure_reason = _summarize_failure(grades, confidence)

    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": REWRITING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": REWRITING_USER_PROMPT.format(
                        query=query, failure_reason=failure_reason
                    ),
                },
            ],
            temperature=0.3,  # Slight creativity for alternative phrasing
        )

        rewritten = response.choices[0].message.content or query
        rewritten = rewritten.strip().strip('"').strip("'")

        # Track cost
        usage = response.usage
        cost = 0.0
        if usage:
            cost = estimate_cost(
                usage.prompt_tokens, usage.completion_tokens, "gpt-4o"
            )

        # On Cycle 2: relax filters
        new_filters = dict(state.get("structured_filters", {}))
        new_cycle = cycle_count + 1

        if new_cycle >= 2:
            # Relax filters: remove fiscal_year and section constraints
            new_filters.pop("fiscal_year", None)
            new_filters.pop("section", None)
            logger.info(
                f"Cycle {new_cycle}: Relaxing filters to {new_filters}"
            )

        logger.info(
            f"Query rewritten (cycle {new_cycle}): "
            f"'{query[:50]}...' → '{rewritten[:50]}...'"
        )

        return {
            "rewritten_query": rewritten,
            "cycle_count": new_cycle,
            "structured_filters": new_filters,
            "cost_accumulated": state.get("cost_accumulated", 0.0) + cost,
        }

    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        return {
            "rewritten_query": query,  # Use original as fallback
            "cycle_count": cycle_count + 1,
        }


def _summarize_failure(grades: list[dict], confidence: str) -> str:
    """Summarize why retrieval failed for the rewriting prompt.

    Args:
        grades: Grading results from the grader node.
        confidence: Current confidence level.

    Returns:
        Human-readable failure summary.
    """
    if not grades:
        return "No relevant documents were found in the initial search."

    irrelevant_count = sum(
        1 for g in grades if g.get("relevance") == "irrelevant"
    )
    partial_count = sum(
        1 for g in grades if g.get("relevance") == "partially_relevant"
    )
    relevant_count = sum(
        1 for g in grades if g.get("relevance") == "relevant"
    )

    reasons = []
    if irrelevant_count > 0:
        reasons.append(f"{irrelevant_count} chunks were irrelevant")
    if partial_count > 0:
        reasons.append(f"{partial_count} chunks were only partially relevant")
    if relevant_count == 0:
        reasons.append("no chunks directly contained the needed data")

    # Include specific grading reasons
    for g in grades[:3]:  # First 3 grades
        reason = g.get("reason", "")
        if reason:
            reasons.append(f"- {reason}")

    return "; ".join(reasons)
