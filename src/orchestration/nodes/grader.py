"""CRAG document relevance grader node.

Grades all reranked chunks in a single batch GPT-4o call using
ternary relevance (relevant/partially_relevant/irrelevant).
Determines the CRAG action based on grade distribution.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.orchestration.prompts.grading import (
    GRADING_SYSTEM_PROMPT,
    GRADING_USER_PROMPT,
    format_chunks_for_grading,
)
from src.orchestration.state import CRAGState
from src.utils.logging import get_logger
from src.utils.tokens import cost_from_usage

logger = get_logger(__name__)


def grader_node(state: CRAGState) -> dict[str, Any]:
    """Grade document relevance and determine CRAG action.

    Grades all reranked chunks in a single API call. Determines
    whether to generate, rewrite, or refuse based on grade distribution.

    Args:
        state: Current CRAG pipeline state.

    Returns:
        State update with grading_results, confidence, and crag_action.
    """
    enriched = state.get("enriched_contexts", [])
    query = state.get("original_query", "")
    cycle_count = state.get("cycle_count", 0)
    max_cycles = 2  # From design decisions

    if not enriched:
        return {
            "grading_results": [],
            "confidence": "insufficient",
            "crag_action": "refuse",
        }

    # Format chunks for grading
    chunks_text = format_chunks_for_grading(
        [{"text": e.get("child_text", ""), "metadata": e.get("metadata", {})}
         for e in enriched]
    )

    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": GRADING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": GRADING_USER_PROMPT.format(
                        query=query, chunks_text=chunks_text
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result_text = response.choices[0].message.content or "{}"
        result = json.loads(result_text)

        # Track cost (using actual usage including cache discount)
        cost, token_detail = cost_from_usage(response.usage, "gpt-4o")

        grades = result.get("grades", [])

        # Count by relevance level
        relevant_count = sum(
            1 for g in grades if g.get("relevance") == "relevant"
        )
        partial_count = sum(
            1 for g in grades if g.get("relevance") == "partially_relevant"
        )
        irrelevant_count = sum(
            1 for g in grades if g.get("relevance") == "irrelevant"
        )

        # Determine confidence and action
        confidence, action = _determine_action(
            relevant_count, partial_count, irrelevant_count,
            cycle_count, max_cycles,
        )

        logger.info(
            f"Grading: {relevant_count} relevant, "
            f"{partial_count} partial, {irrelevant_count} irrelevant → "
            f"confidence={confidence}, action={action}"
        )

        # Filter out irrelevant chunks from enriched contexts
        filtered_contexts = []
        for i, ctx in enumerate(enriched):
            grade = next(
                (g for g in grades if g.get("chunk_index") == i), None
            )
            if grade and grade.get("relevance") != "irrelevant":
                filtered_contexts.append(ctx)

        trace = {
            "node": "grader",
            "model": "gpt-4o",
            "prompt_tokens": token_detail["prompt_tokens"],
            "cached_tokens": token_detail["cached_tokens"],
            "completion_tokens": token_detail["completion_tokens"],
            "cost_usd": cost,
            "relevant": relevant_count,
            "partial": partial_count,
            "irrelevant": irrelevant_count,
            "confidence": confidence,
            "action": action,
            "grades": grades,
        }
        prev_trace = state.get("pipeline_trace") or []

        return {
            "grading_results": grades,
            "confidence": confidence,
            "crag_action": action,
            "enriched_contexts": filtered_contexts if filtered_contexts else enriched,
            "cost_accumulated": state.get("cost_accumulated", 0.0) + cost,
            "pipeline_trace": prev_trace + [trace],
        }

    except Exception as e:
        logger.error(f"Grading failed: {e}")
        # Fallback: proceed with generation on error
        return {
            "grading_results": [],
            "confidence": "medium",
            "crag_action": "generate",
        }


def _determine_action(
    relevant: int,
    partial: int,
    irrelevant: int,
    cycle_count: int,
    max_cycles: int,
) -> tuple[str, str]:
    """Determine confidence level and CRAG action from grade distribution.

    Args:
        relevant: Count of relevant grades.
        partial: Count of partially_relevant grades.
        irrelevant: Count of irrelevant grades.
        cycle_count: Current CRAG cycle number.
        max_cycles: Maximum allowed cycles.

    Returns:
        Tuple of (confidence, action).
    """
    # Decision matrix from Q22
    if relevant >= 3:
        return "high", "generate"

    if relevant >= 1 and (relevant + partial) >= 2:
        return "medium", "generate"

    if relevant == 0 and partial >= 2:
        if cycle_count < max_cycles:
            return "low", "rewrite"
        else:
            return "low", "generate"  # Generate with low confidence after max cycles

    # All or mostly irrelevant
    if cycle_count < max_cycles:
        return "insufficient", "rewrite"
    else:
        return "insufficient", "refuse"
