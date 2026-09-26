"""CRAG document relevance grader node.

Grades all reranked chunks in a single batch GPT-4o call using
ternary relevance (relevant/partially_relevant/irrelevant).
Determines the CRAG action based on grade distribution.
"""

from __future__ import annotations

import json
import time
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

    # Stage 1 — coarse gate: reranker/similarity score threshold on the top retrieved chunk
    # Below threshold → abstain immediately, skip generation. Targets out-of-corpus queries.
    from config.settings import get_settings
    settings = get_settings()

    reranked = state.get("reranked_results", [])
    top_score = float(reranked[0].get("score", -999.0)) if reranked else -999.0

    if not reranked or not enriched or top_score < settings.reranker_coarse_threshold:
        logger.info(
            f"Stage 1 coarse gate: top rerank score {top_score:.4f} < {settings.reranker_coarse_threshold}. Abstaining immediately."
        )
        return {
            "grading_results": [],
            "confidence": "insufficient",
            "crag_action": "refuse",
            "is_abstention": True,
            "abstention_stage": 1,
            "abstention_reason": f"Stage 1 coarse gate: top rerank score {top_score:.2f} below threshold {settings.reranker_coarse_threshold}",
        }

    # Format chunks for grading
    chunks_text = format_chunks_for_grading(
        [{"text": e.get("child_text", ""), "metadata": e.get("metadata", {})}
         for e in enriched]
    )

    client = OpenAI()
    start_t = time.perf_counter()

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

        # Enrich grades with chunk metadata so the user can inspect what was graded and why
        enriched_grades = []
        for g in grades:
            idx = g.get("chunk_index", 0)
            chunk_obj = enriched[idx] if idx < len(enriched) else {}
            if hasattr(chunk_obj, "metadata"):
                chunk_meta = chunk_obj.metadata or {}
                chunk_text = chunk_obj.child_text or ""
            elif isinstance(chunk_obj, dict):
                chunk_meta = chunk_obj.get("metadata", {}) or {}
                chunk_text = chunk_obj.get("child_text", "") or ""
            else:
                chunk_meta = {}
                chunk_text = str(chunk_obj)

            parent_text = (
                getattr(chunk_obj, "parent_text", None)
                if hasattr(chunk_obj, "parent_text")
                else (chunk_obj.get("parent_text") if isinstance(chunk_obj, dict) else None)
            )

            enriched_grades.append({
                "chunk_index": idx,
                "relevance": g.get("relevance", "irrelevant"),
                "reason": g.get("reason", "No reason provided"),
                "company": chunk_meta.get("company_ticker", "AAPL"),
                "fiscal_year": chunk_meta.get("fiscal_year", ""),
                "section": chunk_meta.get("section_title", chunk_meta.get("section_id", "Section")),
                "text_preview": chunk_text[:200].replace("\n", " "),
                "text_full": chunk_text,
                "parent_text": parent_text,
            })

        duration_s = round(time.perf_counter() - start_t, 3)

        trace = {
            "node": "grader",
            "model": "gpt-4o",
            "prompt_tokens": token_detail["prompt_tokens"],
            "cached_tokens": token_detail["cached_tokens"],
            "completion_tokens": token_detail["completion_tokens"],
            "cost_usd": cost,
            "relevant": relevant_count,
            "relevant_count": relevant_count,
            "partial": partial_count,
            "ambiguous_count": partial_count,
            "irrelevant": irrelevant_count,
            "irrelevant_count": irrelevant_count,
            "confidence": confidence,
            "action": action,
            "grades": enriched_grades,
            "duration_s": duration_s,
        }
        prev_trace = state.get("pipeline_trace") or []

        state_update = {
            "grading_results": grades,
            "confidence": confidence,
            "crag_action": action,
            "enriched_contexts": filtered_contexts if filtered_contexts else enriched,
            "cost_accumulated": state.get("cost_accumulated", 0.0) + cost,
            "pipeline_trace": prev_trace + [trace],
        }

        if action == "refuse":
            state_update["is_abstention"] = True
            state_update["abstention_stage"] = 2
            state_update["abstention_reason"] = (
                "Stage 2 fine-grained check: insufficient evidence in filing to answer question after retries"
            )
        else:
            state_update["is_abstention"] = False

        return state_update

    except Exception as e:
        logger.error(f"Grading failed: {e}")
        # Fallback: proceed with generation on error
        return {
            "grading_results": [],
            "confidence": "medium",
            "crag_action": "generate",
            "is_abstention": False,
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
    if relevant >= 3:
        return "high", "generate"

    if relevant >= 1 and (relevant + partial) >= 2:
        return "medium", "generate"

    if relevant == 0 and partial >= 2:
        if cycle_count < max_cycles:
            return "low", "rewrite"
        else:
            # Stage 2: Terminate in abstention after retries are exhausted, instead of forcing an answer
            return "insufficient", "refuse"

    # All or mostly irrelevant
    if cycle_count < max_cycles:
        return "insufficient", "rewrite"
    else:
        return "insufficient", "refuse"
