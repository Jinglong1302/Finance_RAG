"""Pooled-context sufficiency grader node (Part B).

Replaces per-chunk relevance grading with a single LLM call that evaluates
whether the *concatenated* top-k context is collectively sufficient to answer
the question.

Decision logic
--------------
Stage 1 (coarse gate, unchanged):
  If top reranker score < threshold → refuse immediately (catches OOC queries)

Stage 2 (pooled sufficiency call):
  SUFFICIENT  → confidence=high,   action=generate
  PARTIAL     → confidence=medium, action=generate   [KEY CHANGE vs per-chunk]
  INSUFFICIENT:
    if cycles_remaining > 0 → action=rewrite
    else                    → action=refuse

The critical difference from the per-chunk grader: PARTIAL → generate instead
of PARTIAL → rewrite.  Per-chunk grading sees each chunk as "partially relevant"
even when the chunks *together* fully answer the question (e.g., multi-component
calculations spread across chunks, qualitative discussions).
"""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from src.orchestration.prompts.sufficiency import (
    SUFFICIENCY_SYSTEM_PROMPT,
    SUFFICIENCY_USER_PROMPT,
    format_context_for_sufficiency,
)
from src.orchestration.state import CRAGState
from src.utils.logging import get_logger
from src.utils.tokens import cost_from_usage

logger = get_logger(__name__)


def sufficiency_grader_node(state: CRAGState) -> dict[str, Any]:
    """Pooled-context sufficiency grader.

    Single LLM call on the concatenated top-k context to determine whether
    generation should proceed, retrieval should be retried, or the query
    should be refused.

    Args:
        state: Current CRAG pipeline state.

    Returns:
        State update with crag_action, confidence, and grading metadata.
    """
    enriched = state.get("enriched_contexts", [])
    query = state.get("original_query", "")
    cycle_count = state.get("cycle_count", 0)
    max_cycles = 2

    # ── Stage 1: coarse gate (unchanged from per-chunk grader) ─────────
    from config.settings import get_settings
    settings = get_settings()

    reranked = state.get("reranked_results", [])
    top_score = float(reranked[0].get("score", -999.0)) if reranked else -999.0

    if not reranked or not enriched or top_score < settings.reranker_coarse_threshold:
        logger.info(
            "Stage 1 coarse gate: top rerank score %.4f < %.1f → refuse",
            top_score,
            settings.reranker_coarse_threshold,
        )
        return {
            "grading_results": [],
            "confidence": "insufficient",
            "crag_action": "refuse",
            "is_abstention": True,
            "abstention_stage": 1,
            "abstention_reason": (
                f"Stage 1 coarse gate: top rerank score {top_score:.2f} "
                f"below threshold {settings.reranker_coarse_threshold}"
            ),
        }

    # ── Stage 2: pooled sufficiency call ─────────────────────────────
    context_text = format_context_for_sufficiency(enriched)
    client = OpenAI()
    start_t = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SUFFICIENCY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SUFFICIENCY_USER_PROMPT.format(
                        question=query,
                        context=context_text,
                    ),
                },
            ],
            temperature=0,
            seed=42,
            max_tokens=10,  # One-word response
        )
        duration_s = round(time.perf_counter() - start_t, 3)
        cost, token_detail = cost_from_usage(response.usage, "gpt-4o")

        raw_verdict = (response.choices[0].message.content or "").strip().upper()
        # Normalise: take first word only in case the model adds punctuation
        verdict = raw_verdict.split()[0] if raw_verdict.split() else "INSUFFICIENT"
        if verdict not in ("SUFFICIENT", "PARTIAL", "INSUFFICIENT"):
            logger.warning("Unexpected verdict '%s'; treating as PARTIAL", verdict)
            verdict = "PARTIAL"

        logger.info(
            "Pooled sufficiency: verdict=%s cycle=%d dur=%.2fs",
            verdict,
            cycle_count,
            duration_s,
        )

    except Exception as exc:
        logger.error("Sufficiency grader failed: %s; falling back to generate", exc)
        verdict = "PARTIAL"
        cost, token_detail = 0.0, {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        duration_s = 0.0

    # ── Decision logic ────────────────────────────────────────────────
    if verdict == "SUFFICIENT":
        confidence = "high"
        action = "generate"
    elif verdict == "PARTIAL":
        # KEY CHANGE: always try to generate even on PARTIAL verdict
        # (per-chunk grader would rewrite here on cycle 0)
        confidence = "medium"
        action = "generate"
    else:  # INSUFFICIENT
        if cycle_count < max_cycles:
            confidence = "low"
            action = "rewrite"
        else:
            confidence = "insufficient"
            action = "refuse"

    prev_trace = state.get("pipeline_trace") or []
    trace = {
        "node": "sufficiency_grader",
        "model": "gpt-4o",
        "verdict": verdict,
        "action": action,
        "confidence": confidence,
        "cycle": cycle_count,
        "prompt_tokens": token_detail["prompt_tokens"],
        "cached_tokens": token_detail["cached_tokens"],
        "completion_tokens": token_detail["completion_tokens"],
        "cost_usd": cost,
        "duration_s": duration_s,
    }

    return {
        "grading_results": [{"verdict": verdict}],
        "confidence": confidence,
        "crag_action": action,
        "cost_accumulated": state.get("cost_accumulated", 0.0) + cost,
        "pipeline_trace": prev_trace + [trace],
    }
