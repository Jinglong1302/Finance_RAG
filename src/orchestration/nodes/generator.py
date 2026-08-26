"""Answer generator node for the CRAG pipeline.

Generates grounded financial answers with citations using GPT-4o.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.orchestration.prompts.generation import (
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT,
    format_context_for_generation,
)
from src.orchestration.state import CRAGState
from src.utils.logging import get_logger
from src.utils.tokens import cost_from_usage

logger = get_logger(__name__)


def generator_node(state: CRAGState) -> dict[str, Any]:
    """Generate a grounded answer with citations.

    Uses enriched contexts (child + parent text) to produce an answer
    with numbered citations and optional structured metrics.

    Args:
        state: Current CRAG pipeline state.

    Returns:
        State update with generation, citations, and structured_metrics.
    """
    query = state.get("original_query", "")
    enriched = state.get("enriched_contexts", [])
    confidence = state.get("confidence", "medium")

    if not enriched:
        return {
            "generation": "I could not find sufficient evidence in the available SEC filings to answer this question.",
            "citations": [],
            "confidence": "insufficient",
        }

    # Format context
    context = format_context_for_generation(enriched)

    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": GENERATION_USER_PROMPT.format(
                        confidence=confidence,
                        query=query,
                        context=context,
                    ),
                },
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content or ""

        # Track cost (using actual usage including cache discount)
        cost, token_detail = cost_from_usage(response.usage, "gpt-4o")

        # Extract structured citations from the enriched contexts
        citations = _build_citations(enriched)

        # Try to extract structured metrics from the answer
        structured_metrics = _extract_structured_metrics(answer)

        logger.info(
            f"Generated answer: {len(answer)} chars, "
            f"{len(citations)} citations, "
            f"confidence={confidence}"
        )

        trace = {
            "node": "generator",
            "model": "gpt-4o",
            "prompt_tokens": token_detail["prompt_tokens"],
            "cached_tokens": token_detail["cached_tokens"],
            "completion_tokens": token_detail["completion_tokens"],
            "cost_usd": cost,
            "answer_length": len(answer),
            "citations_count": len(citations),
        }
        prev_trace = state.get("pipeline_trace") or []

        return {
            "generation": answer,
            "citations": citations,
            "structured_metrics": structured_metrics,
            "cost_accumulated": state.get("cost_accumulated", 0.0) + cost,
            "pipeline_trace": prev_trace + [trace],
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {
            "generation": f"Error generating answer: {e}",
            "citations": [],
            "error": str(e),
        }


def _build_citations(enriched_contexts: list[dict]) -> list[dict]:
    """Build structured citation objects from enriched contexts.

    Args:
        enriched_contexts: List of context dicts with metadata.

    Returns:
        List of citation dicts.
    """
    citations = []
    for i, ctx in enumerate(enriched_contexts):
        meta = ctx.get("metadata", {})
        citation = {
            "index": i + 1,
            "company": meta.get("company_name", meta.get("company_ticker", "")),
            "filing_type": meta.get("filing_type", ""),
            "fiscal_year": meta.get("fiscal_year", ""),
            "section": meta.get("section_title", ""),
            "chunk_id": ctx.get("chunk_id", ""),
            "evidence_snippet": ctx.get("child_text", "")[:200],
        }
        citations.append(citation)
    return citations


def _extract_structured_metrics(answer: str) -> list[dict] | None:
    """Extract structured metrics JSON block from the answer.

    Looks for a ```json code block in the answer and parses it.

    Args:
        answer: The generated answer text.

    Returns:
        List of metric dicts or None if no structured output found.
    """
    import re

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", answer, re.DOTALL)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group(1))
        return data.get("metrics", [])
    except (json.JSONDecodeError, AttributeError):
        return None
