"""Answer generator node for the CRAG pipeline.

Generates grounded financial answers with citations using GPT-4o.
"""

from __future__ import annotations

import json
import time
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
    user_prompt = GENERATION_USER_PROMPT.format(
        confidence=confidence,
        query=query,
        context=context,
    )

    client = OpenAI()
    start_t = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content or ""
        duration_s = round(time.perf_counter() - start_t, 3)

        # Track cost (using actual usage including cache discount)
        cost, token_detail = cost_from_usage(response.usage, "gpt-4o")

        # Extract structured citations from the enriched contexts
        citations = _build_citations(enriched)

        # Try to extract structured metrics from the answer
        structured_metrics = _extract_structured_metrics(answer)

        logger.info(
            f"Generated answer: {len(answer)} chars, "
            f"{len(citations)} citations, "
            f"confidence={confidence}, time={duration_s}s"
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
            "duration_s": duration_s,
            "llm_input_system": GENERATION_SYSTEM_PROMPT,
            "llm_input_user": user_prompt,
            "llm_output": answer,
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


def _build_citations(enriched_contexts: list[Any]) -> list[dict]:
    """Build structured citation objects from enriched contexts.

    Args:
        enriched_contexts: List of context dicts or EnrichedContext objects.

    Returns:
        List of citation dicts with full metadata, full text, and parent context.
    """
    import re

    citations = []
    for i, ctx in enumerate(enriched_contexts):
        if hasattr(ctx, "metadata"):
            meta = getattr(ctx, "metadata", {}) or {}
            child_text = getattr(ctx, "child_text", "") or ""
            parent_text = getattr(ctx, "parent_text", "") or ""
            chunk_id = getattr(ctx, "chunk_id", "") or ""
        elif isinstance(ctx, dict):
            meta = ctx.get("metadata", {}) or {}
            child_text = ctx.get("child_text", "") or ctx.get("text", "") or ""
            parent_text = ctx.get("parent_text", "") or ""
            chunk_id = ctx.get("chunk_id", "") or ""
        else:
            meta = {}
            child_text = str(ctx)
            parent_text = ""
            chunk_id = ""

        company = meta.get("company_name", meta.get("company_ticker", "AAPL"))
        company_ticker = meta.get("company_ticker", company)
        section = meta.get("section_title", meta.get("section_id", "Financial Statements"))
        year = meta.get("fiscal_year", "")
        filing_type = meta.get("filing_type", "10-K")

        # Create a clean text summary for excerpt preview (strip HTML and markdown hashes)
        clean_text = re.sub(r"<[^>]+>", " ", child_text)
        clean_text = re.sub(r"^#+\s*", "", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        snippet = clean_text[:280].strip() if clean_text else child_text[:280].strip()

        citation = {
            "index": i + 1,
            "id": i + 1,
            "company": company,
            "company_ticker": company_ticker,
            "filing_type": filing_type,
            "fiscal_year": year,
            "section": section,
            "section_title": section,
            "chunk_id": chunk_id,
            "evidence_snippet": snippet,
            "excerpt": snippet,
            "full_text": child_text,
            "text_full": child_text,
            "parent_text": parent_text,
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
