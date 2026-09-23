"""Query decomposition node for the CRAG pipeline.

Decomposes complex financial queries into sub-queries and extracts
structured metadata filters using GPT-4o.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from src.orchestration.prompts.decomposition import (
    DECOMPOSITION_SYSTEM_PROMPT,
    DECOMPOSITION_USER_PROMPT,
)
from src.orchestration.state import CRAGState
from src.utils.logging import get_logger
from src.utils.tokens import cost_from_usage

logger = get_logger(__name__)


def query_decomposer_node(state: CRAGState) -> dict[str, Any]:
    """Decompose the query into sub-queries and extract filters.

    Uses GPT-4o with JSON mode to produce:
    - sub_queries: focused retrieval queries
    - structured_filters: metadata constraints
    - query_type: classification for downstream behavior
    - result_count: adaptive top-K for reranking

    Args:
        state: Current CRAG pipeline state.

    Returns:
        State update dict with decomposition results.
    """
    query = state.get("original_query", "")
    if not query:
        return {"error": "No query provided"}

    start_t = time.perf_counter()
    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": DECOMPOSITION_USER_PROMPT.format(query=query),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result_text = response.choices[0].message.content or "{}"
        result = json.loads(result_text)

        # Track cost (using actual usage including cache discount)
        cost, token_detail = cost_from_usage(response.usage, "gpt-4o")

        # Extract and validate fields
        sub_queries = result.get("sub_queries", [query])
        if not sub_queries:
            sub_queries = [query]

        structured_filters = result.get("structured_filters", {})
        # Remove null values from filters
        structured_filters = {
            k: v for k, v in structured_filters.items() if v is not None
        }

        query_type = result.get("query_type", "factual_numeric")
        result_count = result.get("result_count", 5)

        logger.info(
            f"Query decomposed: {len(sub_queries)} sub-queries, "
            f"type={query_type}, filters={structured_filters}"
        )

        duration_s = round(time.perf_counter() - start_t, 3)

        trace = {
            "node": "query_decomposer",
            "model": "gpt-4o",
            "prompt_tokens": token_detail["prompt_tokens"],
            "cached_tokens": token_detail["cached_tokens"],
            "completion_tokens": token_detail["completion_tokens"],
            "cost_usd": cost,
            "sub_queries": sub_queries,
            "filters": structured_filters,
            "query_type": query_type,
            "duration_s": duration_s,
        }
        prev_trace = state.get("pipeline_trace") or []

        return {
            "sub_queries": sub_queries,
            "structured_filters": structured_filters,
            "query_type": query_type,
            "result_count": result_count,
            "cost_accumulated": state.get("cost_accumulated", 0.0) + cost,
            "pipeline_trace": prev_trace + [trace],
        }

    except Exception as e:
        logger.error(f"Query decomposition failed: {e}")
        # Fallback: use original query as-is
        return {
            "sub_queries": [query],
            "structured_filters": {},
            "query_type": "factual_numeric",
            "result_count": 5,
            "error": f"Decomposition failed: {e}",
        }
