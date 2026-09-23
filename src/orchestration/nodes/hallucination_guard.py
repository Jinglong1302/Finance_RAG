"""Hallucination guardrail node.

Post-generation faithfulness check that verifies the generated answer
is fully supported by the source context.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from src.orchestration.prompts.generation import format_context_for_generation
from src.orchestration.prompts.guardrail import (
    GUARDRAIL_SYSTEM_PROMPT,
    GUARDRAIL_USER_PROMPT,
)
from src.orchestration.state import CRAGState
from src.utils.logging import get_logger
from src.utils.tokens import cost_from_usage

logger = get_logger(__name__)


def hallucination_guard_node(state: CRAGState) -> dict[str, Any]:
    """Check generated answer for hallucinated claims.

    Compares the generated answer against the source context to detect
    unsupported claims, incorrect numbers, or fabricated information.

    Args:
        state: Current CRAG pipeline state.

    Returns:
        State update with hallucination_check result and final_answer.
    """
    answer = state.get("generation", "")
    enriched = state.get("enriched_contexts", [])

    if not answer or not enriched:
        return {
            "hallucination_check": "pass",
            "final_answer": answer,
        }

    # Format context for verification
    context = format_context_for_generation(enriched, include_parent=False)

    start_t = time.perf_counter()
    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": GUARDRAIL_USER_PROMPT.format(
                        answer=answer, context=context
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

        check_result = result.get("result", "pass")
        issues = result.get("issues", [])

        # Build the trace entry ONCE — single return path prevents duplicate entries.
        # Previously there were two separate return statements (early-return for
        # fail+cycle>=2, fall-through for all other cases) which caused a second
        # trace entry to be written for the fall-through path.
        duration_s = round(time.perf_counter() - start_t, 3)

        trace = {
            "node": "hallucination_guard",
            "model": "gpt-4o",
            "prompt_tokens": token_detail["prompt_tokens"],
            "cached_tokens": token_detail["cached_tokens"],
            "completion_tokens": token_detail["completion_tokens"],
            "cost_usd": cost,
            "result": check_result,
            "issues": issues,  # always preserved, not reset to []
            "duration_s": duration_s,
        }
        prev_trace = state.get("pipeline_trace") or []

        if check_result == "fail":
            logger.warning(
                f"Hallucination detected: {len(issues)} issues: "
                + "; ".join(issues[:3])
            )
            cycle_count = state.get("cycle_count", 0)
            if cycle_count >= 2:
                # Budget exhausted — annotate the answer with unverified warnings
                warning_text = (
                    "\n\n⚠️ **Verification Notes:**\n"
                    + "\n".join(f"- [UNVERIFIED] {issue}" for issue in issues)
                )
                final_answer = answer + warning_text
            else:
                # CRAG graph will trigger a rewrite cycle; pass answer through unchanged
                final_answer = answer
        else:
            logger.info("Hallucination check: PASS")
            final_answer = answer

        return {
            "hallucination_check": check_result,
            "final_answer": final_answer,
            "cost_accumulated": state.get("cost_accumulated", 0.0) + cost,
            "pipeline_trace": prev_trace + [trace],
        }

    except Exception as e:
        logger.error(f"Hallucination guard failed: {e}")
        # On error, pass through the answer with a warning
        return {
            "hallucination_check": "error",
            "final_answer": answer,
        }
