"""Token counting utilities using tiktoken.

Provides consistent token counting across all modules to ensure chunk sizes
and cost estimates are accurate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import tiktoken

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Pricing table (USD per token).  Update here when a provider reprices models.
# OpenAI: cached input tokens are billed at 50% of the standard input rate.
# Gemini: paid-tier list prices; the free tier costs $0. The OpenAI-compatible
# endpoint does not report cached tokens, so no cache discount is applied.
# Gemini prices were taken from third-party price trackers on 2026-10-23
# (Google's own page was not reachable); check ai.google.dev before relying on them.
# ---------------------------------------------------------------------------
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "input":         2.50 / 1_000_000,
        "input_cached":  1.25 / 1_000_000,  # 50 % cache discount
        "output":       10.00 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input":         0.150 / 1_000_000,
        "input_cached":  0.075 / 1_000_000,
        "output":        0.600 / 1_000_000,
    },
    "gemini-2.5-flash": {
        "input":         0.30 / 1_000_000,
        "input_cached":  0.30 / 1_000_000,
        "output":        2.50 / 1_000_000,
    },
    "gemini-3.5-flash": {
        "input":         1.50 / 1_000_000,
        "input_cached":  1.50 / 1_000_000,
        "output":        9.00 / 1_000_000,
    },
}

# Models already warned about, so the log is not flooded on every call
_UNPRICED_WARNED: set[str] = set()


# Cache the encoder instance for performance
_encoder: tiktoken.Encoding | None = None


def get_encoder(model: str = "gpt-4o") -> tiktoken.Encoding:
    """Get a cached tiktoken encoder for the specified model.

    Args:
        model: The model name to get the encoder for. Defaults to gpt-4o.

    Returns:
        A tiktoken Encoding instance.
    """
    global _encoder
    if _encoder is None:
        try:
            _encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (used by GPT-4, GPT-4o)
            _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to count tokens for.
        model: The model to use for tokenization.

    Returns:
        Number of tokens.
    """
    encoder = get_encoder(model)
    return len(encoder.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens to keep.
        model: The model to use for tokenization.

    Returns:
        Truncated text.
    """
    encoder = get_encoder(model)
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o",
    cached_tokens: int = 0,
) -> float:
    """Estimate the USD cost for an OpenAI API call.

    Applies the 50% cached-token discount when ``cached_tokens`` is provided.
    Pass ``usage.prompt_tokens_details.cached_tokens`` from the OpenAI
    response to get an accurate figure that matches the OpenAI dashboard.

    Args:
        input_tokens: Total prompt tokens (includes cached tokens).
        output_tokens: Completion tokens.
        model: The model used.
        cached_tokens: Subset of input_tokens that were served from cache
            (billed at half the standard input rate).

    Returns:
        Estimated cost in USD.
    """
    rates = _PRICING.get(model)
    if rates is None:
        if model not in _UNPRICED_WARNED:
            _UNPRICED_WARNED.add(model)
            logging.getLogger(__name__).warning(
                "No price for model %r in src/utils/tokens.py; costing it as gpt-4o", model
            )
        rates = _PRICING["gpt-4o"]
    non_cached = input_tokens - cached_tokens
    return (
        non_cached      * rates["input"]
        + cached_tokens * rates["input_cached"]
        + output_tokens * rates["output"]
    )


def cost_from_usage(usage: Any, model: str = "gpt-4o") -> tuple[float, dict[str, int]]:
    """Compute accurate cost from an OpenAI CompletionUsage object.

    Extracts cached-token counts from ``usage.prompt_tokens_details`` so the
    result matches the OpenAI billing dashboard instead of using the list price
    for all input tokens.

    Args:
        usage: ``response.usage`` from an OpenAI chat completion.
        model: Model name for pricing lookup.

    Returns:
        Tuple of (cost_usd, token_detail_dict) where token_detail_dict has keys
        ``prompt_tokens``, ``cached_tokens``, ``completion_tokens``.
    """
    if usage is None:
        return 0.0, {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}

    prompt_tokens: int = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens: int = getattr(usage, "completion_tokens", 0) or 0

    # cached_tokens lives in prompt_tokens_details (may be None on older SDK)
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens: int = 0
    if details is not None:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0

    cost = estimate_cost(prompt_tokens, completion_tokens, model, cached_tokens)
    token_detail = {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
    }
    return cost, token_detail
