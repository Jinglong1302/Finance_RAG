"""Deterministic numerical matching for TAT-QA evaluation.

Bypasses LLM-as-judge for numerical answers by parsing and comparing
financial numbers with tolerance.
"""

from __future__ import annotations

import re
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Patterns for parsing financial numbers
_MULTIPLIERS = {
    "thousand": 1e3,
    "thousands": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "trillion": 1e12,
    "trillions": 1e12,
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
}


def extract_numbers(text: str) -> list[float]:
    """Extract all candidate financial numbers from text."""
    if not text:
        return []
    tokens = re.findall(
        r"[-−]?\$?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|billion|billions|million|millions|thousand|thousands|[bmk])?",
        text,
        flags=re.IGNORECASE,
    )
    numbers: list[float] = []
    for token in tokens:
        token = token.strip().rstrip(".:;")
        if token:
            val = parse_financial_number(token)
            if val is not None:
                numbers.append(val)
    return numbers


def numeric_match(
    predicted: str,
    expected: str,
    tolerance: float = 0.01,
) -> bool:
    """Compare financial numbers with tolerance for formatting differences.

    Handles various formats:
    - "$394.3B", "$394,328 million", "394328"
    - "2.88%", "2.88 percent"
    - Negative numbers: "$(1,234)" or "-1,234"

    Args:
        predicted: Predicted answer string.
        expected: Expected (ground truth) answer string.
        tolerance: Relative tolerance for matching. Default: 1%.

    Returns:
        True if numbers match within tolerance.
    """
    exp_num = parse_financial_number(expected)
    if exp_num is None:
        return False

    pred_num = parse_financial_number(predicted)
    if pred_num is not None:
        if abs(exp_num) < 1e-9:
            if abs(pred_num) < 1e-9:
                return True
        elif abs(pred_num - exp_num) / abs(exp_num) <= tolerance:
            return True

    # Fallback: scan candidate numbers in predicted text (useful when answer contains reasoning/citations)
    for cand in extract_numbers(predicted):
        if abs(exp_num) < 1e-9:
            if abs(cand) < 1e-9:
                return True
        elif abs(cand - exp_num) / abs(exp_num) <= tolerance:
            return True

    return False


def parse_financial_number(text: str) -> float | None:
    """Parse a financial number from text.

    Handles common formats found in SEC filings and answers:
    - "$394,328" → 394328.0
    - "$394.3 billion" → 394300000000.0
    - "394.3B" → 394300000000.0
    - "(1,234)" → -1234.0
    - "2.88%" → 2.88
    - "-$1.5M" → -1500000.0

    Args:
        text: Text containing a financial number.

    Returns:
        Parsed float value, or None if no number found.
    """
    if not text or not text.strip():
        return None

    # Check for structured JSON metrics block (from CRAG generator)
    import json
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            metrics = data.get("metrics", [])
            if metrics and "value" in metrics[0]:
                val = metrics[0]["value"]
                if isinstance(val, (int, float)):
                    return float(val)
        except Exception:
            pass

    text = text.strip().rstrip(".:;")

    # Detect negative (parenthetical notation)
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    elif text.startswith("-") or text.startswith("−"):
        negative = True
        text = text[1:]

    # Remove currency symbols and whitespace
    text = re.sub(r"[$€£¥]", "", text).strip()

    # Remove "negative" or "loss of"
    text = re.sub(r"(?:negative|loss\s+of)\s*", "", text, flags=re.IGNORECASE)

    # Check for percentage
    is_percent = False
    if text.endswith("%") or "percent" in text.lower():
        is_percent = True
        text = re.sub(r"[%]|percent", "", text, flags=re.IGNORECASE).strip()

    # Find multiplier suffix
    multiplier = 1.0
    text_lower = text.lower().strip()
    for suffix, mult in _MULTIPLIERS.items():
        if text_lower.endswith(suffix):
            multiplier = mult
            text = text[: -len(suffix)].strip()
            break

    # Also check for written-out multipliers
    for word, mult in _MULTIPLIERS.items():
        if word in text_lower and len(word) > 1:  # Skip single-letter
            multiplier = mult
            text = re.sub(rf"\s*{word}\s*", "", text, flags=re.IGNORECASE).strip()
            break

    # Remove remaining non-numeric chars except . and ,
    text = re.sub(r"[^\d.,\-]", "", text)

    # Remove commas (thousands separator)
    text = text.replace(",", "")

    # Parse the number
    try:
        value = float(text) * multiplier
        if negative:
            value = -value
        return value
    except (ValueError, TypeError):
        return None


def evaluate_numeric_accuracy(
    predictions: list[str],
    ground_truths: list[str],
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Evaluate numeric accuracy across a set of predictions.

    Args:
        predictions: List of predicted answer strings.
        ground_truths: List of ground truth answer strings.
        tolerance: Relative tolerance. Default: 1%.

    Returns:
        Dict with accuracy, match count, and per-sample results.
    """
    results = []
    matches = 0

    for pred, gt in zip(predictions, ground_truths):
        is_match = numeric_match(pred, gt, tolerance)
        if is_match:
            matches += 1
        results.append({
            "predicted": pred,
            "expected": gt,
            "parsed_predicted": parse_financial_number(pred),
            "parsed_expected": parse_financial_number(gt),
            "match": is_match,
        })

    total = len(predictions)
    accuracy = matches / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "matches": matches,
        "total": total,
        "per_sample": results,
    }
