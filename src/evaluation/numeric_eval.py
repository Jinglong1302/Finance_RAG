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
    "trillions": 1e12,
    "trillion": 1e12,
    "billions": 1e9,
    "billion": 1e9,
    "millions": 1e6,
    "million": 1e6,
    "thousands": 1e3,
    "thousand": 1e3,
    "bn": 1e9,
    "mn": 1e6,
    "b": 1e9,
    "m": 1e6,
    "k": 1e3,
    "t": 1e12,
}


def extract_numbers(text: str) -> list[float]:
    """Extract all candidate financial numbers from text."""
    if not text:
        return []
    # Strip footnote citations like [1], [2], [12]
    cleaned_text = re.sub(r"\[\d+\]", " ", text)
    # Strip SEC form designations like 10-K, 10-Q, 8-K
    cleaned_text = re.sub(r"\b\d+-[KQ]\b", " ", cleaned_text, flags=re.IGNORECASE)
    # Strip Note references like Note 16, Notes 3 and 11
    cleaned_text = re.sub(r"\bNotes?\s+\d+\b", " ", cleaned_text, flags=re.IGNORECASE)
    # Avoid extracting company name '3M' as 3 million
    cleaned_text = re.sub(r"\b3M\b|\b3M's\b", " ", cleaned_text, flags=re.IGNORECASE)

    tokens = re.findall(
        r"[-−]?\$?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|trillions|trillion|billions|billion|millions|million|thousands|thousand|bn|mn|[bmkt])?",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    numbers: list[float] = []
    for token in tokens:
        token = token.strip().rstrip(".:;")
        if token and token.upper() not in ("3M", "3M'S"):
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
    def _is_close(val: float, target: float) -> bool:
        if abs(target) < 1e-9:
            return abs(val) < 1e-9
        diff = abs(val - target) / abs(target)
        if diff <= tolerance:
            return True
        # Check scale multipliers (k, m, b, t) when one answer includes unit multiplier and other omitted it
        for scale in (1e3, 1e6, 1e9, 1e12, 1e-3, 1e-6, 1e-9, 1e-12):
            scaled_target = target * scale
            if abs(val - scaled_target) / abs(scaled_target) <= tolerance:
                return True
        return False

    exp_cands = extract_numbers(expected)
    exp_num = parse_financial_number(expected)
    if exp_num is not None and exp_num not in exp_cands:
        exp_cands.insert(0, exp_num)

    if not exp_cands:
        return False

    pred_cands = extract_numbers(predicted)
    pred_num = parse_financial_number(predicted)
    if pred_num is not None and pred_num not in pred_cands:
        pred_cands.insert(0, pred_num)

    if not pred_cands:
        return False

    for exp_val in exp_cands:
        # Ignore standalone 4-digit calendar years (e.g. 2018, 2022) unless it is the only candidate
        if len(exp_cands) > 1 and 1900 <= exp_val <= 2099 and exp_val.is_integer():
            continue
        for pred_val in pred_cands:
            if len(pred_cands) > 1 and 1900 <= pred_val <= 2099 and pred_val.is_integer():
                continue
            if _is_close(pred_val, exp_val):
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

    if text.strip().upper() in ("3M", "3M'S"):
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
            "extracted_predicted": extract_numbers(pred),
            "extracted_expected": extract_numbers(gt),
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
