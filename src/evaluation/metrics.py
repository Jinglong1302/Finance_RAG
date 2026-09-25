"""Shared retrieval and generation evaluation metrics.

Provides Recall@k, Hit@k, MRR for retrieval, and exact-match / numeric
accuracy for generation — no LLM-as-judge required for these.
"""
from __future__ import annotations

import re
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def hit_at_k(retrieved_texts: list[str], evidence_texts: list[str], k: int) -> bool:
    """Return True if at least one of the top-k chunks contains any evidence.

    Evidence match uses case-insensitive substring: the first 120 chars of
    each evidence snippet must appear verbatim in a retrieved chunk.
    """
    probes = [_probe(e) for e in evidence_texts if e]
    for text in retrieved_texts[:k]:
        text_lower = text.lower()
        for probe in probes:
            if probe in text_lower:
                return True
    return False


def recall_at_k(retrieved_texts: list[str], evidence_texts: list[str], k: int) -> float:
    """Fraction of evidence snippets that appear in the top-k retrieved chunks."""
    if not evidence_texts:
        return 0.0
    found = 0
    for evidence in evidence_texts:
        probe = _probe(evidence)
        for text in retrieved_texts[:k]:
            if probe in text.lower():
                found += 1
                break
    return found / len(evidence_texts)


def mrr(retrieved_texts: list[str], evidence_texts: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first chunk that contains any evidence."""
    probes = [_probe(e) for e in evidence_texts if e]
    for rank, text in enumerate(retrieved_texts, start=1):
        text_lower = text.lower()
        for probe in probes:
            if probe in text_lower:
                return 1.0 / rank
    return 0.0


def _probe(evidence_text: str, max_len: int = 120) -> str:
    """Normalised first `max_len` chars of evidence text for substring matching."""
    text = re.sub(r"\s+", " ", evidence_text.strip()).lower()
    return text[:max_len]


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def aggregate_retrieval_metrics(
    per_question: list[dict[str, Any]],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Aggregate per-question retrieval metrics into means."""
    if not per_question:
        return {}

    agg: dict[str, float] = {}
    for k in k_values:
        key = f"Hit@{k}"
        agg[key] = float(
            sum(q.get(key, 0) for q in per_question) / len(per_question)
        )

    for metric in ("Recall@5", "Recall@10", "MRR"):
        vals = [q[metric] for q in per_question if metric in q]
        if vals:
            agg[metric] = float(sum(vals) / len(vals))

    return agg
