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
# Retrieval metrics (FinanceBench & standard)
# ---------------------------------------------------------------------------


def chunk_matches_evidence(
    chunk: Any,
    evidence_entry: dict[str, Any] | str,
    text_overlap_threshold: float = 0.50,
) -> bool:
    """Return True if chunk matches evidence via Page match OR Text-overlap match >= 50%.

    Rule (FinanceBench):
    A retrieved chunk counts as a hit if EITHER:
    1. Page match — chunk's source page metadata equals evidence_page_num, OR
    2. Text-overlap match — word-level overlap between chunk text and evidence_text is >= 50%
    Use OR, not AND — parent-child chunking can span page boundaries, and PDF
    extraction can shift page numbers by one. For questions with multiple evidence
    entries, count a hit if any entry matches.
    """
    chunk_text = ""
    chunk_page = None

    if isinstance(chunk, str):
        chunk_text = chunk
    elif isinstance(chunk, dict):
        chunk_text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        if isinstance(meta, dict):
            chunk_page = meta.get("page_number", chunk.get("page_number"))
        else:
            chunk_page = getattr(meta, "page_number", None)
    else:
        chunk_text = getattr(chunk, "text", "")
        meta = getattr(chunk, "metadata", None)
        if meta is not None:
            chunk_page = getattr(meta, "page_number", None) if not isinstance(meta, dict) else meta.get("page_number")

    if isinstance(evidence_entry, str):
        ev_text = evidence_entry
        ev_page = None
    elif isinstance(evidence_entry, dict):
        ev_text = evidence_entry.get("evidence_text", "")
        ev_page = evidence_entry.get("evidence_page_num")
    else:
        ev_text = getattr(evidence_entry, "evidence_text", "")
        ev_page = getattr(evidence_entry, "evidence_page_num", None)

    # 1. Page match: chunk's source page metadata equals evidence_page_num
    if ev_page is not None and chunk_page is not None:
        if chunk_page == ev_page:
            return True

    # 2. Text-overlap match: word-level overlap between chunk text and evidence_text is >= 50%
    if ev_text:
        ev_words = set(re.findall(r"\w+", ev_text.lower()))
        if ev_words:
            chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
            overlap = len(ev_words & chunk_words) / len(ev_words)
            if overlap >= text_overlap_threshold:
                return True

        # Fallback substring check (for short evidence phrases)
        probe = _probe(ev_text)
        if probe and probe in chunk_text.lower():
            return True

    return False


def hit_at_k(
    retrieved_items: list[Any],
    evidence_list: list[Any],
    k: int,
    text_overlap_threshold: float = 0.50,
) -> bool:
    """Return True if at least one of the top-k chunks matches any evidence entry."""
    for item in retrieved_items[:k]:
        for ev in evidence_list:
            if chunk_matches_evidence(item, ev, text_overlap_threshold=text_overlap_threshold):
                return True
    return False


def recall_at_k(
    retrieved_items: list[Any],
    evidence_list: list[Any],
    k: int,
    text_overlap_threshold: float = 0.50,
) -> float:
    """Fraction of evidence entries matched in top-k chunks."""
    if not evidence_list:
        return 0.0
    matched = 0
    for ev in evidence_list:
        found = False
        for item in retrieved_items[:k]:
            if chunk_matches_evidence(item, ev, text_overlap_threshold=text_overlap_threshold):
                found = True
                break
        if found:
            matched += 1
    return matched / len(evidence_list)


def mrr(
    retrieved_items: list[Any],
    evidence_list: list[Any],
    text_overlap_threshold: float = 0.50,
) -> float:
    """Mean Reciprocal Rank: 1/rank of first chunk that matches any evidence."""
    for rank, item in enumerate(retrieved_items, start=1):
        for ev in evidence_list:
            if chunk_matches_evidence(item, ev, text_overlap_threshold=text_overlap_threshold):
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
