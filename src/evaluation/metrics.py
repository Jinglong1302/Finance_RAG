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


def check_chunk_evidence_detailed(
    chunk: Any,
    evidence_entry: dict[str, Any] | str,
    text_overlap_threshold: float = 0.50,
) -> dict[str, Any]:
    """Check a chunk against a single evidence entry with granular diagnostics.

    Returns dict with keys:
        - is_hit: bool (page_match OR overlap_match)
        - page_match: bool
        - overlap_match: bool
        - overlap_ratio: float
        - chunk_page: int | None
        - evidence_page: int | None
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

    page_match = False
    if ev_page is not None and chunk_page is not None:
        try:
            # Tolerant to +-1 page shift (0-indexed vs 1-indexed, cover pages)
            if chunk_page == ev_page or abs(int(chunk_page) - int(ev_page)) <= 1:
                page_match = True
        except (ValueError, TypeError):
            pass

    overlap_match = False
    overlap_ratio = 0.0
    if ev_text:
        ev_words = set(re.findall(r"\w+", ev_text.lower()))
        if ev_words:
            chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
            overlap_ratio = len(ev_words & chunk_words) / len(ev_words)
            if overlap_ratio >= text_overlap_threshold:
                overlap_match = True

        probe = _probe(ev_text)
        if probe and probe in chunk_text.lower():
            overlap_match = True

    is_hit = page_match or overlap_match
    return {
        "is_hit": is_hit,
        "page_match": page_match,
        "overlap_match": overlap_match,
        "overlap_ratio": round(overlap_ratio, 4),
        "chunk_page": chunk_page,
        "evidence_page": ev_page,
    }


def chunk_matches_evidence(
    chunk: Any,
    evidence_entry: dict[str, Any] | str | list[Any],
    text_overlap_threshold: float = 0.50,
) -> bool:
    """Return True if chunk matches evidence via Page match OR Text-overlap match >= 50%.

    Accepts either a single evidence dict/str or a list of evidence entries.
    """
    if isinstance(evidence_entry, (list, tuple)):
        return any(
            chunk_matches_evidence(chunk, ev, text_overlap_threshold=text_overlap_threshold)
            for ev in evidence_entry
        )
    detailed = check_chunk_evidence_detailed(
        chunk, evidence_entry, text_overlap_threshold=text_overlap_threshold
    )
    return detailed["is_hit"]


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


# ---------------------------------------------------------------------------
# Semantic span equivalence & 4-way classification
# ---------------------------------------------------------------------------


def normalize_span(s: str) -> str:
    """Normalize text for semantic span comparison:
    - Lowercase
    - Strip punctuation
    - Remove English articles (a, an, the)
    - Collapse extra whitespace
    """
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\d\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def span_match(pred: str, gt: str) -> bool:
    """Check semantic/span equivalence between prediction and ground truth.

    1. If numeric match passes via numeric_match, return True.
    2. Normalize pred and gt (remove articles, punctuation, lowercase).
    3. If normalized gt == normalized pred, return True.
    4. If normalized gt is a substring of normalized pred (or vice-versa), return True.
    5. Token overlap: if >=50% of meaningful tokens (length > 2) in ground truth are in prediction.
    """
    if not pred or not gt:
        return False
    from src.evaluation.numeric_eval import extract_numbers, numeric_match

    if numeric_match(pred, gt):
        return True

    norm_p = normalize_span(pred)
    norm_g = normalize_span(gt)
    if not norm_g or not norm_p:
        return False

    if norm_g == norm_p or norm_g in norm_p or norm_p in norm_g:
        return True

    # If ground truth specifies financial numbers (e.g., "$8.70", "0.9%") and prediction also
    # contains financial numbers, but numeric_match failed, do not allow pure token overlap
    # to declare a wrong number as correct on short numerical answers.
    if len(gt.split()) <= 10:
        gt_nums = [n for n in extract_numbers(gt) if not (1900 <= n <= 2099 and n.is_integer())]
        pred_nums = [n for n in extract_numbers(pred) if not (1900 <= n <= 2099 and n.is_integer())]
        if gt_nums and pred_nums:
            return False

    tokens_p = set(norm_p.split())
    tokens_g = set(norm_g.split())
    meaningful_g = {w for w in tokens_g if len(w) > 2}
    if meaningful_g and (len(tokens_p.intersection(meaningful_g)) / len(meaningful_g)) >= 0.5:
        return True

    return False


def span_match_rate(preds: list[str], gts: list[str]) -> float:
    """Proportion of predictions matching ground truth under semantic span equivalence."""
    if not preds or not gts:
        return 0.0
    matches = sum(1 for p, g in zip(preds, gts) if span_match(p, g))
    return matches / len(preds)


def classify_qa_result(
    answer: str,
    ground_truth: str,
    is_indexed: bool = True,
    is_refusal: bool = False,
) -> str:
    """Classify an evaluation outcome into one of 4 clean categories (plus exclusion):
    - 'abstained-correctly-real-gap': Corpus/period genuinely missing, correctly abstained.
    - 'excluded: fiscal year not indexed': Corpus/period missing, but pipeline attempted answer without data.
    - 'abstained-incorrectly': Document/period was indexed, but pipeline gave up / refused.
    - 'answered-correct': Document indexed, answer matches ground truth.
    - 'answered-incorrect': Document indexed, answer does not match ground truth.
    """
    if not is_indexed:
        if is_refusal:
            return "abstained-correctly-real-gap"
        return "excluded: fiscal year not indexed"

    if is_refusal:
        return "abstained-incorrectly"

    if span_match(answer, ground_truth):
        return "answered-correct"
    return "answered-incorrect"
