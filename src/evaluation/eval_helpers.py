"""Shared eval helpers: run the pipeline, score with Ragas, classify answers.

Moved here from scripts/eval/ (eval_retrieval.py, eval_abstention.py,
eval_generation_financebench.py, eval_generation_tatqa.py) so eval scripts
stop importing each other. The code is unchanged apart from names.
"""

from __future__ import annotations

import time
from typing import Any

from src.orchestration.state import CRAGState

# Refusal phrases for OOC abstention scoring (broader than ragas_eval._is_refusal)
_REFUSAL_PHRASES = (
    "could not find sufficient evidence",
    "insufficient evidence",
    "not enough information",
    "no information found",
    "unable to locate",
    "not available in",
    "not indexed",
    "outside the scope",
    "not in the provided",
    "i don't have",
    "cannot find",
    "no relevant",
    "i was unable",
    "the filing is not",
)

_HEDGE_PHRASES = (
    "it is unclear",
    "might be",
    "could not verify",
    "unverified",
    "possibly",
    "likely",
    "may have",
    "cannot be definitively",
    "not explicitly stated, but",
)


def classify_response(answer: str, state: dict | None = None) -> str:
    """Deterministically classify response into 'abstain', 'hedge', or 'answer'.

    Machine-classifiable without requiring an LLM judge:
    1. 'abstain': Emits [ABSTAIN] prefix, state['is_abstention'] == True, or explicit refusal phrase.
    2. 'hedge': Non-refusal containing speculative or unverified hedge markers.
    3. 'answer': Confident attempt to provide a factual answer.
    """
    if state and state.get("is_abstention"):
        return "abstain"
    if answer.strip().startswith("[ABSTAIN]"):
        return "abstain"

    lower = answer.lower()
    if any(p in lower for p in _REFUSAL_PHRASES):
        return "abstain"

    if any(p in lower for p in _HEDGE_PHRASES):
        return "hedge"

    return "answer"


def check_filing_indexed(client: Any, collection_name: str, ticker: str, fiscal_year: Any, doc_name: str = "") -> bool:
    """Audit whether the required filing is indexed in Qdrant."""
    if not fiscal_year:
        return True
    try:
        fy_int = int(fiscal_year)
    except (ValueError, TypeError):
        return True
    form = "10-Q" if ("10Q" in doc_name.upper() or "10-Q" in doc_name.upper()) else "10-K"
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    flt = Filter(
        must=[
            FieldCondition(key="company_ticker", match=MatchValue(value=ticker)),
            FieldCondition(key="fiscal_year", match=MatchValue(value=fy_int)),
            FieldCondition(key="filing_type", match=MatchValue(value=form)),
        ]
    )
    records, _ = client.scroll(collection_name=collection_name, scroll_filter=flt, limit=1)
    return len(records) > 0


def run_crag(graph: Any, question: str) -> dict[str, Any]:
    from src.orchestration.graph import run_query
    return run_query(graph, question)


def run_ragas(ragas_data: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, float | None]]]:
    import pandas as pd
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    ds = Dataset.from_dict(ragas_data)
    from src.utils.llm_client import get_ragas_judge

    judge_llm, judge_embeddings = get_ragas_judge()
    results = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    scores: dict[str, float] = {}
    df = results.to_pandas()
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for col in cols:
        if col in df.columns:
            valid_vals = df[col].dropna()
            scores[col] = float(valid_vals.mean()) if len(valid_vals) > 0 else 0.0

    per_sample: list[dict[str, float | None]] = []
    for _, row in df.iterrows():
        sample_scores: dict[str, float | None] = {}
        for col in cols:
            if col in df.columns:
                val = row[col]
                sample_scores[col] = float(val) if pd.notna(val) else None
            else:
                sample_scores[col] = None
        per_sample.append(sample_scores)

    return scores, per_sample


def generate_crag_from_context(question: str, context: str) -> tuple[str, float]:
    """CRAG generation path: generator_node (citations, step-by-step arithmetic, structured JSON) + hallucination_guard_node."""
    from src.orchestration.nodes.generator import generator_node
    from src.orchestration.nodes.hallucination_guard import hallucination_guard_node

    t0 = time.perf_counter()
    ctx_dict = {
        "child_text": context,
        "parent_text": context,
        "metadata": {
            "company_ticker": "TAT-QA",
            "section_title": "Financial Table and Statement Context",
            "fiscal_year": "N/A",
            "filing_type": "Report",
        },
    }
    state: CRAGState = {
        "original_query": question,
        "enriched_contexts": [ctx_dict],
        "confidence": "high",
        "cost_accumulated": 0.0,
    }
    try:
        gen_out = generator_node(state)
        state.update(gen_out)
        guard_out = hallucination_guard_node(state)
        answer = guard_out.get("final_answer") or state.get("generation", "")
    except Exception as e:
        answer = f"Error: {e}"
    return answer, round(time.perf_counter() - t0, 3)
