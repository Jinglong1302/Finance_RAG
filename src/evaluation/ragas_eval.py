"""Ragas evaluation metric runner.

Runs the CRAG pipeline on evaluation questions and computes
Ragas metrics: faithfulness, context_precision, answer_relevancy,
context_recall.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from src.evaluation.benchmarks import EvalSample
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _clean_answer_for_ragas(answer: str) -> str:
    """Clean the raw generated answer for Ragas faithfulness evaluation.

    Strips structured JSON metrics blocks, citation source lists, and
    verification warnings so Ragas decomposes and evaluates the actual
    factual answer claims rather than metadata formatting.
    """
    if not answer:
        return ""
    # Strip markdown code blocks (e.g. ```json ... ```)
    cleaned = re.sub(r"```(?:json)?\s*[\s\S]*?```", "", answer)
    # Strip Sources: or References: sections
    cleaned = re.split(r"\n+(?:Sources?|References?):", cleaned, flags=re.IGNORECASE)[0]
    # Strip Verification Notes
    cleaned = re.split(r"\n+⚠️\s*\*\*Verification Notes:\*\*", cleaned)[0]
    return cleaned.strip()


def _format_contexts_for_ragas(enriched_contexts: list[Any]) -> list[str]:
    """Format retrieved contexts with parent text and metadata for Ragas.

    Mirrors the context provided to the LLM generator so that Ragas sees
    the exact same evidence (including filing headers and expanded parent text).
    """
    formatted: list[str] = []
    for ctx in enriched_contexts:
        if hasattr(ctx, "metadata"):  # EnrichedContext dataclass
            metadata = getattr(ctx, "metadata", {}) or {}
            child = getattr(ctx, "child_text", "") or ""
            parent = getattr(ctx, "parent_text", "") or ""
        elif isinstance(ctx, dict):
            metadata = ctx.get("metadata", {}) or {}
            child = ctx.get("child_text", "") or ""
            parent = ctx.get("parent_text", "") or ""
        else:
            child = str(ctx)
            parent = ""
            metadata = {}

        company = metadata.get("company_ticker", "")
        year = metadata.get("fiscal_year", "")
        section = metadata.get("section_title", "")
        filing = metadata.get("filing_type", "10-K")

        parts = []
        header = f"{company} {filing} (FY{year}) - {section}".strip(" -()")
        if header:
            parts.append(f"[{header}]")
        if child:
            parts.append(child)
        if parent and parent != child:
            p_text = parent[:2000] if len(parent) > 2000 else parent
            parts.append(f"[Expanded Context]\n{p_text}")

        text = "\n".join(parts).strip()
        if text:
            formatted.append(text)

    return formatted if formatted else [""]


def _is_refusal(answer: str) -> bool:
    """Check if an answer is an explicit model refusal/abstention."""
    refusal_phrases = (
        "could not find sufficient evidence",
        "insufficient evidence",
        "not enough information",
        "no information found",
        "unable to locate",
    )
    lower = answer.lower()
    return any(p in lower for p in refusal_phrases)


def run_ragas_evaluation(
    eval_samples: list[EvalSample],
    pipeline_fn: Any,  # Callable that takes a question and returns result dict
    output_dir: str | Path = "results",
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Run Ragas evaluation on a set of samples.

    Args:
        eval_samples: List of EvalSample objects.
        pipeline_fn: Function(question: str) -> dict with keys:
                     'answer', 'contexts' (list[str]), 'ground_truth'.
        output_dir: Directory to save results.
        metrics: List of Ragas metric names to compute.
                Default: ["faithfulness", "context_precision",
                          "answer_relevancy", "context_recall"].

    Returns:
        Dict with aggregate scores and per-sample results.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    if metrics is None:
        metrics = [
            "faithfulness",
            "context_precision",
            "answer_relevancy",
            "context_recall",
        ]

    # Map metric names to objects
    metric_map = {
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "answer_relevancy": answer_relevancy,
        "context_recall": context_recall,
    }
    selected_metrics = [metric_map[m] for m in metrics if m in metric_map]

    # Run pipeline on each sample
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    raw_answers: list[str] = []

    logger.info(f"Running pipeline on {len(eval_samples)} evaluation samples...")

    for i, sample in enumerate(eval_samples):
        try:
            result = pipeline_fn(sample.question)
            raw_answer = result.get("final_answer") or result.get("generation", "")
            raw_answers.append(raw_answer)

            clean_ans = _clean_answer_for_ragas(raw_answer)
            contexts = _format_contexts_for_ragas(result.get("enriched_contexts", []))

            eval_data["question"].append(sample.question)
            eval_data["answer"].append(clean_ans if clean_ans else raw_answer)
            eval_data["contexts"].append(contexts)
            eval_data["ground_truth"].append(sample.ground_truth)

            logger.debug(f"Sample {i + 1}/{len(eval_samples)}: OK")

        except Exception as e:
            logger.warning(f"Sample {i + 1} failed: {e}")
            raw_answers.append(f"Error: {e}")
            eval_data["question"].append(sample.question)
            eval_data["answer"].append(f"Error: {e}")
            eval_data["contexts"].append([""])
            eval_data["ground_truth"].append(sample.ground_truth)

    # Create HuggingFace dataset
    eval_dataset = Dataset.from_dict(eval_data)

    # Run Ragas evaluation
    logger.info(f"Computing Ragas metrics: {metrics}")
    results = evaluate(eval_dataset, metrics=selected_metrics)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save per-sample results with raw answers and refusal flags
    df = results.to_pandas()
    df["raw_answer"] = raw_answers
    df["is_refusal"] = [_is_refusal(a) for a in raw_answers]
    df.to_csv(output_path / "ragas_per_sample.csv", index=False)

    # Save aggregate scores (mean across samples)
    non_refusals = df[~df["is_refusal"]] if "is_refusal" in df.columns else df
    scores: dict[str, float] = {}
    for m in metrics:
        if m in df.columns:
            if m == "faithfulness" and len(non_refusals) > 0:
                scores[m] = float(non_refusals[m].mean())
            else:
                scores[m] = float(df[m].mean())

    if "is_refusal" in df.columns:
        scores["abstention_rate"] = float(df["is_refusal"].mean())

    scores_path = output_path / "ragas_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)

    # Check hard gates
    faithfulness_score = scores.get("faithfulness", 0.0)
    gate_passed = faithfulness_score >= 0.95

    logger.info(
        f"Ragas evaluation complete: {scores} | "
        f"Faithfulness gate: {'PASS' if gate_passed else 'FAIL'} "
        f"({faithfulness_score:.3f} vs 0.95 threshold)"
    )

    return {
        "scores": scores,
        "gate_passed": gate_passed,
        "sample_count": len(eval_samples),
        "output_dir": str(output_path),
    }
