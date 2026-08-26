"""Ragas evaluation metric runner.

Runs the CRAG pipeline on evaluation questions and computes
Ragas metrics: faithfulness, context_precision, answer_relevancy,
context_recall.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.benchmarks import EvalSample
from src.utils.logging import get_logger

logger = get_logger(__name__)


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

    logger.info(f"Running pipeline on {len(eval_samples)} evaluation samples...")

    for i, sample in enumerate(eval_samples):
        try:
            result = pipeline_fn(sample.question)
            answer = result.get("final_answer") or result.get("generation", "")
            contexts = [
                ctx.get("child_text", "")
                for ctx in result.get("enriched_contexts", [])
            ]

            eval_data["question"].append(sample.question)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append(contexts if contexts else [""])
            eval_data["ground_truth"].append(sample.ground_truth)

            logger.debug(f"Sample {i + 1}/{len(eval_samples)}: OK")

        except Exception as e:
            logger.warning(f"Sample {i + 1} failed: {e}")
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

    # Save aggregate scores
    scores = {m: float(results[m]) for m in metrics if m in results}
    scores_path = output_path / "ragas_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)

    # Save per-sample results
    df = results.to_pandas()
    df.to_csv(output_path / "ragas_per_sample.csv", index=False)

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
