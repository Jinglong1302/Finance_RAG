"""Benchmark dataset loaders for FinanceBench and TAT-QA.

Loads evaluation datasets from HuggingFace, filters to SEC-relevant
questions, and provides stratified dev/eval splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample."""

    question: str
    ground_truth: str
    evidence: str | None = None
    source: str = ""  # "financebench" or "tatqa"
    category: str = ""  # e.g., "factual", "arithmetic", "comparative"
    difficulty: str = ""  # e.g., "easy", "medium", "hard"


def load_financebench(split: str = "all") -> list[EvalSample]:
    """Load the FinanceBench evaluation dataset.

    Args:
        split: "all", "dev" (first 20%), or "eval" (last 80%).

    Returns:
        List of EvalSample objects.
    """
    from datasets import load_dataset

    logger.info("Loading FinanceBench dataset...")

    try:
        dataset = load_dataset("PatronusAI/financebench", split="train")
    except Exception as e:
        logger.error(f"Failed to load FinanceBench: {e}")
        return []

    samples: list[EvalSample] = []
    for item in dataset:
        sample = EvalSample(
            question=item.get("question", ""),
            ground_truth=item.get("answer", ""),
            evidence=item.get("evidence", item.get("document", "")),
            source="financebench",
        )
        if sample.question and sample.ground_truth:
            samples.append(sample)

    logger.info(f"Loaded {len(samples)} FinanceBench samples")

    # Apply split
    return _apply_split(samples, split)


def load_tatqa(split: str = "all") -> list[EvalSample]:
    """Load the TAT-QA evaluation dataset (SEC-relevant subset).

    Args:
        split: "all", "dev" (first 20%), or "eval" (last 80%).

    Returns:
        List of EvalSample objects.
    """
    from datasets import load_dataset

    logger.info("Loading TAT-QA dataset...")

    try:
        dataset = load_dataset("next-tat/TAT-QA", split="train")
    except Exception as e:
        logger.error(f"Failed to load TAT-QA: {e}")
        return []

    samples: list[EvalSample] = []
    for item in dataset:
        question = item.get("question", "")
        answer = item.get("answer", "")

        # Handle list answers (TAT-QA can have multiple correct answers)
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer)

        # Classify operation type
        derivation = item.get("derivation", "")
        if derivation:
            category = "arithmetic"
        else:
            category = "extraction"

        sample = EvalSample(
            question=question,
            ground_truth=str(answer),
            source="tatqa",
            category=category,
        )
        if sample.question and sample.ground_truth:
            samples.append(sample)

    logger.info(f"Loaded {len(samples)} TAT-QA samples")

    return _apply_split(samples, split)


def _apply_split(
    samples: list[EvalSample], split: str
) -> list[EvalSample]:
    """Apply dev/eval split to samples.

    Uses fixed seed for reproducibility. 20% dev, 80% eval.

    Args:
        samples: Full list of samples.
        split: "all", "dev", or "eval".

    Returns:
        Filtered samples based on split.
    """
    if split == "all":
        return samples

    import random

    rng = random.Random(42)  # Fixed seed
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    split_point = int(len(indices) * 0.2)

    if split == "dev":
        selected = indices[:split_point]
    elif split == "eval":
        selected = indices[split_point:]
    else:
        return samples

    result = [samples[i] for i in selected]
    logger.info(
        f"Applied '{split}' split: {len(result)} / {len(samples)} samples"
    )
    return result
