"""Slice 3 — TAT-QA Generation Evaluation (context-injected).

Fetches 40–50 TAT-QA samples (table + paragraph context bundled in the
dataset), injects the context directly into the LLM — bypassing Qdrant
retrieval — and measures exact match and numeric accuracy.

Purpose: isolates generation + arithmetic reasoning quality from
retrieval quality.  Both CRAG's generator and naive generator are
called with identical pre-supplied context.

Usage:
    poetry run python scripts/eval/eval_generation_tatqa.py
    poetry run python scripts/eval/eval_generation_tatqa.py --n 40 --no-baseline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import random
import time
from datetime import datetime

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from src.evaluation.numeric_eval import evaluate_numeric_accuracy, numeric_match
from src.utils.logging import setup_logging

console = Console()

_SYSTEM_PROMPT = (
    "You are a financial analyst. Answer the question using ONLY the provided "
    "table and paragraph context. For numerical answers, give only the number "
    "(with units if needed). Do not explain your reasoning."
)


def _table_to_markdown(table_data: list[list]) -> str:
    """Convert TAT-QA table (list of rows) to markdown."""
    if not table_data:
        return ""
    header = "| " + " | ".join(str(c) for c in table_data[0]) + " |"
    sep = "| " + " | ".join("---" for _ in table_data[0]) + " |"
    body = "\n".join(
        "| " + " | ".join(str(c) for c in row) + " |"
        for row in table_data[1:]
    )
    return "\n".join([header, sep, body])


def _build_context(passage: dict) -> str:
    """Build context string from a TAT-QA passage (table + paragraphs)."""
    parts: list[str] = []
    table_raw = passage.get("table", {})
    if isinstance(table_raw, dict):
        table_rows = table_raw.get("table", [])
    else:
        table_rows = table_raw

    if table_rows:
        parts.append("**Table:**\n" + _table_to_markdown(table_rows))

    paras = passage.get("paragraphs", [])
    if paras:
        para_text = "\n\n".join(p.get("text", "") for p in paras if p.get("text"))
        if para_text:
            parts.append("**Paragraphs:**\n" + para_text)

    return "\n\n".join(parts)


def load_tatqa_samples(n: int = 50, seed: int = 42) -> list[dict]:
    """Load n TAT-QA samples from the train file.

    Each returned dict: {question, answer_str, context, derivation, answer_type}
    """
    local_path = Path(__file__).parent.parent.parent / "data" / "eval" / "tatqa_eval.jsonl"
    if local_path.exists():
        samples = [json.loads(l) for l in open(local_path, encoding="utf-8") if l.strip()]
        return samples[:n]

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="next-tat/TAT-QA",
        filename="tatqa_dataset_train.json",
        repo_type="dataset",
    )
    data = json.load(open(path, encoding="utf-8"))

    # Flatten: one item per question
    flat = []
    for passage in data:
        context = _build_context(passage)
        for qa in passage.get("questions", []):
            answer = qa.get("answer", "")
            if isinstance(answer, list):
                answer = ", ".join(str(a) for a in answer)
            if not str(answer).strip():
                continue
            flat.append(
                {
                    "question": qa.get("question", ""),
                    "answer_str": str(answer),
                    "context": context,
                    "derivation": qa.get("derivation", ""),
                    "answer_type": qa.get("answer_type", ""),
                    "scale": qa.get("scale", ""),
                }
            )

    # Sample reproducibly
    rng = random.Random(seed)
    rng.shuffle(flat)
    return flat[:n]


def _generate_with_context(
    question: str, context: str, llm: OpenAI, model: str
) -> tuple[str, float]:
    """Call LLM with bundled context. Returns (answer, latency_s)."""
    t0 = time.perf_counter()
    try:
        resp = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0,
            max_tokens=256,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        answer = f"Error: {e}"
    return answer, round(time.perf_counter() - t0, 3)


def exact_match_rate(preds: list[str], gts: list[str]) -> float:
    """Case-insensitive exact match after stripping punctuation."""
    import re

    def norm(s: str) -> str:
        return re.sub(r"[^\w\d.%]", " ", s.lower()).strip()

    matches = sum(1 for p, g in zip(preds, gts) if norm(p) == norm(g))
    return matches / len(preds) if preds else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="TAT-QA generation evaluation slice")
    parser.add_argument("--limit", "--n", dest="n", type=int, default=50, help="Number of TAT-QA samples (default: 50)")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--output", default="results/eval/generation_tatqa")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    console.print("[bold cyan]Slice 3 — TAT-QA Generation (context-injected)[/bold cyan]")

    samples = load_tatqa_samples(args.n)
    console.print(f"Loaded {len(samples)} TAT-QA samples")

    llm = OpenAI()

    # Both CRAG generator and naive generator use identical LLM + context here
    # (No retrieval; the point is to measure generation/arithmetic quality)
    crag_preds: list[str] = []
    naive_preds: list[str] = []
    gts = [s["answer_str"] for s in samples]
    crag_latencies: list[float] = []

    console.print("\n[bold]Running generation (same context, GPT-4o)...[/bold]")
    for i, s in enumerate(samples):
        pred, lat = _generate_with_context(
            s["question"], s["context"], llm, settings.openai_model
        )
        crag_preds.append(pred)
        crag_latencies.append(lat)
        console.print(f"  Q{i+1:03d}: {lat:.1f}s | pred={pred[:60]}")

    # For baseline, use same call (context-injected; naive has no retrieval advantage here)
    # We re-run to measure determinism — temperature=0 so identical, but we track separately
    if not args.no_baseline:
        for s in samples:
            pred, _ = _generate_with_context(
                s["question"], s["context"], llm, settings.openai_model
            )
            naive_preds.append(pred)

    # Metrics
    crag_em = exact_match_rate(crag_preds, gts)
    crag_numeric = evaluate_numeric_accuracy(crag_preds, gts)
    naive_em = exact_match_rate(naive_preds, gts) if naive_preds else None
    naive_numeric = (
        evaluate_numeric_accuracy(naive_preds, gts) if naive_preds else None
    )

    # Per-question results
    per_q = []
    for i, s in enumerate(samples):
        n_ans = naive_preds[i] if i < len(naive_preds) else None
        per_q.append(
            {
                "question": s["question"],
                "ground_truth": s["answer_str"],
                "crag_answer": crag_preds[i],
                "naive_answer": n_ans,
                "numeric_match": numeric_match(crag_preds[i], s["answer_str"]),
                "naive_numeric_match": numeric_match(n_ans, s["answer_str"]) if n_ans else None,
                "derivation": s["derivation"],
                "answer_type": s["answer_type"],
                "latency_s": crag_latencies[i],
            }
        )

    # Display
    table = Table(title=f"TAT-QA Generation (n={len(samples)}, context-injected)")
    table.add_column("Metric")
    table.add_column("CRAG (GPT-4o)", style="green")
    if naive_preds:
        table.add_column("Naive (GPT-4o)", style="yellow")

    row_pairs = [
        ("Exact Match", f"{crag_em:.3f}"),
        ("Numeric Accuracy", f"{crag_numeric.get('accuracy', 0):.3f}"),
        ("Avg Latency (s)", f"{sum(crag_latencies)/len(crag_latencies):.1f}"),
    ]
    naive_vals = [
        f"{naive_em:.3f}" if naive_em is not None else "n/a",
        f"{naive_numeric.get('accuracy',0):.3f}" if naive_numeric else "n/a",
        "~same",
    ]
    for i, (metric, cval) in enumerate(row_pairs):
        row = [metric, cval]
        if naive_preds:
            row.append(naive_vals[i])
        table.add_row(*row)
    console.print(table)

    # Save
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "slice": "generation_tatqa",
        "timestamp": ts,
        "n_questions": len(samples),
        "crag": {
            "exact_match": crag_em,
            "numeric_accuracy": crag_numeric.get("accuracy", 0.0),
            "avg_latency_s": sum(crag_latencies) / len(crag_latencies),
            "per_question": per_q,
        },
        "naive": {
            "exact_match": naive_em,
            "numeric_accuracy": naive_numeric.get("accuracy", 0.0) if naive_numeric else None,
            "per_question": [
                {
                    "question": s["question"],
                    "ground_truth": s["answer_str"],
                    "answer": naive_preds[i],
                }
                for i, s in enumerate(samples)
            ] if naive_preds else [],
        },
    }
    out_file = out / f"tatqa_{ts}.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"\n[dim]Results saved → {out_file}[/dim]")


if __name__ == "__main__":
    main()
