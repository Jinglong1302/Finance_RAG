"""Phase C: Ragas evaluation on final config (v1.0).

Scores the answered subsets of:
  1. FB27 (no injection, Phase A v1.0) — n=16 answered
  2. AAPL50 (Part B, ticker-injected) — n=42 answered

Metrics: faithfulness, context_precision, answer_relevancy, context_recall
Judge model: gpt-4o (same as pipeline)

Procedure:
  - Run 2-sample pilot first to estimate per-sample cost
  - Report estimate, then run full evaluation
  - Disk cache is ENABLED — identical calls reuse cached responses

Baseline (per-chunk-grader config, SHA 94fdd2f):
  faithfulness=0.834, context_precision=0.892, answer_relevancy=0.856, context_recall=0.324
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.settings import get_settings
get_settings()  # sets OPENAI_API_KEY env var before ragas imports it

from src.utils.llm_cache import enable_disk_cache
enable_disk_cache()

from rich.console import Console
from src.evaluation.ragas_eval import _clean_answer_for_ragas, _format_contexts_for_ragas
from src.utils.llm_cache import cache_stats

console = Console()

BASELINE = {
    "config": "per-chunk-grader (94fdd2f)",
    "faithfulness": 0.834,
    "context_precision": 0.892,
    "answer_relevancy": 0.856,
    "context_recall": 0.324,
}


def get_git_info() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    return sha, dirty


def load_fb27_answered() -> list[dict]:
    """Load answered subset from FB27 v1.0 no-injection result."""
    # Use Phase A v1.0 (first no-injection run, not Phase B)
    files = sorted(Path("results/eval").glob("fb27_no_injection_*.json"))
    if not files:
        raise FileNotFoundError("No fb27_no_injection_*.json found")
    # Use the first one (Phase A v1.0, before Phase B trial)
    path = files[0]
    console.print(f"[dim]FB27 source: {path.name}[/dim]")
    data = json.loads(path.read_text(encoding="utf-8"))
    answered = [r for r in data["generation_per_question"] if not r["is_refusal"]]
    return answered


def load_aapl50_answered() -> list[dict]:
    """Load answered subset from AAPL50 Part B result with enriched_contexts."""
    files = sorted(Path("results/eval").glob("part_b_aapl50_*.json"))
    if not files:
        raise FileNotFoundError("No part_b_aapl50_*.json found")
    path = files[-1]  # latest
    console.print(f"[dim]AAPL50 source: {path.name}[/dim]")
    data = json.loads(path.read_text(encoding="utf-8"))
    answered = [r for r in data["per_question"] if not r["is_refusal"]]
    # Check enriched_contexts present
    has_ctx = sum(1 for r in answered if r.get("enriched_contexts"))
    if has_ctx == 0:
        raise ValueError("AAPL50 result has no enriched_contexts — re-run eval_part_b_aapl50.py first")
    console.print(f"[dim]AAPL50 answered: {len(answered)}, with enriched_contexts: {has_ctx}[/dim]")
    return answered


def build_ragas_dataset(records: list[dict]) -> "Dataset":
    from datasets import Dataset

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for r in records:
        raw_answer = r.get("answer", "")
        clean_ans = _clean_answer_for_ragas(raw_answer) or raw_answer
        ctxs = _format_contexts_for_ragas(r.get("enriched_contexts", []))
        data["question"].append(r["question"])
        data["answer"].append(clean_ans)
        data["contexts"].append(ctxs)
        data["ground_truth"].append(str(r["ground_truth"]))

    return Dataset.from_dict(data)


def run_ragas(dataset: "Dataset", label: str) -> dict[str, float]:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    t0 = time.perf_counter()
    stats_before = cache_stats()

    console.print(f"\n[yellow]Running Ragas on {label} (n={len(dataset)})...[/yellow]")
    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, answer_relevancy, context_recall],
    )

    elapsed = round(time.perf_counter() - t0, 1)
    stats_after = cache_stats()
    new_hits = stats_after["hits"] - stats_before["hits"]
    new_misses = stats_after["misses"] - stats_before["misses"]

    def _score(key: str) -> float:
        val = result[key]
        if isinstance(val, (list, tuple)):
            import statistics
            vals = [v for v in val if v is not None]
            return statistics.mean(vals) if vals else float("nan")
        return float(val)

    scores = {
        "faithfulness": _score("faithfulness"),
        "context_precision": _score("context_precision"),
        "answer_relevancy": _score("answer_relevancy"),
        "context_recall": _score("context_recall"),
    }

    console.print(f"  faithfulness:      {scores['faithfulness']:.3f}  (baseline {BASELINE['faithfulness']:.3f})")
    console.print(f"  context_precision: {scores['context_precision']:.3f}  (baseline {BASELINE['context_precision']:.3f})")
    console.print(f"  answer_relevancy:  {scores['answer_relevancy']:.3f}  (baseline {BASELINE['answer_relevancy']:.3f})")
    console.print(f"  context_recall:    {scores['context_recall']:.3f}  (baseline {BASELINE['context_recall']:.3f})")
    console.print(f"  [dim]time={elapsed}s, cache +{new_hits}H/+{new_misses}M[/dim]")

    return scores


def main() -> None:
    sha, dirty = get_git_info()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(f"[bold cyan]Phase C: Ragas Evaluation — git {sha[:7]}, dirty={dirty}[/bold cyan]")
    console.print(f"[bold]Baseline (94fdd2f, per-chunk-grader):[/bold]")
    console.print(f"  faithfulness={BASELINE['faithfulness']:.3f}  context_precision={BASELINE['context_precision']:.3f}")
    console.print(f"  answer_relevancy={BASELINE['answer_relevancy']:.3f}  context_recall={BASELINE['context_recall']:.3f}")

    fb27_records = load_fb27_answered()
    aapl50_records = load_aapl50_answered()

    console.print(f"\n[bold]── 2-Sample Pilot (cost estimate) ──[/bold]")
    pilot_dataset = build_ragas_dataset(fb27_records[:2])
    stats_before = cache_stats()
    t0 = time.perf_counter()
    pilot_scores = run_ragas(pilot_dataset, "FB27 pilot (n=2)")
    pilot_time = round(time.perf_counter() - t0, 1)
    stats_after = cache_stats()
    pilot_misses = stats_after["misses"] - stats_before["misses"]
    console.print(f"\n  Pilot: {pilot_misses} new LLM calls in {pilot_time}s")
    console.print(f"  Estimated full cost: ~{pilot_misses * (len(fb27_records) + len(aapl50_records)) / 2:.0f} calls total")
    console.print(f"  @ ~$0.004/call = ~${pilot_misses * (len(fb27_records) + len(aapl50_records)) / 2 * 0.004:.2f}")

    # Full FB27
    console.print(f"\n[bold]── FB27 (no injection, v1.0) — n={len(fb27_records)} answered ──[/bold]")
    fb27_dataset = build_ragas_dataset(fb27_records)
    fb27_scores = run_ragas(fb27_dataset, f"FB27 no-injection (n={len(fb27_records)})")

    # Full AAPL50
    console.print(f"\n[bold]── AAPL50 (Part B, ticker-injected, v1.0) — n={len(aapl50_records)} answered ──[/bold]")
    aapl50_dataset = build_ragas_dataset(aapl50_records)
    aapl50_scores = run_ragas(aapl50_dataset, f"AAPL50 (n={len(aapl50_records)})")

    stats_final = cache_stats()

    console.print(f"\n[bold]── Summary ──[/bold]")
    console.print(f"{'Metric':<22} {'Baseline':>9} {'FB27 (n=16)':>12} {'AAPL50 (n=42)':>14}")
    console.print("-" * 60)
    for m in ["faithfulness", "context_precision", "answer_relevancy", "context_recall"]:
        b = BASELINE[m]
        f = fb27_scores[m]
        a = aapl50_scores[m]
        console.print(f"  {m:<20} {b:>9.3f} {f:>12.3f} {a:>14.3f}")

    payload = {
        "evaluation": "phase_c_ragas",
        "git_commit": sha,
        "dirty": dirty,
        "timestamp": ts,
        "judge_model": "gpt-4o",
        "config": "v1.0 (pooled sufficiency grader, no ticker injection for FB27, ticker-injected for AAPL50)",
        "baseline": BASELINE,
        "fb27_no_injection": {
            "n": len(fb27_records),
            "source": "fb27_no_injection_Phase_A_v1.0",
            **fb27_scores,
        },
        "aapl50_part_b": {
            "n": len(aapl50_records),
            "source": "part_b_aapl50",
            **aapl50_scores,
        },
        "meta": {
            "cache_total_hits": stats_final["hits"],
            "cache_total_misses": stats_final["misses"],
        },
    }

    out_dir = Path("results/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ragas_phase_c_{ts}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]Saved → {out_file}[/dim]")


if __name__ == "__main__":
    main()
