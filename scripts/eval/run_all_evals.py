"""Master runner — aggregates all evaluation slices into an integrated summary report.

Reads the latest.json from each slice directory (and custom AAPL Ragas results) and writes:
  results/eval/summary_<timestamp>.json
  results/eval/summary_<timestamp>.md
  results/eval/summary_latest.json
  results/eval/summary_latest.md

Includes:
  - Gate checks against targets
  - Aggregate metrics across all slices (Retrieval, Generation, TAT-QA, Abstention, Custom AAPL)
  - Detailed per-question comparisons: Question, Ground Truth, CRAG Answer, Naive Answer

Usage:
    # Run all slices end-to-end and then summarise:
    poetry run python scripts/eval/run_all_evals.py

    # Or summarise existing latest results:
    poetry run python scripts/eval/run_all_evals.py --summary-only

    # Smoke test:
    poetry run python scripts/eval/run_all_evals.py --limit 5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import csv
import json
import subprocess
from datetime import datetime
from typing import Any, Mapping, cast

from rich.console import Console
from rich.table import Table

console = Console()

SLICE_SCRIPTS = {
    "retrieval": "scripts/eval/eval_retrieval.py",
    "generation_financebench": "scripts/eval/eval_generation_financebench.py",
    "generation_tatqa": "scripts/eval/eval_generation_tatqa.py",
    "abstention": "scripts/eval/eval_abstention.py",
}

SLICE_RESULTS_DIRS = {
    "retrieval": "results/eval/retrieval",
    "generation_financebench": "results/eval/generation_financebench",
    "generation_tatqa": "results/eval/generation_tatqa",
    "abstention": "results/eval/abstention",
}

# Gate thresholds (CRAG target)
THRESHOLDS = {
    "Hit@5": 0.60,
    "MRR": 0.50,
    "faithfulness": 0.85,
    "answer_relevancy": 0.75,
    "numeric_accuracy_fb": 0.50,
    "numeric_accuracy_tatqa": 0.55,
    "abstention_rate": 0.70,
}


def _run_slice(script: str, limit: int | None, extra_args: list[str]) -> bool:
    cmd = [sys.executable, script]
    if limit:
        cmd += ["--limit", str(limit)]
    cmd += extra_args
    console.print(f"\n[bold yellow]→ Running: {' '.join(cmd)}[/bold yellow]")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent.parent))
    return result.returncode == 0


def _load_latest(slice_key: str) -> dict[str, Any] | None:
    path = Path(SLICE_RESULTS_DIRS[slice_key]) / "latest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], data)


def _load_custom_aapl() -> dict[str, Any] | None:
    """Load hand-curated Apple custom evaluation results and per-sample CSV."""
    csv_path = Path("results/ragas_per_sample.csv")
    scores_path = Path("results/ragas_scores.json")
    if not csv_path.exists() and not scores_path.exists():
        return None

    scores: dict[str, float] = {}
    if scores_path.exists():
        try:
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    samples: list[dict[str, Any]] = []
    if csv_path.exists():
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append({
                        "question": row.get("user_input", ""),
                        "ground_truth": row.get("reference", ""),
                        "crag_answer": row.get("response", ""),
                        "naive_answer": row.get("naive_response", ""),
                        "raw_answer": row.get("raw_answer", ""),
                        "faithfulness": float(row["faithfulness"]) if row.get("faithfulness") else None,
                        "answer_relevancy": float(row["answer_relevancy"]) if row.get("answer_relevancy") else None,
                        "context_precision": float(row["context_precision"]) if row.get("context_precision") else None,
                        "context_recall": float(row["context_recall"]) if row.get("context_recall") else None,
                        "is_refusal": row.get("is_refusal", "False").lower() == "true",
                    })
        except Exception:
            pass

    return {
        "n_questions": len(samples) if samples else (1 if scores else 0),
        "scores": scores,
        "per_question": samples,
    }


def _build_summary(reports: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    ret = reports.get("retrieval")
    if ret:
        crag_agg = ret.get("crag", {}).get("aggregate", {})
        naive_agg = ret.get("naive", {}).get("aggregate", {})
        summary["retrieval"] = {
            "n": ret.get("n_questions", 0),
            "crag": {k: round(v, 4) for k, v in crag_agg.items()},
            "naive": {k: round(v, 4) for k, v in naive_agg.items()} if naive_agg else {},
            "per_question": ret.get("crag", {}).get("per_question", []),
            "naive_per_question": ret.get("naive", {}).get("per_question", []),
        }

    gen_fb = reports.get("generation_financebench")
    if gen_fb:
        crag = gen_fb.get("crag", {})
        naive = gen_fb.get("naive", {})
        summary["generation_financebench"] = {
            "n": gen_fb.get("n_questions", 0),
            "crag": {
                "ragas": crag.get("ragas", {}),
                "numeric_accuracy": crag.get("numeric_accuracy", 0.0),
                "avg_latency_s": crag.get("avg_latency_s", 0.0),
            },
            "naive": {
                "numeric_accuracy": naive.get("numeric_accuracy"),
                "avg_latency_s": naive.get("avg_latency_s"),
            },
            "per_question": crag.get("per_question", []),
            "naive_per_question": naive.get("per_question", []),
        }

    gen_tq = reports.get("generation_tatqa")
    if gen_tq:
        crag = gen_tq.get("crag", {})
        naive = gen_tq.get("naive", {})
        summary["generation_tatqa"] = {
            "n": gen_tq.get("n_questions", 0),
            "crag": {
                "exact_match": crag.get("exact_match", 0.0),
                "numeric_accuracy": crag.get("numeric_accuracy", 0.0),
                "avg_latency_s": crag.get("avg_latency_s", 0.0),
            },
            "naive": {
                "exact_match": naive.get("exact_match"),
                "numeric_accuracy": naive.get("numeric_accuracy"),
            },
            "per_question": crag.get("per_question", []),
            "naive_per_question": naive.get("per_question", []),
        }

    abst = reports.get("abstention")
    if abst:
        crag = abst.get("crag", {})
        naive = abst.get("naive", {})
        summary["abstention"] = {
            "n": abst.get("n_questions", 0),
            "crag": {
                "abstention_rate": crag.get("abstention_rate", 0.0),
                "false_answer_rate": crag.get("false_answer_rate", 1.0),
            },
            "naive": {
                "abstention_rate": naive.get("abstention_rate"),
                "false_answer_rate": naive.get("false_answer_rate"),
            },
            "per_question": crag.get("per_question", []),
            "naive_per_question": naive.get("per_question", []),
        }

    aapl = reports.get("custom_aapl")
    if aapl:
        summary["custom_aapl"] = {
            "n": aapl.get("n_questions", len(aapl.get("per_question", []))),
            "scores": aapl.get("scores", {}),
            "per_question": aapl.get("per_question", []),
        }

    return summary


def _gate_check(summary: dict) -> dict[str, bool]:
    gates: dict[str, bool] = {}

    ret = summary.get("retrieval", {}).get("crag", {})
    gates["Hit@5"] = ret.get("Hit@5", 0.0) >= THRESHOLDS["Hit@5"]
    gates["MRR"] = ret.get("MRR", 0.0) >= THRESHOLDS["MRR"]

    gen_fb = summary.get("generation_financebench", {})
    ragas_fb = gen_fb.get("crag", {}).get("ragas", {})
    aapl_scores = summary.get("custom_aapl", {}).get("scores", {})

    # Evaluate Ragas against FB if available, or fall back to AAPL Ragas scores
    faith_val = ragas_fb.get("faithfulness")
    if faith_val is None:
        faith_val = aapl_scores.get("faithfulness", 0.0)
    gates["faithfulness"] = faith_val >= THRESHOLDS["faithfulness"]

    relev_val = ragas_fb.get("answer_relevancy")
    if relev_val is None:
        relev_val = aapl_scores.get("answer_relevancy", 0.0)
    gates["answer_relevancy"] = relev_val >= THRESHOLDS["answer_relevancy"]

    gates["numeric_accuracy_fb"] = (
        gen_fb.get("crag", {}).get("numeric_accuracy", 0.0)
        >= THRESHOLDS["numeric_accuracy_fb"]
    )

    gen_tq = summary.get("generation_tatqa", {})
    gates["numeric_accuracy_tatqa"] = (
        gen_tq.get("crag", {}).get("numeric_accuracy", 0.0)
        >= THRESHOLDS["numeric_accuracy_tatqa"]
    )

    abst = summary.get("abstention", {})
    gates["abstention_rate"] = (
        abst.get("crag", {}).get("abstention_rate", 0.0)
        >= THRESHOLDS["abstention_rate"]
    )

    return gates


def _format_quote(text: str) -> str:
    """Format multiline text cleanly into markdown blockquote."""
    if not text:
        return "> *(No answer generated)*"
    lines = text.strip().split("\n")
    return "\n".join(f"> {line}" for line in lines)


def _render_markdown(summary: dict, gates: dict[str, bool], ts: str) -> str:
    lines = [
        f"# Finance RAG — Integrated Evaluation Report",
        f"",
        f"**Run timestamp:** `{ts}`  ",
        f"**Corpus:** AAPL, MMM (3M), BA (Boeing), KO (Coca-Cola), NFLX (Netflix), PFE (Pfizer)",
        f"**Pipelines Evaluated:** CRAG (Advanced LangGraph Agent) vs. Naive RAG (Dense Baseline)",
        f"",
        f"---",
        f"",
        f"## 1. Executive Gate Status",
        f"",
        f"| Gate Metric | Threshold | Evaluated Value | Status |",
        f"| :--- | :---: | :---: | :---: |",
    ]

    # Map gate to actual score
    ret_crag = summary.get("retrieval", {}).get("crag", {})
    fb_crag = summary.get("generation_financebench", {}).get("crag", {})
    tq_crag = summary.get("generation_tatqa", {}).get("crag", {})
    abst_crag = summary.get("abstention", {}).get("crag", {})
    aapl_scores = summary.get("custom_aapl", {}).get("scores", {})

    actual_values = {
        "Hit@5": ret_crag.get("Hit@5", 0.0),
        "MRR": ret_crag.get("MRR", 0.0),
        "faithfulness": fb_crag.get("ragas", {}).get("faithfulness") or aapl_scores.get("faithfulness", 0.0),
        "answer_relevancy": fb_crag.get("ragas", {}).get("answer_relevancy") or aapl_scores.get("answer_relevancy", 0.0),
        "numeric_accuracy_fb": fb_crag.get("numeric_accuracy", 0.0),
        "numeric_accuracy_tatqa": tq_crag.get("numeric_accuracy", 0.0),
        "abstention_rate": abst_crag.get("abstention_rate", 0.0),
    }

    for gate, passed in gates.items():
        icon = "✅ PASS" if passed else "❌ FAIL"
        val = actual_values.get(gate, 0.0)
        lines.append(f"| `{gate}` | ≥ {THRESHOLDS[gate]:.2f} | **{val:.3f}** | {icon} |")

    all_pass = all(gates.values())
    lines += ["", f"**Overall Status:** {'✅ ALL GATES PASS' if all_pass else '❌ SOME GATES FAIL'}", ""]

    # Slice 1: Retrieval
    ret = summary.get("retrieval", {})
    if ret:
        lines += [
            "---",
            "",
            f"## 2. Slice 1 — Retrieval Benchmark (FinanceBench In-Corpus, n={ret.get('n',0)})",
            "",
            r"| Metric | CRAG | Naive RAG | Delta ($\Delta$) |",
            "| :--- | :---: | :---: | :---: |",
        ]
        crag_r = ret.get("crag", {})
        naive_r = ret.get("naive", {})
        for m in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR"):
            cv = crag_r.get(m, 0.0)
            nv = naive_r.get(m, 0.0) if naive_r else None
            delta = f"{cv - nv:+.3f}" if nv is not None else "n/a"
            nv_str = f"{nv:.3f}" if nv is not None else "n/a"
            lines.append(f"| {m} | {cv:.3f} | {nv_str} | {delta} |")

        # Per question table
        pq = ret.get("per_question", [])
        npq = ret.get("naive_per_question", [])
        if pq:
            lines += [
                "",
                "### Per-Question Retrieval Performance",
                "",
                "| # | Ticker | Question | CRAG Hit@5 | CRAG MRR | Naive Hit@5 | Naive MRR | Latency |",
                "| -: | :---: | :--- | :---: | :---: | :---: | :---: | -: |",
            ]
            for i, item in enumerate(pq):
                n_item = npq[i] if i < len(npq) else {}
                c_hit = "✅ 1" if item.get("Hit@5") else "❌ 0"
                n_hit = "✅ 1" if n_item.get("Hit@5") else "❌ 0"
                c_mrr = f"{item.get('MRR', 0.0):.2f}"
                n_mrr = f"{n_item.get('MRR', 0.0):.2f}" if n_item else "n/a"
                q_text = item.get("question", "")
                lines.append(
                    f"| {i+1} | `{item.get('ticker', '')}` | {q_text} | {c_hit} | {c_mrr} | {n_hit} | {n_mrr} | {item.get('latency_s', 0):.1f}s |"
                )

            # Detailed Retrieved Chunks Comparison
            lines += ["", "### Retrieved Chunks Detail (CRAG vs. Naive RAG)", ""]
            for i, item in enumerate(pq):
                n_item = npq[i] if i < len(npq) else {}
                q_text = item.get("question", "")
                c_chunks = item.get("retrieved_chunks", [])
                n_chunks = n_item.get("retrieved_chunks", [])

                lines += [
                    f"#### Q{i+1}: [{item.get('ticker', '')}] {q_text}",
                    "",
                    "**CRAG Retrieved Chunks (Top-5 via Hybrid + Reranker):**",
                    "",
                    "| Rank | Page | Section | Score | Match? | Text Preview |",
                    "| -: | -: | :--- | -: | :---: | :--- |",
                ]
                if c_chunks:
                    for c in c_chunks:
                        m_icon = "✅ HIT" if c.get("is_hit") else "❌ miss"
                        lines.append(
                            f"| {c.get('rank', 0)} | {c.get('page_number', 'N/A')} | {str(c.get('section', 'N/A'))[:35]} | {c.get('score', 0):.4f} | {m_icon} | {c.get('text_preview', '')} |"
                        )
                else:
                    lines.append("| - | - | - | - | - | *(No chunks recorded)* |")

                lines += [
                    "",
                    "**Naive RAG Retrieved Chunks (Top-5 via Dense-Only):**",
                    "",
                    "| Rank | Page | Section | Score | Match? | Text Preview |",
                    "| -: | -: | :--- | -: | :---: | :--- |",
                ]
                if n_chunks:
                    for c in n_chunks:
                        m_icon = "✅ HIT" if c.get("is_hit") else "❌ miss"
                        lines.append(
                            f"| {c.get('rank', 0)} | {c.get('page_number', 'N/A')} | {str(c.get('section', 'N/A'))[:35]} | {c.get('score', 0):.4f} | {m_icon} | {c.get('text_preview', '')} |"
                        )
                else:
                    lines.append("| - | - | - | - | - | *(No chunks recorded)* |")
                lines.append("")

    # Slice 2: Generation FinanceBench
    gen_fb = summary.get("generation_financebench", {})
    if gen_fb:
        lines += [
            "",
            "---",
            "",
            f"## 3. Slice 2 — End-to-End Generation (FinanceBench In-Corpus, n={gen_fb.get('n',0)})",
            "",
            "| Metric | CRAG | Naive RAG |",
            "| :--- | :---: | :---: |",
        ]
        crag_g = gen_fb.get("crag", {})
        naive_g = gen_fb.get("naive", {})
        for metric, val in crag_g.get("ragas", {}).items():
            lines.append(f"| {metric} | {val:.3f} | n/a |")
        na = crag_g.get("numeric_accuracy", 0.0)
        nn = naive_g.get("numeric_accuracy") if naive_g else None
        lines.append(
            f"| Numeric Accuracy | {na:.3f} | {f'{nn:.3f}' if nn is not None else 'n/a'} |"
        )
        lat = crag_g.get("avg_latency_s", 0.0)
        nlat = naive_g.get("avg_latency_s") if naive_g else None
        lines.append(
            f"| Avg Latency (s) | {lat:.1f}s | {f'{nlat:.1f}s' if nlat is not None else 'n/a'} |"
        )

        pq = gen_fb.get("per_question", [])
        npq = gen_fb.get("naive_per_question", [])
        if pq:
            lines += ["", "### Per-Question Comparison: Pipeline Answer vs. Ground Truth", ""]
            for i, item in enumerate(pq):
                q = item.get("question", "")
                gt = item.get("ground_truth", "")
                ans = item.get("answer", "")
                n_ans = npq[i].get("answer", "") if i < len(npq) else None

                lines += [
                    f"#### Q{i+1}: {q}",
                    f"",
                    f"**Ground Truth:** `{gt}`",
                ]
                f_score = item.get("faithfulness")
                ar_score = item.get("answer_relevancy")
                ragas_pills = []
                if f_score is not None:
                    ragas_pills.append(f"Faithfulness: `{f_score:.3f}`")
                if ar_score is not None:
                    ragas_pills.append(f"Answer Relevancy: `{ar_score:.3f}`")
                if ragas_pills:
                    lines += ["", f"**Ragas (CRAG):** {' | '.join(ragas_pills)}"]

                lines += [
                    f"",
                    f"**CRAG Pipeline Answer:**",
                    _format_quote(ans),
                ]
                if n_ans:
                    lines += [
                        f"",
                        f"**Naive RAG Answer:**",
                        _format_quote(n_ans),
                    ]
                lines.append("")

    # Slice 5: Custom Apple (AAPL)
    aapl = summary.get("custom_aapl", {})
    if aapl:
        a_scores = aapl.get("scores", {})
        lines += [
            "",
            "---",
            "",
            f"## 4. Slice 5 — Custom Apple (AAPL) Evaluation (n={aapl.get('n',0)})",
            "",
            "| Ragas Metric | Score | Target | Status |",
            "| :--- | :---: | :---: | :---: |",
        ]
        for m, target in (
            ("faithfulness", 0.85),
            ("answer_relevancy", 0.75),
            ("context_precision", 0.70),
            ("context_recall", 0.70),
        ):
            val = a_scores.get(m, 0.0)
            status = "✅ PASS" if val >= target else "❌ FAIL"
            lines.append(f"| `{m}` | **{val:.3f}** | ≥ {target:.2f} | {status} |")

        pq = aapl.get("per_question", [])
        if pq:
            lines += ["", "### Per-Question Comparison: Apple Custom QA", ""]
            for i, item in enumerate(pq):
                lines += [
                    f"#### Q{i+1}: {item.get('question', '')}",
                    f"",
                    f"**Ground Truth:** `{item.get('ground_truth', '')}`  ",
                    f"**Ragas Scores:** Faithfulness: `{item.get('faithfulness', 0):.2f}` | Relevancy: `{item.get('answer_relevancy', 0):.2f}` | Precision: `{item.get('context_precision', 0):.2f}`",
                    f"",
                    f"**CRAG Pipeline Answer:**",
                    _format_quote(item.get("crag_answer", "")),
                ]
                if item.get("naive_answer"):
                    lines += [
                        f"",
                        f"**Naive RAG Answer:**",
                        _format_quote(item.get("naive_answer", "")),
                    ]
                lines.append("")

    # Slice 3: TAT-QA
    gen_tq = summary.get("generation_tatqa", {})
    if gen_tq:
        lines += [
            "",
            "---",
            "",
            f"## 5. Slice 3 — Isolated Table Reasoning (TAT-QA, Context-Injected, n={gen_tq.get('n',0)})",
            "",
            "| Metric | CRAG | Naive RAG |",
            "| :--- | :---: | :---: |",
        ]
        crag_t = gen_tq.get("crag", {})
        naive_t = gen_tq.get("naive", {})
        n_em = naive_t.get("exact_match")
        n_na = naive_t.get("numeric_accuracy")
        lines.append(
            f"| Exact Match | {crag_t.get('exact_match',0):.3f} | "
            + (f"{n_em:.3f}" if n_em is not None else "n/a") + " |"
        )
        lines.append(
            f"| Numeric Accuracy | {crag_t.get('numeric_accuracy',0):.3f} | "
            + (f"{n_na:.3f}" if n_na is not None else "n/a") + " |"
        )

        pq = gen_tq.get("per_question", [])
        if pq:
            lines += ["", "### Per-Question Comparison: Table Math & Extraction", ""]
            for i, item in enumerate(pq):
                match_icon = "✅ Match" if item.get("numeric_match") else "❌ Mismatch"
                deriv = f" (Derivation: `{item.get('derivation')}`)" if item.get("derivation") else ""
                n_ans = item.get("naive_answer")
                lines += [
                    f"#### Q{i+1}: {item.get('question', '')}",
                    f"",
                    f"**Ground Truth:** `{item.get('ground_truth', '')}`{deriv}  ",
                    f"- **CRAG Answer:** `{item.get('crag_answer', '')}` — **{match_icon}**",
                ]
                if n_ans is not None:
                    n_match = "✅ Match" if item.get("naive_numeric_match") else "❌ Mismatch"
                    lines.append(f"- **Naive RAG Answer:** `{n_ans}` — **{n_match}**")
                lines.append("")

    # Slice 4: Abstention
    abst = summary.get("abstention", {})
    if abst:
        lines += [
            "",
            "---",
            "",
            f"## 6. Slice 4 — Abstention & Safety (Out-of-Corpus, n={abst.get('n',0)})",
            "",
            "| Metric | CRAG (Two-Stage Gate) | Naive RAG (Baseline) |",
            "| :--- | :---: | :---: |",
        ]
        crag_a = abst.get("crag", {})
        naive_a = abst.get("naive", {})
        for m in ("abstention_rate", "false_answer_rate"):
            cv = crag_a.get(m, 0.0)
            nv = naive_a.get(m) if naive_a else None
            nv_str = f"{nv:.3f}" if nv is not None else "n/a"
            lines.append(f"| {m} | {cv:.3f} | {nv_str} |")

        pq = abst.get("per_question", [])
        npq = abst.get("naive_per_question", [])
        if pq:
            lines += ["", "### Per-Question Comparison: Abstention vs. Hallucination", ""]
            for i, item in enumerate(pq):
                n_item = npq[i] if i < len(npq) else {}
                c_status = "✅ Abstain" if item.get("refused") else "❌ Hallucinate"
                n_status = "✅ Abstain" if n_item.get("refused") else "❌ Hallucinate"
                lines += [
                    f"#### Q{i+1}: [{item.get('company', '')}] {item.get('question', '')}",
                    f"- **Expected Behavior:** Safe Abstention (Ticker unindexed)",
                    f"- **CRAG Result ({c_status}):**",
                    f"  `{item.get('answer_preview', '').replace(chr(10), ' ')}`",
                    f"- **Naive RAG Result ({n_status}):**",
                    f"  `{n_item.get('answer_preview', '').replace(chr(10), ' ')}`",
                    "",
                ]

    lines += [
        "",
        "---",
        "",
        "> *Evaluation Architecture & Methodology:* see `docs/CHUNKING_ANALYSIS_REPORT.md` and `docs/EVALUATION_50_QA_GUIDE.md`.  ",
        "> Re-run with `poetry run python scripts/eval/run_all_evals.py` to refresh all slices.",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all evaluation slices and generate integrated summary report")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip running scripts; just merge existing latest.json files",
    )
    parser.add_argument("--limit", type=int, default=None, help="Per-slice question limit")
    parser.add_argument(
        "--slices",
        nargs="+",
        choices=list(SLICE_SCRIPTS.keys()),
        default=list(SLICE_SCRIPTS.keys()),
        help="Which slices to run (default: all)",
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-reranker", action="store_true", help="Bypass cross-encoder reranker in all slices")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--output", default="results/eval")
    args = parser.parse_args()

    if not args.summary_only:
        extra: list[str] = []
        if args.no_baseline:
            extra.append("--no-baseline")
        if args.no_reranker:
            extra.append("--no-reranker")
        if args.skip_ragas:
            extra.append("--skip-ragas")

        for slice_key in args.slices:
            script = SLICE_SCRIPTS[slice_key]
            limit_args = ["--limit", str(args.limit)] if args.limit else []
            ok = _run_slice(script, None, limit_args + extra)
            if not ok:
                console.print(f"[red]Slice '{slice_key}' failed — continuing[/red]")

    # Merge all slice reports
    reports: dict[str, dict[str, Any] | None] = {k: _load_latest(k) for k in SLICE_SCRIPTS}

    # Load custom AAPL results if available
    aapl_report = _load_custom_aapl()
    if aapl_report:
        reports["custom_aapl"] = aapl_report

    available = {k: v for k, v in reports.items() if v is not None}
    if not available:
        console.print("[red]No result files found. Run the slice scripts first.[/red]")
        return

    summary = _build_summary(available)
    gates = _gate_check(summary)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"summary_{ts}.json"
    md_path = out / f"summary_{ts}.md"
    latest_json = out / "summary_latest.json"
    latest_md = out / "summary_latest.md"

    full_report = {"timestamp": ts, "gates": gates, "summary": summary}
    for p in (json_path, latest_json):
        p.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    md_content = _render_markdown(summary, gates, ts)
    for p in (md_path, latest_md):
        p.write_text(md_content, encoding="utf-8")

    # Rich summary table
    table = Table(title="Evaluation Summary — Gate Check")
    table.add_column("Gate")
    table.add_column("Threshold")
    table.add_column("Status")
    for gate, passed in gates.items():
        icon = "✅ PASS" if passed else "❌ FAIL"
        color = "green" if passed else "red"
        table.add_row(gate, f"≥ {THRESHOLDS[gate]:.2f}", f"[{color}]{icon}[/{color}]")
    console.print(table)

    all_pass = all(gates.values())
    status = "[green]ALL GATES PASS ✅[/green]" if all_pass else "[red]SOME GATES FAIL ❌[/red]"
    console.print(f"\nOverall: {status}")
    console.print(f"[dim]JSON → {json_path}[/dim]")
    console.print(f"[dim]MD   → {md_path}[/dim]")


if __name__ == "__main__":
    main()
