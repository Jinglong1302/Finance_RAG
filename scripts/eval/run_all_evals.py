"""Master runner — aggregates all 4 eval slices into a summary report.

Reads the latest.json from each slice's output directory and writes:
  results/eval/summary_<timestamp>.json
  results/eval/summary_<timestamp>.md

Usage:
    # Run all slices end-to-end and then summarise:
    poetry run python scripts/eval/run_all_evals.py

    # Or run individual slices first, then just summarise:
    poetry run python scripts/eval/run_all_evals.py --summary-only

    # Limit for quick smoke test (each slice limited):
    poetry run python scripts/eval/run_all_evals.py --limit 5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import subprocess
from datetime import datetime

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


def _load_latest(slice_key: str) -> dict | None:
    path = Path(SLICE_RESULTS_DIRS[slice_key]) / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_summary(reports: dict[str, dict | None]) -> dict:
    summary: dict = {}

    ret = reports.get("retrieval")
    if ret:
        crag_agg = ret.get("crag", {}).get("aggregate", {})
        naive_agg = ret.get("naive", {}).get("aggregate", {})
        summary["retrieval"] = {
            "n": ret.get("n_questions", 0),
            "crag": {k: round(v, 4) for k, v in crag_agg.items()},
            "naive": {k: round(v, 4) for k, v in naive_agg.items()} if naive_agg else {},
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
        }

    return summary


def _gate_check(summary: dict) -> dict[str, bool]:
    gates: dict[str, bool] = {}

    ret = summary.get("retrieval", {}).get("crag", {})
    gates["Hit@5"] = ret.get("Hit@5", 0.0) >= THRESHOLDS["Hit@5"]
    gates["MRR"] = ret.get("MRR", 0.0) >= THRESHOLDS["MRR"]

    gen_fb = summary.get("generation_financebench", {})
    ragas = gen_fb.get("crag", {}).get("ragas", {})
    gates["faithfulness"] = ragas.get("faithfulness", 0.0) >= THRESHOLDS["faithfulness"]
    gates["answer_relevancy"] = (
        ragas.get("answer_relevancy", 0.0) >= THRESHOLDS["answer_relevancy"]
    )
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


def _render_markdown(summary: dict, gates: dict[str, bool], ts: str) -> str:
    lines = [
        f"# Finance RAG — Evaluation Summary",
        f"",
        f"**Run timestamp:** `{ts}`  ",
        f"**Corpus:** AAPL, MMM (3M), BA (Boeing), KO (Coca-Cola), NFLX (Netflix), PFE (Pfizer)",
        f"**Pipeline:** CRAG (LangGraph + Qdrant hybrid dense/sparse + BGE-M3 + FlagReranker)",
        f"",
        f"---",
        f"",
        f"## Gate Status",
        f"",
        f"| Gate | Threshold | Status |",
        f"|------|-----------|--------|",
    ]
    for gate, passed in gates.items():
        icon = "✅ PASS" if passed else "❌ FAIL"
        lines.append(f"| {gate} | ≥ {THRESHOLDS[gate]:.2f} | {icon} |")

    all_pass = all(gates.values())
    lines += ["", f"**Overall:** {'✅ ALL GATES PASS' if all_pass else '❌ SOME GATES FAIL'}", ""]

    # Slice 1: Retrieval
    ret = summary.get("retrieval", {})
    if ret:
        lines += [
            "---",
            "",
            f"## Slice 1 — Retrieval (n={ret.get('n',0)})",
            "",
            "| Metric | CRAG | Naive | Delta |",
            "|--------|------|-------|-------|",
        ]
        crag_r = ret.get("crag", {})
        naive_r = ret.get("naive", {})
        for m in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR"):
            cv = crag_r.get(m, 0.0)
            nv = naive_r.get(m, 0.0) if naive_r else None
            delta = f"{cv - nv:+.3f}" if nv is not None else "n/a"
            nv_str = f"{nv:.3f}" if nv is not None else "n/a"
            lines.append(f"| {m} | {cv:.3f} | {nv_str} | {delta} |")

    # Slice 2: Generation FinanceBench
    gen_fb = summary.get("generation_financebench", {})
    if gen_fb:
        lines += [
            "",
            "---",
            "",
            f"## Slice 2 — Generation: FinanceBench + Apple (n={gen_fb.get('n',0)})",
            "",
            "| Metric | CRAG | Naive |",
            "|--------|------|-------|",
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
            f"| Avg Latency (s) | {lat:.1f} | {f'{nlat:.1f}' if nlat is not None else 'n/a'} |"
        )

    # Slice 3: TAT-QA
    gen_tq = summary.get("generation_tatqa", {})
    if gen_tq:
        lines += [
            "",
            "---",
            "",
            f"## Slice 3 — TAT-QA Generation, context-injected (n={gen_tq.get('n',0)})",
            "",
            "| Metric | CRAG | Naive |",
            "|--------|------|-------|",
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

    # Slice 4: Abstention
    abst = summary.get("abstention", {})
    if abst:
        lines += [
            "",
            "---",
            "",
            f"## Slice 4 — Abstention (out-of-corpus, n={abst.get('n',0)})",
            "",
            "| Metric | CRAG | Naive |",
            "|--------|------|-------|",
        ]
        crag_a = abst.get("crag", {})
        naive_a = abst.get("naive", {})
        for m in ("abstention_rate", "false_answer_rate"):
            cv = crag_a.get(m, 0.0)
            nv = naive_a.get(m) if naive_a else None
            nv_str = f"{nv:.3f}" if nv is not None else "n/a"
            lines.append(f"| {m} | {cv:.3f} | {nv_str} |")

    lines += [
        "",
        "---",
        "",
        "> *Golden Rules:* see `docs/CHUNKING_ANALYSIS_REPORT.md` §Golden Rules.  ",
        "> Re-run with `poetry run python scripts/eval/run_all_evals.py` to update.",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all evaluation slices")
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
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--output", default="results/eval")
    args = parser.parse_args()

    if not args.summary_only:
        extra: list[str] = []
        if args.no_baseline:
            extra.append("--no-baseline")
        if args.skip_ragas:
            extra.append("--skip-ragas")

        for slice_key in args.slices:
            script = SLICE_SCRIPTS[slice_key]
            # TAT-QA uses --n instead of --limit
            limit_args = ["--n", str(args.limit)] if slice_key == "generation_tatqa" else []
            if args.limit and slice_key != "generation_tatqa":
                limit_args = ["--limit", str(args.limit)]
            ok = _run_slice(script, None, limit_args + extra)
            if not ok:
                console.print(f"[red]Slice '{slice_key}' failed — continuing[/red]")

    # Merge
    reports: dict[str, dict | None] = {k: _load_latest(k) for k in SLICE_SCRIPTS}
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
    # Also write stable "latest" files
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
