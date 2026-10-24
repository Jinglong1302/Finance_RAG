"""Eval question sets: loaders for FinanceBench, AAPL50, OOC20 and TAT-QA.

Moved here from scripts/eval/ (eval_retrieval.py, eval_abstention.py,
eval_generation_financebench.py, eval_generation_tatqa.py) so eval scripts
stop importing each other. The code is unchanged apart from names.

Two different company lists exist on purpose:
- IN_CORPUS_COMPANIES (5): FinanceBench companies whose 10-Ks are indexed.
- OOC_EXCLUDED_COMPANIES (6, adds Apple): companies left out when building
  the out-of-corpus question set.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rich.console import Console

from src.evaluation.benchmarks import EvalSample

console = Console()

# FinanceBench company name → ticker (for Qdrant filter)
CORPUS_TICKER_MAP: dict[str, str] = {
    "3M": "MMM",
    "Boeing": "BA",
    "Coca-Cola": "KO",
    "Netflix": "NFLX",
    "Pfizer": "PFE",
}
IN_CORPUS_COMPANIES = set(CORPUS_TICKER_MAP.keys())

# Companies whose 10-Ks are indexed in Qdrant (used to pick out-of-corpus questions)
OOC_EXCLUDED_COMPANIES = {"3M", "Boeing", "Coca-Cola", "Netflix", "Pfizer", "Apple"}


def load_in_corpus_financebench() -> list[dict]:
    """Load FinanceBench rows for in-corpus companies, returning raw dicts."""
    import json

    local_path = Path(__file__).parent.parent.parent / "data" / "eval" / "financebench_in_corpus.jsonl"
    if local_path.exists():
        return [json.loads(l) for l in open(local_path, encoding="utf-8") if l.strip()]

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="PatronusAI/financebench",
        filename="financebench_merged.jsonl",
        repo_type="dataset",
    )
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("company", "") in IN_CORPUS_COMPANIES]


def extract_evidence_entries(row: dict) -> list[dict[str, Any]]:
    """Pull evidence entries (text + page_num + answer) from a FinanceBench row."""
    evidences = row.get("evidence", [])
    ans = str(row.get("answer", ""))
    if isinstance(evidences, str):
        return [{"evidence_text": evidences, "evidence_page_num": None, "answer": ans}] if evidences else []
    entries = []
    for e in evidences:
        if isinstance(e, dict):
            entries.append({
                "evidence_text": e.get("evidence_text", ""),
                "evidence_page_num": e.get("evidence_page_num"),
                "answer": ans,
            })
        elif isinstance(e, str) and e:
            entries.append({"evidence_text": e, "evidence_page_num": None, "answer": ans})
    return entries


def load_custom_eval() -> list[EvalSample]:
    path = Path(__file__).parent.parent.parent / "data" / "eval" / "custom_eval.jsonl"
    if not path.exists():
        console.print(f"[yellow]custom_eval.jsonl not found at {path}[/yellow]")
        return []
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        samples.append(
            EvalSample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                evidence=item.get("evidence", ""),
                source="custom_aapl",
                category=item.get("category", "extraction"),
                difficulty=item.get("difficulty", ""),
            )
        )
    console.print(f"Loaded {len(samples)} Apple custom QA samples")
    return samples


def load_in_corpus_fb_samples() -> list[tuple[EvalSample, str]]:
    """Returns (EvalSample, ticker) pairs for in-corpus FinanceBench."""
    local_path = Path(__file__).parent.parent.parent / "data" / "eval" / "financebench_in_corpus.jsonl"
    if local_path.exists():
        rows = [json.loads(l) for l in open(local_path, encoding="utf-8") if l.strip()]
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="PatronusAI/financebench",
            filename="financebench_merged.jsonl",
            repo_type="dataset",
        )
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    pairs = []
    for row in rows:
        company = row.get("company", "")
        if company not in IN_CORPUS_COMPANIES:
            continue
        ev = row.get("evidence", [])
        evidence_text = (
            "\n".join(e.get("evidence_text", "") for e in ev if isinstance(e, dict))
            if isinstance(ev, list)
            else str(ev)
        )
        sample = EvalSample(
            question=row.get("question", ""),
            ground_truth=row.get("answer", ""),
            evidence=evidence_text,
            source="financebench",
            category=row.get("question_reasoning", "extraction"),
            doc_period=row.get("doc_period"),
            doc_name=row.get("doc_name"),
        )
        ticker = CORPUS_TICKER_MAP[company]
        pairs.append((sample, ticker))
    console.print(f"Loaded {len(pairs)} in-corpus FinanceBench samples")
    return pairs


def load_out_of_corpus() -> list[dict]:
    local_path = Path(__file__).parent.parent.parent / "data" / "eval" / "abstention_eval.jsonl"
    if local_path.exists():
        return [json.loads(l) for l in open(local_path, encoding="utf-8") if l.strip()]

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="PatronusAI/financebench",
        filename="financebench_merged.jsonl",
        repo_type="dataset",
    )
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("company", "") not in OOC_EXCLUDED_COMPANIES]


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
