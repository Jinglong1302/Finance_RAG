"""Slice 1 — Retrieval Evaluation.

Compares CRAG hybrid retrieval (BGE-M3 dense+sparse + RRF + FlagReranker)
against naive dense-only baseline using FinanceBench evidence as ground truth.

In-corpus companies: 3M (MMM), Boeing (BA), Coca-Cola (KO), Netflix (NFLX),
Pfizer (PFE).  Evidence match = FinanceBench evidence_text substring found
in a retrieved chunk.

Metrics: Hit@1, Hit@3, Hit@5, Hit@10, Recall@5, Recall@10, MRR.

Usage:
    poetry run python scripts/eval/eval_retrieval.py
    poetry run python scripts/eval/eval_retrieval.py --limit 10 --no-baseline
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from src.evaluation.baseline import NaiveRAG
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.reranker import CrossEncoderReranker

from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from src.evaluation.metrics import (
    aggregate_retrieval_metrics,
    hit_at_k,
    mrr,
    recall_at_k,
)
from src.utils.logging import setup_logging

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


def evaluate_one(
    question: str,
    evidence_list: Sequence[dict[str, Any] | str],
    ticker: str,
    searcher: HybridSearcher,
    reranker: CrossEncoderReranker | None,
    expander: Any,
    top_k_retrieve: int = 25,
    top_k_rerank: int = 10,
) -> dict[str, Any]:
    from src.orchestration.nodes.query_decomposer import query_decomposer_node
    from src.orchestration.nodes.retriever import make_retriever_node
    from src.orchestration.nodes.reranker_node import make_reranker_node
    from src.orchestration.state import CRAGState

    state: CRAGState = {
        "original_query": question,
        "cycle_count": 0,
        "cost_accumulated": 0.0,
        "pipeline_trace": [],
        "result_count": top_k_rerank,
    }

    # 1. Query decomposition
    dec_out = query_decomposer_node(state)
    state.update(dec_out)

    # Ensure company_ticker filter is set for eval isolation
    filters = state.setdefault("structured_filters", {})
    if ticker and "company_ticker" not in filters:
        filters["company_ticker"] = ticker

    # 2. Real retriever node (subqueries + original query + section prefetch + RRF)
    ret_node = make_retriever_node(searcher)
    ret_out = ret_node(state)
    state.update(ret_out)
    candidates = state.get("search_results", [])

    # 3. Real reranker node (cross-encoder + table-aware diversity + parent expansion)
    rerank_node = make_reranker_node(reranker, expander)
    rerank_out = rerank_node(state)
    state.update(rerank_out)
    reranked = state.get("reranked_results", [])
    if not reranked and candidates:
        reranked = candidates[:top_k_rerank]

    from src.evaluation.metrics import check_chunk_evidence_detailed, hit_at_k, mrr, recall_at_k

    result: dict[str, Any] = {"question": question, "ticker": ticker}
    for k in (1, 3, 5, 10):
        result[f"Hit@{k}"] = int(hit_at_k(reranked, evidence_list, k))
    result["Recall@5"] = recall_at_k(reranked, evidence_list, 5)
    result["Recall@10"] = recall_at_k(reranked, evidence_list, 10)
    result["MRR"] = mrr(reranked, evidence_list)

    gold_rank = None
    for r_idx, c in enumerate(reranked):
        details = [check_chunk_evidence_detailed(c, ev) for ev in evidence_list]
        if any(d["is_hit"] for d in details):
            gold_rank = r_idx + 1
            break
    result["gold_evidence_rank"] = gold_rank
    result["post_rerank_rank"] = gold_rank

    gold_candidate_rank = None
    gold_dense_rank = None
    gold_sparse_rank = None
    gold_rrf_rank = None
    for r_idx, c in enumerate(candidates):
        details = [check_chunk_evidence_detailed(c, ev) for ev in evidence_list]
        if any(d["is_hit"] for d in details):
            gold_candidate_rank = r_idx + 1
            meta = c.get("metadata", {}) if isinstance(c, dict) else getattr(c, "metadata", {})
            gold_dense_rank = meta.get("dense_rank")
            gold_sparse_rank = meta.get("sparse_rank")
            gold_rrf_rank = meta.get("rrf_rank")
            break

    # If not in top candidates, check searcher's broader query diagnostic pools
    if gold_candidate_rank is None and hasattr(searcher, "_last_diagnostics"):
        diag = getattr(searcher, "_last_diagnostics", {})
        for idx, p in enumerate(diag.get("dense_points", [])):
            c_dict = {"text": (p.payload or {}).get("text", ""), "metadata": p.payload or {}}
            if any(check_chunk_evidence_detailed(c_dict, ev)["is_hit"] for ev in evidence_list):
                gold_dense_rank = idx + 1
                break
        for idx, p in enumerate(diag.get("sparse_points", [])):
            c_dict = {"text": (p.payload or {}).get("text", ""), "metadata": p.payload or {}}
            if any(check_chunk_evidence_detailed(c_dict, ev)["is_hit"] for ev in evidence_list):
                gold_sparse_rank = idx + 1
                break
        sorted_pids = diag.get("sorted_pids", [])
        point_map = diag.get("point_map", {})
        for idx, pid in enumerate(sorted_pids):
            p = point_map.get(pid)
            if p:
                c_dict = {"text": (p.payload or {}).get("text", ""), "metadata": p.payload or {}}
                if any(check_chunk_evidence_detailed(c_dict, ev)["is_hit"] for ev in evidence_list):
                    gold_rrf_rank = idx + 1
                    break

    result["gold_candidate_rank"] = gold_candidate_rank
    result["in_top_25"] = bool(gold_candidate_rank is not None and gold_candidate_rank <= 25)
    result["gold_dense_rank"] = gold_dense_rank
    result["gold_sparse_rank"] = gold_sparse_rank
    result["gold_rrf_rank"] = gold_rrf_rank

    chunks_diag = []
    for rank, c in enumerate(reranked[:5]):
        details = [check_chunk_evidence_detailed(c, ev) for ev in evidence_list]
        is_hit = any(d["is_hit"] for d in details)
        page_match = any(d["page_match"] for d in details)
        overlap_match = any(d["overlap_match"] for d in details)
        containment_match = any(d.get("containment_match", False) for d in details)
        number_match = any(d.get("number_match", False) for d in details)
        max_overlap = max((d["overlap_ratio"] for d in details), default=0.0)
        pg_num = getattr(c, "metadata", {}).get("page_number") if hasattr(c, "metadata") else c.get("metadata", {}).get("page_number")
        sec = (getattr(c, "metadata", {}).get("section_title") or getattr(c, "metadata", {}).get("section_id", "N/A")) if hasattr(c, "metadata") else (c.get("metadata", {}).get("section_title") or c.get("metadata", {}).get("section_id", "N/A"))
        sc = round(float(getattr(c, "score", 0.0) if hasattr(c, "score") else c.get("score", 0.0)), 4)
        txt = (getattr(c, "text", "") if hasattr(c, "text") else c.get("text", ""))[:140].replace("\n", " ") + "..."

        chunks_diag.append({
            "rank": rank + 1,
            "page_number": pg_num,
            "section": sec,
            "score": sc,
            "text_preview": txt,
            "is_hit": is_hit,
            "page_match": page_match,
            "overlap_match": overlap_match,
            "containment_match": containment_match,
            "number_match": number_match,
            "overlap_ratio": max_overlap,
        })
    result["retrieved_chunks"] = chunks_diag
    return result


def evaluate_one_naive(
    question: str,
    evidence_list: Sequence[dict[str, Any] | str],
    ticker: str,
    naive: NaiveRAG,
    top_k: int = 10,
) -> dict[str, Any]:
    """Dense-only retrieval for one question."""
    chunks = naive.retrieve(question, ticker=ticker)

    from src.evaluation.metrics import check_chunk_evidence_detailed, hit_at_k, mrr, recall_at_k

    result: dict[str, Any] = {"question": question, "ticker": ticker}
    for k in (1, 3, 5, 10):
        result[f"Hit@{k}"] = int(hit_at_k(chunks, evidence_list, k))
    result["Recall@5"] = recall_at_k(chunks, evidence_list, 5)
    result["Recall@10"] = recall_at_k(chunks, evidence_list, 10)
    result["MRR"] = mrr(chunks, evidence_list)

    chunks_diag = []
    for rank, c in enumerate(chunks[:5]):
        details = [check_chunk_evidence_detailed(c, ev) for ev in evidence_list]
        is_hit = any(d["is_hit"] for d in details)
        page_match = any(d["page_match"] for d in details)
        overlap_match = any(d["overlap_match"] for d in details)
        max_overlap = max((d["overlap_ratio"] for d in details), default=0.0)
        pg_num = c.get("metadata", {}).get("page_number") if isinstance(c, dict) else getattr(c, "metadata", {}).get("page_number")
        sec = (c.get("metadata", {}).get("section_title") or c.get("metadata", {}).get("section_id", "N/A")) if isinstance(c, dict) else (getattr(c, "metadata", {}).get("section_title") or getattr(c, "metadata", {}).get("section_id", "N/A"))
        sc = round(float(c.get("score", 0.0) if isinstance(c, dict) else getattr(c, "score", 0.0)), 4)
        txt = (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""))[:140].replace("\n", " ") + "..."

        chunks_diag.append({
            "rank": rank + 1,
            "page_number": pg_num,
            "section": sec,
            "score": sc,
            "text_preview": txt,
            "is_hit": is_hit,
            "page_match": page_match,
            "overlap_match": overlap_match,
            "overlap_ratio": max_overlap,
        })
    result["retrieved_chunks"] = chunks_diag
    return result


def _extract_evidence_entries(row: dict) -> list[dict[str, Any]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation slice")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Starting index in dataset")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-reranker", action="store_true", help="Bypass cross-encoder reranker")
    parser.add_argument("--output", default="results/eval/retrieval")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    console.print("[bold cyan]Slice 1 — Retrieval Evaluation[/bold cyan]")

    # Load data
    rows = load_in_corpus_financebench()
    if args.offset:
        rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]
    console.print(f"Loaded {len(rows)} in-corpus FinanceBench questions")

    # Init CRAG components
    from qdrant_client import QdrantClient
    from src.embedding.embedder import BGEEmbedder
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_expander import ParentExpander
    from src.retrieval.reranker import CrossEncoderReranker
    from src.evaluation.baseline import NaiveRAG

    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = BGEEmbedder(model_name=settings.embedding_model)
    searcher = HybridSearcher(qdrant, embedder, settings.qdrant_collection)
    expander = ParentExpander(qdrant, settings.qdrant_collection)
    use_reranker = not args.no_reranker and settings.reranker_model.lower() not in ("none", "", "null", "false")
    reranker = CrossEncoderReranker(model_name=settings.reranker_model) if use_reranker else None
    if not use_reranker:
        console.print("[yellow]Cross-encoder reranker disabled (bypassed).[/yellow]")
    naive = (
        None
        if args.no_baseline
        else NaiveRAG(qdrant, embedder, settings.qdrant_collection, top_k=10)
    )

    crag_results: list[dict] = []
    naive_results: list[dict] = []

    for i, row in enumerate(rows):
        ticker = CORPUS_TICKER_MAP[row["company"]]
        q = row.get("question", "")
        evidence_entries = _extract_evidence_entries(row)

        if not evidence_entries:
            console.print(f"[dim]  Q{i+1}: no evidence, skipping[/dim]")
            continue

        # Filing audit: check if required fiscal year is indexed
        doc_period = row.get("doc_period")
        doc_name = row.get("doc_name", "")
        if not check_filing_indexed(qdrant, settings.qdrant_collection, ticker, doc_period, doc_name):
            console.print(f"[yellow]  Q{i+1:02d} [{ticker}]: excluded: fiscal year not indexed ({doc_name})[/yellow]")
            continue

        # CRAG retrieval (real path: decomposition + retriever_node + reranker_node)
        t0 = time.perf_counter()
        res = evaluate_one(q, evidence_entries, ticker, searcher, reranker, expander)
        res["latency_s"] = round(time.perf_counter() - t0, 3)
        res["doc_period"] = row.get("doc_period", "")
        crag_results.append(res)

        # Naive baseline
        if naive:
            t0 = time.perf_counter()
            nres = evaluate_one_naive(q, evidence_entries, ticker, naive)
            nres["latency_s"] = round(time.perf_counter() - t0, 3)
            naive_results.append(nres)

        status = "✅" if res["Hit@5"] else "❌"
        rank_info = f"Gold Rank: {res['gold_evidence_rank']}" if res.get("gold_evidence_rank") else f"Candidate Rank: {res.get('gold_candidate_rank', 'Not in top-25')}"
        console.print(
            f"  {status} Q{i+1:02d} [{ticker}] Hit@5={res['Hit@5']} "
            f"MRR={res['MRR']:.2f} ({res['latency_s']}s) | {rank_info}"
        )

    # Aggregate
    crag_agg = aggregate_retrieval_metrics(crag_results)
    naive_agg = aggregate_retrieval_metrics(naive_results) if naive_results else {}

    # Display table
    table = Table(title="Retrieval Metrics (FinanceBench in-corpus, n=27)")
    table.add_column("Metric")
    table.add_column("CRAG", style="green")
    if naive_agg:
        table.add_column("Naive RAG", style="yellow")
        table.add_column("Delta", style="cyan")

    for metric in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR"):
        cval = crag_agg.get(metric, 0.0)
        row_data = [metric, f"{cval:.3f}"]
        if naive_agg:
            nval = naive_agg.get(metric, 0.0)
            delta = cval - nval
            sign = "+" if delta >= 0 else ""
            row_data += [f"{nval:.3f}", f"{sign}{delta:.3f}"]
        table.add_row(*row_data)
    console.print(table)

    # Save
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "slice": "retrieval",
        "timestamp": ts,
        "n_questions": len(crag_results),
        "crag": {"aggregate": crag_agg, "per_question": crag_results},
        "naive": {"aggregate": naive_agg, "per_question": naive_results},
    }
    out_file = out / f"retrieval_{ts}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Also write a stable "latest" file for the merger script
    (out / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"\n[dim]Results saved → {out_file}[/dim]")


if __name__ == "__main__":
    main()
