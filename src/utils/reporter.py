"""Analysis report generator and query execution logger for Finance RAG.

Generates persistent, structured Markdown reports and JSON traces for every
query execution (from UI or CLI) to enable offline review, auditing, and evaluation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REPORTS_DIR = Path("reports") / "runs"


def _slugify(text: str, max_len: int = 40) -> str:
    """Create a safe filename slug from query text."""
    slug = re.sub(r"[^\w\s-]", "", text.strip())
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:max_len].strip("_") or "query"


def generate_markdown_report(data: dict[str, Any]) -> str:
    """Generate a comprehensive Markdown analysis report from query result payload."""
    query = data.get("query", "")
    final_answer = data.get("final_answer", "") or data.get("generation", "No answer generated")
    confidence = (data.get("confidence") or "UNKNOWN").upper()
    cycles = data.get("cycle_count", 0)
    cost = data.get("cost_accumulated", 0.0)
    latency = data.get("latency_s") or data.get("latency_seconds", 0.0)
    guard = (data.get("hallucination_check") or "pass").upper()
    citations = data.get("citations", [])
    metrics = data.get("structured_metrics", [])
    pipeline_trace = data.get("pipeline_trace", [])
    search_results = data.get("search_results", [])
    reranked_results = data.get("reranked_results", [])
    created_at = data.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        "# 📊 Finance RAG — Analysis & Evaluation Report",
        "",
        f"**Generated:** {created_at}  ",
        f"**Query:** `{query}`  ",
        f"**Confidence:** `{confidence}` | **Hallucination Guard:** `{guard}` | **Cycles:** `{cycles}` | **Latency:** `{latency:.2f}s` | **Accumulated Cost:** `${cost:.4f}`",
        "",
        "---",
        "",
        "## 1. Verified Financial Answer",
        "",
        final_answer.strip() if final_answer else "_No answer generated._",
        "",
    ]

    # ── Section 2: Structured Financial Metrics ──────────────────────────────
    if metrics:
        lines.extend([
            "---",
            "",
            "## 2. Extracted Financial Metrics",
            "",
            "| Metric | Value | Fiscal Year | Unit | Company | Filing | Section |",
            "|:---|:---|:---:|:---:|:---:|:---:|:---|",
        ])
        for m in metrics:
            lines.append(
                f"| **{m.get('metric', '-')}** | {m.get('value', '-')} | "
                f"{m.get('fiscal_year', '-')} | {m.get('unit', '-')} | "
                f"{m.get('company', '-')} | {m.get('filing_type', '-')} | "
                f"{m.get('section', '-')} |"
            )
        lines.append("")

    # ── Section 3: Grounding Citations ────────────────────────────────────────
    lines.extend([
        "---",
        "",
        "## 3. Grounding Citations",
        "",
    ])
    if citations:
        lines.extend([
            "| Ref | Company | Filing | Fiscal Year | Section | Quote / Context |",
            "|:---:|:---:|:---:|:---:|:---|:---|",
        ])
        for cit in citations:
            ref = cit.get("index", cit.get("citation_index", "?"))
            co = cit.get("company", "-")
            filing = cit.get("filing_type", "-")
            fy = cit.get("fiscal_year", "-")
            sec = cit.get("section", "-")
            text = cit.get("context_excerpt", cit.get("text", "")).replace("\n", " ").strip()
            if len(text) > 120:
                text = text[:117] + "..."
            lines.append(f"| `[{ref}]` | {co} | {filing} | FY{fy} | {sec} | {text} |")
        lines.append("")
    else:
        lines.extend(["_No direct citations recorded for this query._", ""])

    # ── Section 4: Pipeline Execution Trace ──────────────────────────────────
    lines.extend([
        "---",
        "",
        "## 4. CRAG Pipeline Stage-by-Stage Inspector",
        "",
    ])

    if not pipeline_trace:
        lines.extend(["_No pipeline trace recorded._", ""])
    else:
        for idx, entry in enumerate(pipeline_trace, 1):
            node = entry.get("node", "Unknown")
            stage_name = node.replace("_", " ").title()
            stage_cost = entry.get("cost_usd", 0.0)
            stage_ms = entry.get("latency_ms", 0.0)

            lines.extend([
                f"### {idx}. Stage: {stage_name}",
                f"- **Latency:** `{stage_ms:.1f} ms` | **Cost:** `${stage_cost:.5f}`",
            ])

            if node == "query_decomposer":
                lines.append(f"- **Query Type:** `{entry.get('query_type', '-')}`")
                lines.append(f"- **Token Usage:** Prompt: {entry.get('prompt_tokens', 0)} | Cached: {entry.get('cached_tokens', 0)} | Completion: {entry.get('completion_tokens', 0)}")
                subs = entry.get("sub_queries", [])
                if subs:
                    lines.append("- **Generated Sub-Queries:**")
                    for s in subs:
                        lines.append(f"  - `{s}`")
                filters = entry.get("filters", {})
                if filters:
                    lines.append(f"- **Filters Detected:** `{json.dumps(filters)}`")

            elif node == "retriever":
                lines.append(f"- **Queries Issued:** {len(entry.get('queries', []))}")
                lines.append(f"- **Raw Candidates:** {entry.get('raw_results', 0)} → **After Dedup:** {entry.get('unique_results', 0)}")
                top_scores = entry.get("top_scores", [])
                if top_scores:
                    lines.append(f"- **Top RRF Scores:** {', '.join(f'{s:.4f}' for s in top_scores[:5])}")

            elif node == "reranker":
                lines.append(f"- **Input Candidates:** {entry.get('input_chunks', 0)} → **Selected Reranked:** {entry.get('reranked_chunks', 0)}")
                top_s = entry.get("top_score")
                low_s = entry.get("lowest_score")
                if top_s is not None and low_s is not None:
                    lines.append(f"- **Cross-Encoder Score Range:** `{low_s:.4f}` to `{top_s:.4f}`")

            elif node == "grader":
                lines.append(f"- **Decision:** `{entry.get('decision', '-')}`")
                lines.append(f"- **Relevant Count:** {entry.get('relevant_count', 0)} | **Filtered Out:** {entry.get('filtered_count', 0)}")
                lines.append(f"- **Tokens:** Prompt: {entry.get('prompt_tokens', 0)} | Cached: {entry.get('cached_tokens', 0)} | Completion: {entry.get('completion_tokens', 0)}")
                reason = entry.get("reasoning", "")
                if reason:
                    lines.append(f"- **Grading Reasoning:** {reason}")

            elif node == "query_rewriter":
                lines.append(f"- **Rewritten Query:** `{entry.get('rewritten_query', '-')}`")
                reason = entry.get("reasoning", "")
                if reason:
                    lines.append(f"- **Rewrite Reasoning:** {reason}")

            elif node == "generator":
                lines.append(f"- **Tokens:** Prompt: {entry.get('prompt_tokens', 0)} | Cached: {entry.get('cached_tokens', 0)} | Completion: {entry.get('completion_tokens', 0)}")
                lines.append(f"- **Extracted Citations:** {entry.get('citations_count', 0)} | **Structured Metrics:** {entry.get('metrics_count', 0)}")

            elif node == "hallucination_guard":
                status = "PASS" if entry.get("passed", True) else "WARNING"
                lines.append(f"- **Guardrail Verdict:** `{status}`")
                lines.append(f"- **Tokens:** Prompt: {entry.get('prompt_tokens', 0)} | Cached: {entry.get('cached_tokens', 0)} | Completion: {entry.get('completion_tokens', 0)}")
                claims = entry.get("hallucinated_claims", [])
                if claims:
                    lines.append("- **Flagged Claims:**")
                    for c in claims:
                        lines.append(f"  - ⚠️ {c}")

            lines.append("")

    # ── Section 5: Retrieved Evidence Chunks ─────────────────────────────────
    if reranked_results:
        lines.extend([
            "---",
            "",
            "## 5. Top Reranked Evidence Chunks",
            "",
        ])
        for idx, chunk in enumerate(reranked_results[:5], 1):
            cid = chunk.get("chunk_id", f"chunk_{idx}")
            score = chunk.get("score", 0.0)
            meta = chunk.get("metadata", {})
            company = meta.get("company_name", meta.get("company_ticker", "Unknown"))
            fy = meta.get("fiscal_year", "N/A")
            sec = meta.get("section_name", meta.get("section", "N/A"))
            text = chunk.get("text", "") or chunk.get("child_text", "")
            lines.extend([
                f"#### [{idx}] Chunk `{cid}` (Score: {score:.4f})",
                f"**Company:** {company} | **Fiscal Year:** {fy} | **Section:** {sec}",
                "```text",
                text.strip()[:600] + ("..." if len(text) > 600 else ""),
                "```",
                "",
            ])

    return "\n".join(lines)


def save_query_report(
    data: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, str]:
    """Save execution analysis report in Markdown, JSON, and append to history log.

    Returns:
        dict with report_id, md_path, and json_path.
    """
    target_dir = Path(output_dir or DEFAULT_REPORTS_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    timestamp_iso = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    query = data.get("query", "query")
    slug = _slugify(query)
    report_id = f"{timestamp_str}_{slug}"

    # Enrich data with metadata if missing
    enriched_data = dict(data)
    enriched_data["report_id"] = report_id
    enriched_data["timestamp"] = enriched_data.get("timestamp") or timestamp_iso

    # 1. Save JSON
    json_path = target_dir / f"{report_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    # 2. Save Markdown
    md_content = generate_markdown_report(enriched_data)
    md_path = target_dir / f"{report_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 3. Append to History Index
    history_path = target_dir.parent / "history.jsonl"
    history_record = {
        "report_id": report_id,
        "timestamp": timestamp_iso,
        "query": query,
        "confidence": data.get("confidence", "unknown"),
        "cost": data.get("cost_accumulated", 0.0),
        "latency": data.get("latency_s") or data.get("latency_seconds", 0.0),
        "hallucination_check": data.get("hallucination_check", "pass"),
        "citations_count": len(data.get("citations", [])),
        "md_file": str(md_path.relative_to(target_dir.parent.parent)),
        "json_file": str(json_path.relative_to(target_dir.parent.parent)),
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_record, ensure_ascii=False) + "\n")

    logger.info(f"Analysis report saved: {md_path}")

    return {
        "report_id": report_id,
        "md_path": str(md_path),
        "json_path": str(json_path),
        "filename_md": f"{report_id}.md",
    }


def list_reports(output_dir: Path | str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """List recent saved query execution reports from history.jsonl."""
    target_dir = Path(output_dir or DEFAULT_REPORTS_DIR)
    history_path = target_dir.parent / "history.jsonl"
    if not history_path.exists():
        return []

    records = []
    with open(history_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

    # Return latest first
    records.reverse()
    return records[:limit]
