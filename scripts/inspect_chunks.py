"""CLI utility to inspect chunking results directly from Qdrant.

Usage:
    poetry run python scripts/inspect_chunks.py --keyword "Net income"
    poetry run python scripts/inspect_chunks.py --table table_22 --year 2023
    poetry run python scripts/inspect_chunks.py --section item8_financial_statements --limit 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient, models
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import get_settings


# Canonical section ordering for 10-K filings
SECTION_ORDER = [
    "cover_page",
    "item1_business",
    "item1a_risk_factors",
    "item1b_unresolved",
    "item1c_cybersecurity",
    "item2_properties",
    "item3_legal",
    "item4_mine_safety",
    "item5_market",
    "item6_selected_data",
    "item7_mda",
    "item7a_market_risk",
    "item8_financial_statements",
    "item8_notes",
    "item9_changes",
    "item9a_controls",
    "item9b_other",
    "item10_directors",
    "item11_compensation",
    "item12_ownership",
    "item13_relationships",
    "item14_fees",
    "item15_exhibits",
]


def _get_section_sort_key(section_id: str) -> int:
    try:
        return SECTION_ORDER.index(section_id)
    except ValueError:
        return 999


def inspect_document_flow(
    year: int = 2023,
    ticker: str = "AAPL",
    show_full: bool = False,
    section_filter: str | None = None,
    output: str | None = None,
) -> None:
    """Display the entire filing chunk-by-chunk in document order."""
    console = Console()
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    conditions = [
        models.FieldCondition(key="company_ticker", match=models.MatchValue(value=ticker)),
        models.FieldCondition(key="fiscal_year", match=models.MatchValue(value=year)),
        models.FieldCondition(key="chunk_type", match=models.MatchValue(value="child")),
    ]
    if section_filter:
        conditions.append(
            models.FieldCondition(key="section", match=models.MatchValue(value=section_filter))
        )

    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=models.Filter(must=conditions),
        limit=500,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        console.print(f"[yellow]No chunks found for {ticker} FY{year}.[/yellow]")
        return

    # Sort chunks by document section order and chunk_index (preserving exact document order)
    sorted_points = sorted(
        points,
        key=lambda p: (
            _get_section_sort_key((p.payload or {}).get("section", "")),
            (p.payload or {}).get("chunk_index", 0),
            (p.payload or {}).get("chunk_id", ""),
        ),
    )

    total_tokens = sum((p.payload or {}).get("token_count", 0) for p in sorted_points)

    console.print(
        Panel(
            f"[bold green]{ticker} 10-K (FY{year}) Document Chunk Flow[/bold green]\n"
            f"Total Child Chunks: {len(sorted_points)} | Total Tokens: {total_tokens:,}",
            border_style="green",
        )
    )

    # 1. Summary Table
    table = Table(title="Chunk-by-Chunk Inventory", border_style="dim", header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Section", style="yellow", width=26)
    table.add_column("Type", width=7)
    table.add_column("Format", width=10)
    table.add_column("Tokens", justify="right", width=7)
    table.add_column("Chunk ID", style="cyan", width=40)
    table.add_column("Preview Snippet", style="dim", width=45)

    for idx, p in enumerate(sorted_points, 1):
        payload = p.payload or {}
        text = (payload.get("text") or "").strip().replace("\n", " ")
        preview = (text[:42] + "...") if len(text) > 42 else text
        ctype = payload.get("content_type", "prose")
        cformat = payload.get("table_format") or ("markdown" if ctype == "table" else "prose")
        table.add_row(
            str(idx),
            payload.get("section", "N/A"),
            ctype,
            cformat,
            str(payload.get("token_count", 0)),
            payload.get("chunk_id", ""),
            preview,
        )

    console.print(table)

    # 2. Detailed View or Export
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            f"# {ticker} 10-K (FY{year}) Document Chunks\n",
            f"- **Total Child Chunks:** {len(sorted_points)}",
            f"- **Total Tokens:** {total_tokens:,}",
            f"- **Collection:** `{settings.qdrant_collection}`\n",
            "## Chunk Inventory\n",
            "| # | Section | Type | Format | Tokens | Chunk ID |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for idx, p in enumerate(sorted_points, 1):
            payload = p.payload or {}
            ctype = payload.get("content_type", "prose")
            cformat = payload.get("table_format") or ("markdown" if ctype == "table" else "prose")
            lines.append(
                f"| {idx} | `{payload.get('section', 'N/A')}` | {ctype} | {cformat} | {payload.get('token_count', 0)} | `{payload.get('chunk_id', '')}` |"
            )

        lines.append("\n---\n")
        lines.append("## Detailed Chunks (Document Sequence)\n")

        curr_sec = None
        for idx, p in enumerate(sorted_points, 1):
            payload = p.payload or {}
            sec = payload.get("section", "N/A")
            sec_title = payload.get("section_title", "")
            if sec != curr_sec:
                lines.append(f"\n### Section: {sec} ({sec_title})\n")
                curr_sec = sec

            cid = payload.get("chunk_id", "")
            ctype = payload.get("content_type", "prose")
            tokens = payload.get("token_count", 0)
            cformat = payload.get("table_format", "text")
            tid = payload.get("table_id") or ""

            lines.append(f"#### Chunk {idx}/{len(sorted_points)}: `{cid}`")
            meta_str = f"**Type:** `{ctype}` | **Tokens:** `{tokens}` | **Format:** `{cformat}`"
            if tid:
                meta_str += f" | **Table ID:** `{tid}`"
            lines.append(meta_str + "\n")
            lines.append("```markdown" if ctype == "table" else "")
            lines.append(payload.get("text", "").strip())
            lines.append("```\n" if ctype == "table" else "\n")
            lines.append("---\n")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[bold green]Successfully exported {len(sorted_points)} chunks to [underline]{out_path}[/underline][/bold green]")

    elif show_full:
        console.print("\n[bold cyan]=== DETAILED CHUNK-BY-CHUNK CONTENT ===[/bold cyan]\n")
        curr_sec = None
        for idx, p in enumerate(sorted_points, 1):
            payload = p.payload or {}
            sec = payload.get("section", "N/A")
            sec_title = payload.get("section_title", "")
            if sec != curr_sec:
                console.print(f"\n[bold magenta]══════ SECTION: {sec} ({sec_title}) ══════[/bold magenta]\n")
                curr_sec = sec

            console.print(
                Panel(
                    payload.get("text", ""),
                    title=f"Chunk {idx}/{len(sorted_points)}: {payload.get('chunk_id')}",
                    subtitle=f"{payload.get('content_type')} | {payload.get('token_count')} tokens | format={payload.get('table_format')}",
                    border_style="blue",
                )
            )
            console.print("-" * 80)


def inspect_chunks(
    keyword: str | None = None,
    table_id: str | None = None,
    year: int | None = None,
    section: str | None = None,
    chunk_type: str = "child",
    limit: int = 5,
) -> None:
    console = Console()
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    conditions = []
    if chunk_type:
        conditions.append(
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchValue(value=chunk_type),
            )
        )
    if year:
        conditions.append(
            models.FieldCondition(
                key="fiscal_year",
                match=models.MatchValue(value=year),
            )
        )
    if section:
        conditions.append(
            models.FieldCondition(
                key="section",
                match=models.MatchValue(value=section),
            )
        )

    scroll_filter = models.Filter(must=conditions) if conditions else None

    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=scroll_filter,
        limit=300,
        with_payload=True,
        with_vectors=False,
    )

    matches = []
    for p in points:
        payload = p.payload or {}
        text = payload.get("text", "")
        tid = payload.get("table_id") or ""

        if table_id and table_id.lower() not in tid.lower():
            continue
        if keyword and keyword.lower() not in text.lower():
            continue

        matches.append((p.id, text, payload))
        if len(matches) >= limit:
            break

    if not matches:
        console.print("[yellow]No matching chunks found in Qdrant.[/yellow]")
        return

    console.print(f"[bold green]Found {len(matches)} matching chunk(s):[/bold green]\n")

    for idx, (point_id, text, meta) in enumerate(matches, 1):
        summary_tbl = Table(show_header=False, box=None, padding=(0, 1))
        summary_tbl.add_row("[bold cyan]Chunk ID:[/bold cyan]", str(meta.get("chunk_id", point_id)))
        summary_tbl.add_row("[bold cyan]Section:[/bold cyan]", f"{meta.get('section')} ({meta.get('section_title')})")
        summary_tbl.add_row("[bold cyan]Table ID:[/bold cyan]", str(meta.get("table_id") or "N/A"))
        summary_tbl.add_row("[bold cyan]Format:[/bold cyan]", str(meta.get("table_format", "text")))
        summary_tbl.add_row("[bold cyan]Tokens:[/bold cyan]", str(meta.get("token_count", "N/A")))
        summary_tbl.add_row("[bold cyan]Year:[/bold cyan]", str(meta.get("fiscal_year", "N/A")))

        console.print(
            Panel(
                f"{text}",
                title=f"Chunk {idx}/{len(matches)}: {meta.get('chunk_id')}",
                subtitle=f"{meta.get('company_ticker')} FY{meta.get('fiscal_year')} | {meta.get('table_format')}",
                border_style="bright_blue",
            )
        )
        console.print(summary_tbl)
        console.print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect chunks stored in Qdrant.")
    parser.add_argument("--doc", action="store_true", help="Display full document chunk flow in order")
    parser.add_argument("--full", action="store_true", help="Show full chunk text for each chunk in document flow")
    parser.add_argument("--output", "-o", type=str, default=None, help="Export document chunks to a markdown file")
    parser.add_argument("--keyword", type=str, default=None, help="Filter by text keyword")
    parser.add_argument("--table", type=str, default=None, help="Filter by table ID (e.g. table_22)")
    parser.add_argument("--year", type=int, default=2023, help="Fiscal year (default: 2023)")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Company ticker (default: AAPL)")
    parser.add_argument("--section", type=str, default=None, help="Filter by section ID")
    parser.add_argument("--type", type=str, default="child", choices=["child", "parent", "all"], help="Chunk type")
    parser.add_argument("--limit", type=int, default=3, help="Max chunks to display in search mode")
    args = parser.parse_args()

    if args.doc or args.output:
        inspect_document_flow(
            year=args.year,
            ticker=args.ticker,
            show_full=args.full,
            section_filter=args.section,
            output=args.output,
        )
    else:
        inspect_chunks(
            keyword=args.keyword,
            table_id=args.table,
            year=args.year,
            section=args.section,
            chunk_type="" if args.type == "all" else args.type,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
