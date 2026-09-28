"""Tag source page numbers onto Qdrant chunks from raw SEC filings.

Extracts page break markers (<hr>, page-break styles) from primary-document.htm,
matches each chunk text to its source position, and updates chunk payload
in Qdrant with `page_number`.

Usage:
    poetry run python scripts/tag_chunk_pages.py
    poetry run python scripts/tag_chunk_pages.py --ticker MMM
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
from bisect import bisect_right
from pathlib import Path
import re
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config.settings import get_settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def build_page_index(html_content: str) -> list[int]:
    """Find character offsets of all page break indicators in the HTML.
    
    Uses <hr> tags without double-counting nested page-break CSS properties.
    """
    hr_matches = [m.start() for m in re.finditer(r"<hr\b", html_content, re.IGNORECASE)]
    if hr_matches:
        return hr_matches
    return [m.start() for m in re.finditer(r"page-break-(?:before|after)", html_content, re.IGNORECASE)]


def find_chunk_page(
    chunk_text: str,
    raw_content: str,
    page_break_offsets: list[int],
    raw_lower: str | None = None,
) -> int | None:
    """Find the source page number of a chunk within the raw HTML."""
    from collections import Counter

    if raw_lower is None:
        raw_lower = raw_content.lower()

    # 1. Clean chunk_text: strip markdown headers (# ...) and table formatting lines
    lines = [
        l.strip()
        for l in chunk_text.splitlines()
        if not l.strip().startswith(("#", "| :", "|:")) and len(l.strip()) >= 15
    ]
    phrases: list[str] = []
    for l in lines:
        cleaned = re.sub(r"[|:_\-]+", " ", l).strip().lower()
        parts = [p.strip() for p in cleaned.split("  ") if len(p.strip()) >= 15 and not p.strip().isdigit()]
        phrases.extend(parts)

    matched_pages: list[int] = []
    for phrase in phrases:
        p_sub = phrase[:40].strip()
        if len(p_sub) < 12:
            continue
        idx = raw_lower.find(p_sub)
        if idx != -1:
            pg = bisect_right(page_break_offsets, idx)
            matched_pages.append(pg)

    if matched_pages:
        deeper = [p for p in matched_pages if p > 2]
        if deeper:
            return Counter(deeper).most_common(1)[0][0]
        return Counter(matched_pages).most_common(1)[0][0]

    # Fallback to distinct words
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", chunk_text) if w.lower() not in ("line", "item", "table", "content")]
    for i in range(0, min(10, max(1, len(words) - 2)), 2):
        phrase = " ".join(words[i : i + 3]).lower()
        idx = raw_lower.find(phrase)
        if idx != -1:
            return bisect_right(page_break_offsets, idx)

    return None


def tag_filing_chunks(
    client: QdrantClient,
    collection_name: str,
    ticker: str,
    fiscal_year: int,
    filing_type: str,
    html_path: Path,
    force: bool = False,
) -> int:
    """Tag all chunks of a specific filing with their source page number."""
    if not html_path.exists():
        logger.warning(f"File not found: {html_path}")
        return 0

    content = html_path.read_text(encoding="utf-8", errors="ignore")
    raw_lower = content.lower()
    pb_offsets = build_page_index(content)
    if not pb_offsets:
        logger.info(f"No page breaks found in {html_path.name}")
        return 0

    # Fetch all chunks for this filing from Qdrant
    flt = Filter(
        must=[
            FieldCondition(key="company_ticker", match=MatchValue(value=ticker)),
            FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year)),
            FieldCondition(key="filing_type", match=MatchValue(value=filing_type)),
        ]
    )

    offset = None
    tagged = 0
    total = 0

    while True:
        pts, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=flt,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break

        for p in pts:
            total += 1
            payload = p.payload or {}
            # If page_number is already set and not forced, skip
            if not force and payload.get("page_number") is not None:
                tagged += 1
                continue

            text = payload.get("text", "")
            page_num = find_chunk_page(text, content, pb_offsets, raw_lower=raw_lower)
            if page_num is not None:
                client.set_payload(
                    collection_name=collection_name,
                    payload={"page_number": page_num},
                    points=[p.id],
                )
                tagged += 1

        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"  {ticker} {filing_type} FY{fiscal_year}: Tagged {tagged}/{total} chunks with source page numbers.")
    return tagged


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag source page numbers onto Qdrant chunks")
    parser.add_argument("--ticker", default=None, help="Filter by ticker")
    parser.add_argument("--force", action="store_true", help="Re-compute and overwrite existing page numbers")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    # Discover all downloaded primary-document.htm files
    raw_dir = Path("data/raw/sec-edgar-filings")
    if not raw_dir.exists():
        logger.error(f"Directory {raw_dir} does not exist.")
    from src.ingestion.downloader import SECDownloader
    downloader = SECDownloader(company=settings.sec_user_agent_company, email=settings.sec_user_agent_email)

    tickers = [args.ticker.upper()] if args.ticker else ["AAPL", "MMM", "BA", "KO", "NFLX", "PFE"]
    forms = ["10-K", "10-Q"]

    total_tagged = 0
    for ticker in tickers:
        for form in forms:
            filings = downloader._discover_filings(ticker, form)
            for f in filings:
                if not f.file_path or not f.fiscal_year:
                    continue
                tagged = tag_filing_chunks(
                    client=client,
                    collection_name=settings.qdrant_collection,
                    ticker=ticker,
                    fiscal_year=f.fiscal_year,
                    filing_type=form,
                    html_path=f.file_path,
                    force=args.force,
                )
                total_tagged += tagged

    logger.info(f"Done. Successfully tagged {total_tagged} chunks with page numbers.")


if __name__ == "__main__":
    main()
