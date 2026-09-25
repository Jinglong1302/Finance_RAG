"""CLI script for end-to-end SEC filing ingestion.

Downloads, parses, chunks, embeds, and indexes SEC filings into Qdrant.

Usage:
    poetry run python scripts/ingest.py --tickers AAPL MSFT --filing-type 10-K --limit 3
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):  # reconfigure stdout to UTF-8 before Rich loads
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.chunking.chunker import ChunkingPipeline
from src.chunking.note_chunker import NoteChunker
from src.chunking.prose_chunker import ProseChunker
from src.chunking.table_chunker import TableChunker
from src.embedding.embedder import BGEEmbedder
from src.embedding.indexer import QdrantIndexer
from src.ingestion.downloader import SECDownloader
from src.ingestion.html_parser import HTMLParser
from src.ingestion.section_splitter import SectionSplitter
from src.ingestion.table_extractor import TableExtractor
from src.ingestion.xbrl_extractor import XBRLExtractor
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SEC filings into Qdrant")
    parser.add_argument(
        "--tickers", nargs="+", required=True, help="Stock ticker symbols"
    )
    parser.add_argument(
        "--filing-type", default="10-K", help="Filing type (default: 10-K)"
    )
    parser.add_argument(
        "--limit", type=int, default=3, help="Max filings per ticker (default: 3)"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Recreate Qdrant collection"
    )
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    logger.info(f"Starting ingestion for tickers: {args.tickers}")

    # === Step 1: Download filings ===
    downloader = SECDownloader(
        company=settings.sec_user_agent_company,
        email=settings.sec_user_agent_email,
    )
    filings = downloader.download_batch(
        tickers=args.tickers,
        filing_type=args.filing_type,
        limit=args.limit,
    )
    logger.info(f"Downloaded {len(filings)} filings")

    if not filings:
        logger.error("No filings downloaded. Check tickers and network.")
        return

    # === Step 2: Initialize pipeline components ===
    xbrl_extractor = XBRLExtractor()
    html_parser = HTMLParser(xbrl_extractor=xbrl_extractor)
    table_extractor = TableExtractor()
    section_splitter = SectionSplitter()

    prose_chunker = ProseChunker(
        target_tokens=settings.prose_chunk_size,
        overlap_ratio=settings.prose_chunk_overlap,
    )
    table_chunker = TableChunker(
        max_child_tokens=1024,
        max_parent_tokens=settings.max_table_chunk_tokens,
    )
    note_chunker = NoteChunker(prose_chunker=prose_chunker)
    chunking_pipeline = ChunkingPipeline(
        prose_chunker=prose_chunker,
        table_chunker=table_chunker,
        note_chunker=note_chunker,
    )

    embedder = BGEEmbedder(
        model_name=settings.embedding_model,
        use_fp16=settings.use_fp16,
    )
    indexer = QdrantIndexer(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
        embedding_dim=settings.embedding_dim,
    )

    # === Step 3: Create Qdrant collection ===
    indexer.create_collection(recreate=args.recreate)

    # === Step 4: Process each filing ===
    total_chunks = 0
    for filing in filings:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {filing.ticker} {filing.filing_type}")
        logger.info(f"{'='*60}")

        try:
            # Parse HTML
            cleaned = html_parser.parse(filing)

            # Extract tables
            tables = table_extractor.extract_tables(
                cleaned.clean_html,
                ticker=cleaned.ticker,
                fiscal_year=cleaned.fiscal_year,
            )

            # Split into sections (using annotated HTML containing table placeholders)
            annotated_html = table_extractor.annotated_html or cleaned.clean_html
            sections = section_splitter.split(annotated_html, tables)

            # Build filing metadata
            filing_meta = {
                "company_ticker": filing.ticker,
                "company_name": filing.ticker,  # Could be enriched
                "filing_type": filing.filing_type,
                "fiscal_year": filing.fiscal_year,
                "filing_date": filing.filing_date,
            }

            # Chunk all sections
            chunks = chunking_pipeline.process_filing(sections, filing_meta)

            # Separate parent and child chunks
            child_chunks = [c for c in chunks if c.metadata.chunk_type == "child"]
            parent_chunks = [c for c in chunks if c.metadata.chunk_type == "parent"]

            # Embed child chunks
            embedded = embedder.embed_chunks(child_chunks)

            # Index into Qdrant
            indexer.upsert_chunks(embedded)
            indexer.upsert_parent_chunks(parent_chunks)

            total_chunks += len(chunks)
            logger.info(
                f"[OK] {filing.ticker}: {len(chunks)} chunks indexed "
                f"({len(child_chunks)} children, {len(parent_chunks)} parents)"
            )

        except Exception as e:
            logger.error(f"[FAIL] Failed to process {filing.ticker}: {e}")
            import traceback
            traceback.print_exc()

    # === Summary ===
    info = indexer.get_collection_info()
    logger.info(f"\n{'='*60}")
    logger.info(f"Ingestion complete!")
    logger.info(f"Total chunks indexed: {total_chunks}")
    logger.info(f"Qdrant collection: {info}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
