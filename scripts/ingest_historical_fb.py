"""Ingest the 12 exact historical SEC filings required by FinanceBench.

Covers:
- 3M (MMM): 2018 10-K, 2022 10-K, 2023 Q2 10-Q
- Boeing (BA): 2018 10-K, 2022 10-K
- Coca-Cola (KO): 2017 10-K, 2021 10-K, 2022 10-K
- Netflix (NFLX): 2015 10-K, 2017 10-K
- Pfizer (PFE): 2021 10-K, 2023 Q2 10-Q

Indexes child chunks (dense + sparse) and parent chunks into Qdrant ('sec_filings')
without recreating the collection (preserving existing AAPL filings).
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
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
from src.ingestion.downloader import FilingMetadata, SECDownloader
from src.ingestion.html_parser import HTMLParser
from src.ingestion.section_splitter import SectionSplitter
from src.ingestion.table_extractor import TableExtractor
from src.ingestion.xbrl_extractor import XBRLExtractor
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

TARGET_FILINGS = [
    ("MMM", "10-K", 2018),
    ("MMM", "10-K", 2022),
    ("MMM", "10-Q", 2023),
    ("BA", "10-K", 2018),
    ("BA", "10-K", 2022),
    ("KO", "10-K", 2017),
    ("KO", "10-K", 2021),
    ("KO", "10-K", 2022),
    ("NFLX", "10-K", 2015),
    ("NFLX", "10-K", 2017),
    ("PFE", "10-K", 2021),
    ("PFE", "10-Q", 2023),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest historical FinanceBench filings")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--ticker", default=None, help="Optionally ingest only one ticker")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    downloader = SECDownloader(
        company=settings.sec_user_agent_company,
        email=settings.sec_user_agent_email,
    )

    targets = TARGET_FILINGS
    if args.ticker:
        targets = [t for t in targets if t[0].upper() == args.ticker.upper()]

    logger.info(f"Targeting {len(targets)} historical filings for FinanceBench...")

    # Discover the exact downloaded filing paths
    filings_to_process: list[FilingMetadata] = []
    for ticker, form, target_year in targets:
        all_f = downloader._discover_filings(ticker, form)
        matched = [f for f in all_f if f.fiscal_year == target_year]
        if matched:
            filings_to_process.append(matched[0])
            logger.info(f"  Matched {ticker} {form} FY{target_year}: {matched[0].file_path}")
        else:
            logger.error(f"  MISSING filing: {ticker} {form} FY{target_year}!")

    if not filings_to_process:
        logger.error("No filings found to process.")
        return

    logger.info(f"Ready to ingest {len(filings_to_process)} filings.")

    # Initialize ingestion components
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

    # Ensure collection exists without recreating
    indexer.create_collection(recreate=False)

    total_chunks = 0
    for idx, filing in enumerate(filings_to_process, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx}/{len(filings_to_process)}] Processing: {filing.ticker} {filing.filing_type} FY{filing.fiscal_year}")
        logger.info(f"{'='*60}")

        try:
            cleaned = html_parser.parse(filing)

            tables = table_extractor.extract_tables(
                cleaned.clean_html,
                ticker=cleaned.ticker,
                fiscal_year=cleaned.fiscal_year,
            )

            annotated_html = table_extractor.annotated_html or cleaned.clean_html
            sections = section_splitter.split(annotated_html, tables)

            filing_meta = {
                "company_ticker": filing.ticker,
                "company_name": filing.ticker,
                "filing_type": filing.filing_type,
                "fiscal_year": filing.fiscal_year,
                "filing_date": filing.filing_date,
            }

            chunks = chunking_pipeline.process_filing(sections, filing_meta)
            child_chunks = [c for c in chunks if c.metadata.chunk_type == "child"]
            parent_chunks = [c for c in chunks if c.metadata.chunk_type == "parent"]

            logger.info(f"Embedding {len(child_chunks)} child chunks with BGE-M3...")
            embedded = embedder.embed_chunks(child_chunks)

            logger.info(f"Upserting to Qdrant...")
            indexer.upsert_chunks(embedded)
            indexer.upsert_parent_chunks(parent_chunks)

            # Tag chunks with source page numbers
            try:
                from scripts.tag_chunk_pages import tag_filing_chunks
                if filing.file_path:
                    tag_filing_chunks(
                        client=indexer.client,
                        collection_name=indexer.collection_name,
                        ticker=filing.ticker,
                        fiscal_year=filing.fiscal_year,
                        filing_type=filing.filing_type,
                        html_path=filing.file_path,
                    )
            except Exception as e_tag:
                logger.warning(f"Could not tag page numbers for {filing.ticker}: {e_tag}")

            total_chunks += len(chunks)
            logger.info(f"[OK] {filing.ticker} FY{filing.fiscal_year}: {len(chunks)} chunks indexed.")

        except Exception as e:
            logger.error(f"[FAIL] Error processing {filing.ticker}: {e}")
            import traceback
            traceback.print_exc()

    info = indexer.get_collection_info()
    logger.info(f"\n{'='*60}")
    logger.info(f"Historical FB Ingestion Complete!")
    logger.info(f"Total new chunks indexed: {total_chunks}")
    logger.info(f"Qdrant collection info: {info}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
