"""Chunking pipeline orchestrator.

Routes each filing section to the appropriate chunker (prose, table, note)
and collects all chunks for a single filing.
"""

from __future__ import annotations

from src.chunking.metadata import Chunk
from src.chunking.note_chunker import NoteChunker
from src.chunking.prose_chunker import ProseChunker
from src.chunking.table_chunker import TableChunker
from src.ingestion.section_splitter import FilingSection
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChunkingPipeline:
    """Orchestrates chunking across all section types in a filing.

    Routes sections to appropriate chunkers based on content_type:
    - "prose" → ProseChunker
    - "financial_statements" → TableChunker (for tables) + ProseChunker (for inter-table prose)
    - "notes" → NoteChunker

    Args:
        prose_chunker: ProseChunker instance for text sections.
        table_chunker: TableChunker instance for financial tables.
        note_chunker: NoteChunker instance for footnotes.
    """

    def __init__(
        self,
        prose_chunker: ProseChunker | None = None,
        table_chunker: TableChunker | None = None,
        note_chunker: NoteChunker | None = None,
    ) -> None:
        self.prose_chunker = prose_chunker or ProseChunker()
        self.table_chunker = table_chunker or TableChunker()
        self.note_chunker = note_chunker or NoteChunker(
            prose_chunker=self.prose_chunker
        )

    def process_filing(
        self,
        sections: list[FilingSection],
        filing_meta: dict,
    ) -> list[Chunk]:
        """Process all sections of a filing into chunks.

        Args:
            sections: List of FilingSection objects from the section splitter.
            filing_meta: Filing metadata dict with keys:
                company_ticker, company_name, filing_type,
                fiscal_year, filing_date.

        Returns:
            List of all Chunk objects for the filing.
        """
        all_chunks: list[Chunk] = []

        for section in sections:
            section_chunks = self._process_section(section, filing_meta)
            all_chunks.extend(section_chunks)

        # Log summary
        parent_count = sum(
            1 for c in all_chunks if c.metadata.chunk_type == "parent"
        )
        child_count = sum(
            1 for c in all_chunks if c.metadata.chunk_type == "child"
        )
        total_tokens = sum(c.metadata.token_count for c in all_chunks)

        logger.info(
            f"Filing chunked: {len(all_chunks)} total chunks "
            f"({parent_count} parents, {child_count} children, "
            f"{total_tokens:,} total tokens)"
        )

        return all_chunks

    def _process_section(
        self,
        section: FilingSection,
        filing_meta: dict,
    ) -> list[Chunk]:
        """Process a single section based on its content type.

        Args:
            section: A FilingSection to chunk.
            filing_meta: Filing metadata dict.

        Returns:
            List of chunks for this section.
        """
        chunks: list[Chunk] = []

        if section.content_type == "notes":
            # Notes section → NoteChunker
            chunks.extend(
                self.note_chunker.chunk(
                    section_html=section.html_content,
                    section_id=section.section_id,
                    section_name=section.section_name,
                    filing_meta=filing_meta,
                )
            )

        elif section.content_type == "financial_statements":
            # Financial statements → tables + inter-table prose
            # Process tables first
            for table in section.tables:
                table_chunks = self.table_chunker.chunk(
                    table=table,
                    section_id=section.section_id,
                    section_name=section.section_name,
                    filing_meta=filing_meta,
                )
                chunks.extend(table_chunks)

            # Also chunk any prose that's not inside tables
            prose_chunks = self.prose_chunker.chunk(
                section_html=section.html_content,
                section_id=section.section_id,
                section_name=section.section_name,
                filing_meta=filing_meta,
            )
            # Only include prose chunks that have meaningful content
            for chunk in prose_chunks:
                if chunk.metadata.token_count >= 30:
                    chunks.append(chunk)

        elif section.content_type == "prose":
            # Prose sections → ProseChunker
            # Also process any tables within prose sections (e.g., tables in MD&A)
            for table in section.tables:
                table_chunks = self.table_chunker.chunk(
                    table=table,
                    section_id=section.section_id,
                    section_name=section.section_name,
                    filing_meta=filing_meta,
                )
                chunks.extend(table_chunks)

            # Chunk the prose content
            prose_chunks = self.prose_chunker.chunk(
                section_html=section.html_content,
                section_id=section.section_id,
                section_name=section.section_name,
                filing_meta=filing_meta,
            )
            chunks.extend(prose_chunks)

        return chunks
