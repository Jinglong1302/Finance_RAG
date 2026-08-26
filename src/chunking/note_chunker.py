"""Footnote chunker for SEC filing Notes sections.

Extracts and chunks individual footnotes from Item 8 Notes section.
Each Note becomes a labeled chunk with note_id metadata for
deterministic lookup by the agentic context expansion node.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from src.chunking.metadata import Chunk, ChunkMetadata
from src.chunking.prose_chunker import ProseChunker
from src.utils.logging import get_logger
from src.utils.tokens import count_tokens

logger = get_logger(__name__)

# Patterns for detecting Note boundaries
_NOTE_HEADING_PATTERNS = [
    re.compile(r"note\s+(\d+)\s*[.\s:\-–—]+\s*(.+)", re.IGNORECASE),
    re.compile(r"(\d+)\.\s+(.+?)(?:\s*$)", re.IGNORECASE),
]


class NoteChunker:
    """Chunks footnotes from the Notes to Financial Statements section.

    Detects Note boundaries using heading patterns, creates one chunk
    per Note, and sub-chunks Notes that exceed the token limit.

    Args:
        max_note_tokens: Maximum tokens before sub-chunking a Note.
                        Default: 2048.
        prose_chunker: Optional ProseChunker for sub-chunking large Notes.
    """

    def __init__(
        self,
        max_note_tokens: int = 2048,
        prose_chunker: ProseChunker | None = None,
    ) -> None:
        self.max_note_tokens = max_note_tokens
        self.prose_chunker = prose_chunker or ProseChunker()

    def chunk(
        self,
        section_html: str,
        section_id: str,
        section_name: str,
        filing_meta: dict,
    ) -> list[Chunk]:
        """Chunk a Notes section into individual Note chunks.

        Args:
            section_html: HTML content of the Notes section.
            section_id: Section identifier (e.g., "item8_notes").
            section_name: Section name.
            filing_meta: Filing metadata dict.

        Returns:
            List of Chunk objects (one per Note, plus sub-chunks for large Notes).
        """
        # Extract individual notes
        notes = self._extract_notes(section_html)

        if not notes:
            # If no notes detected, fall back to prose chunking
            logger.warning(
                "No individual notes detected. "
                "Falling back to prose chunking."
            )
            return self.prose_chunker.chunk(
                section_html, section_id, section_name, filing_meta
            )

        chunks: list[Chunk] = []
        for note_num, note_title, note_html in notes:
            note_chunks = self._chunk_single_note(
                note_num=note_num,
                note_title=note_title,
                note_html=note_html,
                section_id=section_id,
                filing_meta=filing_meta,
            )
            chunks.extend(note_chunks)

        logger.info(
            f"Chunked notes section: {len(notes)} notes → {len(chunks)} chunks"
        )
        return chunks

    def _extract_notes(
        self, html: str
    ) -> list[tuple[int, str, str]]:
        """Extract individual Notes from the Notes section HTML.

        Args:
            html: Notes section HTML content.

        Returns:
            List of (note_number, note_title, note_html) tuples.
        """
        soup = BeautifulSoup(html, "lxml")

        # Find heading elements that contain Note patterns
        note_elements: list[tuple[int, str, int]] = []  # (note_num, title, position)

        search_tags = soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "b", "strong", "p"]
        )

        for tag in search_tags:
            text = tag.get_text(strip=True)
            if len(text) < 3 or len(text) > 300:
                continue

            for pattern in _NOTE_HEADING_PATTERNS:
                match = pattern.search(text)
                if match:
                    note_num = int(match.group(1))
                    note_title = match.group(2).strip()
                    # Get position in HTML
                    tag_str = str(tag)
                    pos = html.find(tag_str)
                    if pos >= 0:
                        note_elements.append((note_num, note_title, pos))
                    break

        if not note_elements:
            return []

        # Sort by position
        note_elements.sort(key=lambda x: x[2])

        # Extract content between note boundaries
        notes: list[tuple[int, str, str]] = []
        for i, (note_num, note_title, start_pos) in enumerate(note_elements):
            if i + 1 < len(note_elements):
                end_pos = note_elements[i + 1][2]
            else:
                end_pos = len(html)

            note_html = html[start_pos:end_pos]
            notes.append((note_num, note_title, note_html))

        return notes

    def _chunk_single_note(
        self,
        note_num: int,
        note_title: str,
        note_html: str,
        section_id: str,
        filing_meta: dict,
    ) -> list[Chunk]:
        """Chunk a single Note into one or more chunks.

        If the Note is under max_note_tokens, it becomes a single chunk.
        Otherwise, it's sub-chunked using the prose chunker.

        Args:
            note_num: Note number (e.g., 12).
            note_title: Note title (e.g., "Segment Reporting").
            note_html: HTML content of this Note.
            section_id: Parent section ID.
            filing_meta: Filing metadata.

        Returns:
            List of chunks for this Note.
        """
        # Clean the note text
        soup = BeautifulSoup(note_html, "lxml")
        note_text = soup.get_text(separator="\n", strip=True)
        note_text = re.sub(r"\n{3,}", "\n\n", note_text)

        # Add title prefix
        full_text = f"Note {note_num}: {note_title}\n\n{note_text}"
        token_count = count_tokens(full_text)

        note_id = f"Note {note_num}"
        base_chunk_id = (
            f"{filing_meta['company_ticker']}_"
            f"{filing_meta['filing_type']}_"
            f"{filing_meta.get('fiscal_year', 'unknown')}_"
            f"note_{note_num}"
        )

        # Common metadata fields
        meta_kwargs = {
            "company_ticker": filing_meta["company_ticker"],
            "company_name": filing_meta.get("company_name", ""),
            "filing_type": filing_meta["filing_type"],
            "fiscal_year": filing_meta.get("fiscal_year", 0),
            "filing_date": filing_meta.get("filing_date", ""),
            "section": section_id,
            "section_title": f"Note {note_num}: {note_title}",
            "content_type": "note",
            "note_id": note_id,
            "note_title": note_title,
        }

        if token_count <= self.max_note_tokens:
            # Single chunk for this note
            chunk_id = f"{base_chunk_id}_full"
            metadata = ChunkMetadata(
                **meta_kwargs,
                chunk_type="child",
                chunk_id=chunk_id,
                parent_chunk_id=None,
                token_count=token_count,
            )
            return [Chunk(chunk_id=chunk_id, text=full_text, metadata=metadata)]
        else:
            # Sub-chunk using prose chunker, then annotate with note metadata
            sub_chunks = self.prose_chunker.chunk(
                note_html,
                section_id=section_id,
                section_name=f"Note {note_num}: {note_title}",
                filing_meta=filing_meta,
            )
            # Override metadata with note-specific fields
            for i, chunk in enumerate(sub_chunks):
                chunk.metadata.content_type = "note"
                chunk.metadata.note_id = note_id
                chunk.metadata.note_title = note_title
                chunk.chunk_id = f"{base_chunk_id}_part_{i}"
                chunk.metadata.chunk_id = chunk.chunk_id

            # Also create a parent chunk for the full note
            parent_id = f"{base_chunk_id}_parent"
            parent_metadata = ChunkMetadata(
                **meta_kwargs,
                chunk_type="parent",
                chunk_id=parent_id,
                parent_chunk_id=None,
                token_count=token_count,
            )
            parent = Chunk(
                chunk_id=parent_id, text=full_text, metadata=parent_metadata
            )

            # Set parent references on children
            for chunk in sub_chunks:
                chunk.metadata.parent_chunk_id = parent_id

            return [parent] + sub_chunks
