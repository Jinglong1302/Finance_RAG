"""Prose chunker for SEC filing text sections.

Splits prose sections (MD&A, Risk Factors, Business) into ~512-token
chunks using paragraph and sentence boundaries. Applies 10% overlap
between adjacent chunks and extracts footnote references.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from src.chunking.metadata import Chunk, ChunkMetadata
from src.utils.logging import get_logger
from src.utils.tokens import count_tokens

logger = get_logger(__name__)

# Regex for detecting footnote references
_NOTE_REF_PATTERN = re.compile(
    r"(?:See|Refer\s+to|see|refer\s+to)\s+Note[s]?\s+(\d+(?:\s*(?:,|and)\s*\d+)*)",
    re.IGNORECASE,
)

# Sentence boundary pattern (handles common abbreviations)
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
)


class ProseChunker:
    """Chunks prose sections into target-sized segments.

    Splits on paragraph boundaries first. If a paragraph exceeds
    the target size, falls back to sentence-level splitting.
    Applies configurable overlap between adjacent chunks.

    Args:
        target_tokens: Target chunk size in tokens. Default: 512.
        overlap_ratio: Fraction of target_tokens to overlap between
                      adjacent chunks. Default: 0.1 (10%).
    """

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_ratio: float = 0.1,
    ) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = int(target_tokens * overlap_ratio)
        self.max_tokens = int(target_tokens * 1.5)  # Allow 50% overflow

    CHECKBOX_CHARS: set[str] = set("\u2610\u2611\u2612\u25a2\u25a0☒☐☑")

    def chunk(
        self,
        section_html: str,
        section_id: str,
        section_name: str,
        filing_meta: dict,
    ) -> list[Chunk]:
        """Chunk a prose section into target-sized segments.

        Args:
            section_html: HTML content of the section.
            section_id: Section identifier (e.g., "item7_mda").
            section_name: Human-readable section name.
            filing_meta: Filing metadata dict with keys:
                company_ticker, company_name, filing_type,
                fiscal_year, filing_date.

        Returns:
            List of Chunk objects.
        """
        paragraphs = self._extract_paragraphs(section_html)
        if not paragraphs:
            return []

        chunks = self.chunk_paragraphs(
            paragraphs=paragraphs,
            section_id=section_id,
            section_name=section_name,
            filing_meta=filing_meta,
        )

        logger.info(
            f"Chunked {section_id}: {len(paragraphs)} paragraphs → "
            f"{len(chunks)} chunks (avg {sum(c.metadata.token_count for c in chunks) // max(len(chunks), 1)} tokens)"
        )
        return chunks

    def chunk_paragraphs(
        self,
        paragraphs: list[str],
        section_id: str,
        section_name: str,
        filing_meta: dict,
        start_idx: int = 0,
    ) -> list[Chunk]:
        """Chunk a pre-extracted list of paragraphs into target-sized segments.

        Args:
            paragraphs: List of paragraph text strings.
            section_id: Section identifier (e.g., "item7_mda").
            section_name: Human-readable section name.
            filing_meta: Filing metadata dict.
            start_idx: Starting index for chunk ID numbering.

        Returns:
            List of Chunk objects.
        """
        if not paragraphs:
            return []

        raw_chunks = self._build_chunks_from_paragraphs(paragraphs)
        overlapped_chunks = self._apply_overlap(raw_chunks)

        chunks: list[Chunk] = []
        for offset, text in enumerate(overlapped_chunks):
            idx = start_idx + offset
            chunk_id = (
                f"{filing_meta['company_ticker']}_"
                f"{filing_meta['filing_type']}_"
                f"{filing_meta.get('fiscal_year', 'unknown')}_"
                f"{section_id}_prose_{idx}"
            )

            note_refs = self._detect_note_references(text)
            token_count = count_tokens(text)

            metadata = ChunkMetadata(
                company_ticker=filing_meta["company_ticker"],
                company_name=filing_meta.get("company_name", ""),
                filing_type=filing_meta["filing_type"],
                fiscal_year=filing_meta.get("fiscal_year", 0),
                filing_date=filing_meta.get("filing_date", ""),
                section=section_id,
                section_title=section_name,
                chunk_type="child",
                chunk_id=chunk_id,
                referenced_notes=note_refs,
                token_count=token_count,
                content_type="prose",
            )
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))

        return chunks

    def _extract_paragraphs(self, html: str) -> list[str]:
        """Extract paragraph text strings from section HTML.

        Strips non-content tags. Extracts paragraphs from leaf block
        elements (p, div, headings) and double newlines, preserving
        checkboxes and merging short checkbox lines with their preceding statements.
        Filters out running headers/footers and noise fragments.

        Args:
            html: Section HTML content.

        Returns:
            List of paragraph text strings.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove tables (they're handled by TableChunker) and non-content tags
        for tag in soup.find_all(["table", "script", "style", "ix:header", "ix:hidden"]):
            tag.decompose()

        # Extract leaf block elements
        raw_paras: list[str] = []
        for el in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6"]):
            # Skip if element contains child block elements (we want innermost/leaf blocks)
            if el.find(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6"]):
                continue
            t = el.get_text(separator=" ", strip=True)
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                raw_paras.append(t)

        # Fallback if no block tags found
        if not raw_paras:
            text = soup.get_text(separator="\n")
            for para in re.split(r"\n\s*\n+", text):
                para = re.sub(r"\s+", " ", para).strip()
                if para:
                    raw_paras.append(para)

        # Process paragraphs: preserve checkboxes and merge short checkbox lines
        paragraphs: list[str] = []
        for p in raw_paras:
            has_checkbox = any(c in self.CHECKBOX_CHARS for c in p)
            # If it's a short checkbox answer (e.g. "Yes ☒ No ☐" or "☒"), merge into preceding statement
            if has_checkbox and len(p) <= 30 and paragraphs:
                paragraphs[-1] = f"{paragraphs[-1]}  {p}"
                continue
            # Skip empty or very short non-checkbox fragments
            if len(p) <= 10 and not has_checkbox:
                continue
            # Filter running headers/footers (e.g., "Apple Inc. | 2023 Form 10-K | 12")
            if re.match(r"^.+?\s*\|\s*\d{4}\s+form\s+10-k\s*\|\s*\d+$", p, re.IGNORECASE):
                continue
            paragraphs.append(p)

        return paragraphs

    def _build_chunks_from_paragraphs(
        self, paragraphs: list[str]
    ) -> list[str]:
        """Build chunks by accumulating paragraphs up to target_tokens.

        If a single paragraph exceeds max_tokens, split it into
        sentence-level chunks.

        Args:
            paragraphs: List of paragraph text strings.

        Returns:
            List of chunk text strings (before overlap).
        """
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)

            # If a single paragraph is too large, split by sentences
            if para_tokens > self.max_tokens:
                # Flush current chunk first
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split paragraph into sentences and re-accumulate
                sentence_chunks = self._split_by_sentences(para)
                chunks.extend(sentence_chunks)
                continue

            # If adding this paragraph would exceed target, start new chunk
            if current_tokens + para_tokens > self.target_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            current_chunk.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split a long paragraph into sentence-level chunks.

        Args:
            text: A paragraph that exceeds max_tokens.

        Returns:
            List of sentence-level chunk strings.
        """
        sentences = _SENTENCE_BOUNDARY.split(text)
        if not sentences:
            return [text]

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_tokens = count_tokens(sentence)

            if current_tokens + sent_tokens > self.target_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            current_chunk.append(sentence)
            current_tokens += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply token overlap between adjacent chunks.

        Prepends the last overlap_tokens of chunk[i-1] to chunk[i].

        Args:
            chunks: List of chunk text strings (no overlap yet).

        Returns:
            List of chunk text strings with overlap applied.
        """
        if self.overlap_tokens <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: list[str] = [chunks[0]]  # First chunk has no predecessor

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1]
            # Get the last overlap_tokens of the previous chunk
            prev_sentences = _SENTENCE_BOUNDARY.split(prev_text)

            # Accumulate from the end until we hit overlap_tokens
            overlap_parts: list[str] = []
            overlap_token_count = 0
            for sentence in reversed(prev_sentences):
                sent_tokens = count_tokens(sentence)
                if overlap_token_count + sent_tokens > self.overlap_tokens:
                    break
                overlap_parts.insert(0, sentence.strip())
                overlap_token_count += sent_tokens

            if overlap_parts:
                overlap_text = " ".join(overlap_parts)
                overlapped.append(overlap_text + " " + chunks[i])
            else:
                overlapped.append(chunks[i])

        return overlapped

    def _detect_note_references(self, text: str) -> list[str]:
        """Detect footnote references in chunk text.

        Finds patterns like "See Note 12", "Refer to Notes 2 and 5".

        Args:
            text: Chunk text to scan.

        Returns:
            List of note references, e.g., ["Note 2", "Note 12"].
        """
        refs: list[str] = []
        for match in _NOTE_REF_PATTERN.finditer(text):
            # Extract individual note numbers
            numbers_str = match.group(1)
            numbers = re.findall(r"\d+", numbers_str)
            for num in numbers:
                note_ref = f"Note {num}"
                if note_ref not in refs:
                    refs.append(note_ref)
        return refs
