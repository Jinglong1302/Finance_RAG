"""Table chunker for SEC filing financial tables.

Chunks tables as atomic units with parent-child hierarchy.
Small tables become a single chunk. Large tables are split into
row-group children with the header row injected as prefix.
"""

from __future__ import annotations

from src.chunking.metadata import Chunk, ChunkMetadata
from src.ingestion.table_extractor import ExtractedTable
from src.utils.logging import get_logger
from src.utils.tokens import count_tokens

logger = get_logger(__name__)


class TableChunker:
    """Chunks financial tables with parent-child hierarchy.

    Small tables (< max_child_tokens) become a single child chunk.
    Large tables are split into row-group children with the header
    row prepended. Every child references a parent chunk containing
    the full table.

    Args:
        max_child_tokens: Maximum tokens for a child chunk before splitting.
                         Default: 1024.
        max_parent_tokens: Maximum tokens before a table gets a parent chunk.
                          Default: 2048.
    """

    def __init__(
        self,
        max_child_tokens: int = 1024,
        max_parent_tokens: int = 2048,
    ) -> None:
        self.max_child_tokens = max_child_tokens
        self.max_parent_tokens = max_parent_tokens

    def chunk(
        self,
        table: ExtractedTable,
        section_id: str,
        section_name: str,
        filing_meta: dict,
    ) -> list[Chunk]:
        """Chunk a single extracted table into parent and child chunks.

        Args:
            table: ExtractedTable with content, header, and metadata.
            section_id: Parent section ID.
            section_name: Human-readable section name.
            filing_meta: Filing metadata dict.

        Returns:
            List of Chunk objects (parent + children).
        """
        # Construct the full table text with title context
        full_text = self._build_full_text(table, section_name)
        full_tokens = count_tokens(full_text)

        # Generate base chunk ID
        base_id = (
            f"{filing_meta['company_ticker']}_"
            f"{filing_meta['filing_type']}_"
            f"{filing_meta.get('fiscal_year', 'unknown')}_"
            f"{table.table_id}"
        )
        parent_id = f"{base_id}_parent"

        # Base metadata kwargs
        meta_kwargs = {
            "company_ticker": filing_meta["company_ticker"],
            "company_name": filing_meta.get("company_name", ""),
            "filing_type": filing_meta["filing_type"],
            "fiscal_year": filing_meta.get("fiscal_year", 0),
            "filing_date": filing_meta.get("filing_date", ""),
            "section": section_id,
            "section_title": section_name,
            "table_id": table.table_id,
            "table_format": table.format_type,
            "content_type": "table",
        }

        chunks: list[Chunk] = []

        if full_tokens <= self.max_parent_tokens:
            # Small table: single child chunk (no separate parent needed)
            child_id = f"{base_id}_child_0"
            metadata = ChunkMetadata(
                **meta_kwargs,
                chunk_type="child",
                chunk_id=child_id,
                parent_chunk_id=parent_id,
                token_count=full_tokens,
            )
            chunks.append(Chunk(chunk_id=child_id, text=full_text, metadata=metadata))

            # Also store as parent (same content) for consistency
            parent_metadata = ChunkMetadata(
                **meta_kwargs,
                chunk_type="parent",
                chunk_id=parent_id,
                parent_chunk_id=None,
                token_count=full_tokens,
            )
            chunks.append(
                Chunk(chunk_id=parent_id, text=full_text, metadata=parent_metadata)
            )
        else:
            # Large table: create parent + split into children
            # Parent chunk (full table, no embedding)
            parent_metadata = ChunkMetadata(
                **meta_kwargs,
                chunk_type="parent",
                chunk_id=parent_id,
                parent_chunk_id=None,
                token_count=full_tokens,
            )
            chunks.append(
                Chunk(chunk_id=parent_id, text=full_text, metadata=parent_metadata)
            )

            # Split into children with header injection
            children = self._split_table(
                table, section_name, base_id, parent_id, meta_kwargs
            )
            chunks.extend(children)

        logger.debug(
            f"Table {table.table_id}: {full_tokens} tokens → "
            f"{len(chunks)} chunks (1 parent + {len(chunks) - 1} children)"
        )
        return chunks

    def _build_full_text(
        self, table: ExtractedTable, section_name: str
    ) -> str:
        """Build the full table text with title context prefix.

        Args:
            table: ExtractedTable with content.
            section_name: Parent section name for context.

        Returns:
            Full table text with title prefix.
        """
        parts: list[str] = []

        # Add section and table title context
        if section_name:
            parts.append(f"## {section_name}")
        if table.title:
            parts.append(f"### {table.title}")

        parts.append(table.content)
        return "\n\n".join(parts)

    def _split_table(
        self,
        table: ExtractedTable,
        section_name: str,
        base_id: str,
        parent_id: str,
        meta_kwargs: dict,
    ) -> list[Chunk]:
        """Split a large table into row-group child chunks.

        Each child chunk has the header row prepended for context.

        Args:
            table: The ExtractedTable to split.
            section_name: Parent section name.
            base_id: Base chunk ID prefix.
            parent_id: Parent chunk ID for linking.
            meta_kwargs: Base metadata kwargs.

        Returns:
            List of child Chunk objects.
        """
        content = table.content
        header = table.header_row

        if table.format_type == "markdown":
            return self._split_markdown_table(
                content, header, section_name, table.title,
                base_id, parent_id, meta_kwargs,
            )
        else:
            return self._split_html_table(
                content, header, section_name, table.title,
                base_id, parent_id, meta_kwargs,
            )

    def _split_markdown_table(
        self,
        content: str,
        header: str,
        section_name: str,
        table_title: str,
        base_id: str,
        parent_id: str,
        meta_kwargs: dict,
    ) -> list[Chunk]:
        """Split a Markdown table into row-group children.

        Args:
            content: Full Markdown table string.
            header: Header row text.
            section_name: Section name for context prefix.
            table_title: Table title for context prefix.
            base_id: Base chunk ID.
            parent_id: Parent chunk ID.
            meta_kwargs: Base metadata kwargs.

        Returns:
            List of child chunks.
        """
        lines = content.strip().split("\n")
        if len(lines) < 3:
            # Too small to split meaningfully
            return []

        # First two lines are header + separator
        header_lines = lines[:2]
        data_lines = lines[2:]

        children: list[Chunk] = []
        current_lines: list[str] = []
        current_tokens = count_tokens("\n".join(header_lines))
        child_idx = 0

        for line in data_lines:
            line_tokens = count_tokens(line)

            if (
                current_tokens + line_tokens > self.max_child_tokens
                and current_lines
            ):
                # Flush current child
                child_text = self._build_child_text(
                    header_lines, current_lines, section_name, table_title
                )
                child_id = f"{base_id}_child_{child_idx}"
                metadata = ChunkMetadata(
                    **meta_kwargs,
                    chunk_type="child",
                    chunk_id=child_id,
                    parent_chunk_id=parent_id,
                    token_count=count_tokens(child_text),
                )
                children.append(
                    Chunk(chunk_id=child_id, text=child_text, metadata=metadata)
                )
                current_lines = []
                current_tokens = count_tokens("\n".join(header_lines))
                child_idx += 1

            current_lines.append(line)
            current_tokens += line_tokens

        # Flush remaining
        if current_lines:
            child_text = self._build_child_text(
                header_lines, current_lines, section_name, table_title
            )
            child_id = f"{base_id}_child_{child_idx}"
            metadata = ChunkMetadata(
                **meta_kwargs,
                chunk_type="child",
                chunk_id=child_id,
                parent_chunk_id=parent_id,
                token_count=count_tokens(child_text),
            )
            children.append(
                Chunk(chunk_id=child_id, text=child_text, metadata=metadata)
            )

        return children

    def _split_html_table(
        self,
        content: str,
        header: str,
        section_name: str,
        table_title: str,
        base_id: str,
        parent_id: str,
        meta_kwargs: dict,
    ) -> list[Chunk]:
        """Split an HTML table into row-group children.

        For HTML tables, splits on <tr> boundaries.

        Args:
            content: Full HTML table string.
            header: Header row text.
            section_name: Section name.
            table_title: Table title.
            base_id: Base chunk ID.
            parent_id: Parent chunk ID.
            meta_kwargs: Base metadata.

        Returns:
            List of child chunks.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(content, "lxml")
        table_tag = soup.find("table")
        if not table_tag:
            return []

        rows = table_tag.find_all("tr")
        if len(rows) < 2:
            return []

        # Identify header rows
        header_row_tags = []
        data_row_tags = []
        for row in rows:
            if row.find("th"):
                header_row_tags.append(str(row))
            else:
                data_row_tags.append(str(row))

        if not header_row_tags:
            header_row_tags = [str(rows[0])]
            data_row_tags = [str(r) for r in rows[1:]]

        header_html = "\n".join(header_row_tags)
        header_tokens = count_tokens(header_html)

        children: list[Chunk] = []
        current_rows: list[str] = []
        current_tokens = header_tokens
        child_idx = 0

        for row_html in data_row_tags:
            row_tokens = count_tokens(row_html)

            if (
                current_tokens + row_tokens > self.max_child_tokens
                and current_rows
            ):
                child_content = (
                    f"<table>\n{header_html}\n"
                    + "\n".join(current_rows)
                    + "\n</table>"
                )
                title_prefix = ""
                if section_name:
                    title_prefix += f"## {section_name}\n\n"
                if table_title:
                    title_prefix += f"### {table_title}\n\n"

                child_text = title_prefix + child_content
                child_id = f"{base_id}_child_{child_idx}"
                metadata = ChunkMetadata(
                    **meta_kwargs,
                    chunk_type="child",
                    chunk_id=child_id,
                    parent_chunk_id=parent_id,
                    token_count=count_tokens(child_text),
                )
                children.append(
                    Chunk(chunk_id=child_id, text=child_text, metadata=metadata)
                )
                current_rows = []
                current_tokens = header_tokens
                child_idx += 1

            current_rows.append(row_html)
            current_tokens += row_tokens

        # Flush remaining
        if current_rows:
            child_content = (
                f"<table>\n{header_html}\n"
                + "\n".join(current_rows)
                + "\n</table>"
            )
            title_prefix = ""
            if section_name:
                title_prefix += f"## {section_name}\n\n"
            if table_title:
                title_prefix += f"### {table_title}\n\n"

            child_text = title_prefix + child_content
            child_id = f"{base_id}_child_{child_idx}"
            metadata = ChunkMetadata(
                **meta_kwargs,
                chunk_type="child",
                chunk_id=child_id,
                parent_chunk_id=parent_id,
                token_count=count_tokens(child_text),
            )
            children.append(
                Chunk(chunk_id=child_id, text=child_text, metadata=metadata)
            )

        return children

    def _build_child_text(
        self,
        header_lines: list[str],
        data_lines: list[str],
        section_name: str,
        table_title: str,
    ) -> str:
        """Build child chunk text with title prefix and injected header.

        Args:
            header_lines: Markdown header + separator lines.
            data_lines: Data rows for this child chunk.
            section_name: Section name for context.
            table_title: Table title for context.

        Returns:
            Complete child chunk text.
        """
        parts: list[str] = []
        if section_name:
            parts.append(f"## {section_name}")
        if table_title:
            parts.append(f"### {table_title}")
        parts.append("\n".join(header_lines + data_lines))
        return "\n\n".join(parts)
