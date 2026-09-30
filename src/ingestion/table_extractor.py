"""Table extractor for SEC filings.

Detects, merges multi-page tables, and converts them to Markdown
(for simple tables) or semantic-stripped HTML (for complex tables
with colspan/rowspan).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from src.ingestion.table_normalizer import normalize_html_table_to_markdown
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedTable:
    """A single extracted and converted financial table."""

    table_id: str  # e.g., "AAPL_10K_2024_table_3"
    title: str  # From preceding heading or caption
    content: str  # Markdown or stripped HTML
    format_type: str  # "markdown" | "html_fallback"
    header_row: str  # Header text for injection into child chunks
    row_count: int
    has_complex_structure: bool  # colspan/rowspan present
    position_in_doc: int  # Character offset in source HTML (approximate)


class TableExtractor:
    """Extracts and converts financial tables from cleaned SEC filing HTML.

    Pipeline:
    1. Find all <table> elements in cleaned HTML
    2. Detect header rows (<thead> or first <tr> with <th> elements)
    3. Merge consecutive tables with identical headers (multi-page tables)
    4. Convert to Markdown (simple) or stripped HTML (complex)
    5. Return ExtractedTable objects with metadata

    Args:
        merge_threshold: Fuzzy match threshold for header similarity
                        when detecting multi-page tables (0.0 to 1.0).
    """

    def __init__(self, merge_threshold: float = 0.85) -> None:
        self.merge_threshold = merge_threshold
        self.annotated_html = ""

    def extract_tables(
        self,
        clean_html: str,
        ticker: str,
        fiscal_year: int | None = None,
    ) -> list[ExtractedTable]:
        """Extract all tables from cleaned filing HTML.

        Args:
            clean_html: Cleaned HTML string (XBRL stripped).
            ticker: Company ticker symbol.
            fiscal_year: Filing fiscal year.

        Returns:
            List of ExtractedTable objects.
        """
        soup = BeautifulSoup(clean_html, "lxml")
        raw_tables = soup.find_all("table")
        logger.info(f"Found {len(raw_tables)} raw tables in HTML")

        if not raw_tables:
            return []

        # Find character offsets for each raw table in clean_html
        raw_positions: list[int] = []
        last_pos = 0
        for t in raw_tables:
            snip = str(t)[:80]
            pos = clean_html.find(snip, last_pos)
            if pos == -1:
                pos = clean_html.find("<table", last_pos)
            raw_positions.append(pos if pos != -1 else last_pos)
            if pos != -1:
                last_pos = pos + 1

        # Step 1: Merge multi-page tables
        merged_tables, merged_positions = self._merge_multipage_tables_with_positions(
            raw_tables, raw_positions
        )
        logger.info(
            f"After merging: {len(merged_tables)} logical tables"
        )

        # Step 2: Convert each table
        extracted: list[ExtractedTable] = []
        for idx, (table_element, table_pos) in enumerate(
            zip(merged_tables, merged_positions)
        ):
            table = self._convert_table(
                table_element,
                idx=idx,
                ticker=ticker,
                fiscal_year=fiscal_year,
                position_in_doc=table_pos,
            )
            if table is not None:
                extracted.append(table)
                placeholder = soup.new_tag(
                    "div", attrs={"data-table-placeholder": table.table_id}
                )
                table_element.replace_with(placeholder)
            else:
                table_element.decompose()

        # Remove any residual tables (e.g. secondary tables merged into primary)
        for residual in soup.find_all("table"):
            residual.decompose()

        self.annotated_html = str(soup)

        logger.info(
            f"Extracted {len(extracted)} tables "
            f"({sum(1 for t in extracted if t.format_type == 'markdown')} markdown, "
            f"{sum(1 for t in extracted if t.format_type == 'html_fallback')} html)"
        )
        return extracted

    def _merge_multipage_tables(
        self, tables: list[Tag]
    ) -> list[Tag]:
        """Merge consecutive tables that share identical header rows."""
        merged, _ = self._merge_multipage_tables_with_positions(
            tables, [0] * len(tables)
        )
        return merged

    def _merge_multipage_tables_with_positions(
        self, tables: list[Tag], positions: list[int]
    ) -> tuple[list[Tag], list[int]]:
        """Merge consecutive tables that share identical header rows with positions.

        SEC filings often repeat headers across page breaks. If two
        consecutive <table> elements have matching headers (>threshold
        similarity), merge their body rows into a single logical table.

        Args:
            tables: List of <table> BeautifulSoup Tag elements.
            positions: List of start character positions for each table.

        Returns:
            Tuple of (merged table elements, starting positions).
        """
        if len(tables) <= 1:
            return list(tables), list(positions)

        merged: list[Tag] = []
        merged_positions: list[int] = []
        current = tables[0]
        curr_pos = positions[0] if positions else 0

        for next_table, next_pos in zip(tables[1:], positions[1:]):
            if self._should_merge(current, next_table):
                current = self._do_merge(current, next_table)
                logger.debug("Merged consecutive table (matching headers)")
            else:
                merged.append(current)
                merged_positions.append(curr_pos)
                current = next_table
                curr_pos = next_pos

        merged.append(current)
        merged_positions.append(curr_pos)
        return merged, merged_positions

    def _should_merge(self, table_a: Tag, table_b: Tag) -> bool:
        """Check if two tables should be merged based on header similarity.

        Args:
            table_a: First table element.
            table_b: Second table element.

        Returns:
            True if headers match above the merge threshold.
        """
        header_a = self._extract_header_text(table_a)
        header_b = self._extract_header_text(table_b)

        if not header_a or not header_b:
            return False

        # Simple similarity: ratio of matching words
        words_a = set(header_a.lower().split())
        words_b = set(header_b.lower().split())

        if not words_a or not words_b:
            return False

        intersection = words_a & words_b
        union = words_a | words_b
        similarity = len(intersection) / len(union)

        return similarity >= self.merge_threshold

    def _do_merge(self, table_a: Tag, table_b: Tag) -> Tag:
        """Merge table_b's body rows into table_a.

        Args:
            table_a: Primary table (keeps header).
            table_b: Secondary table (body rows merged in).

        Returns:
            Merged table element.
        """
        # Get body rows from table_b (skip header rows)
        header_rows_b = self._get_header_rows(table_b)
        header_row_count = len(header_rows_b)

        all_rows_b = table_b.find_all("tr")
        body_rows_b = all_rows_b[header_row_count:]

        # Append body rows to table_a
        tbody_a = table_a.find("tbody")
        if tbody_a is None:
            tbody_a = table_a

        for row in body_rows_b:
            tbody_a.append(row.__copy__())

        return table_a

    def _convert_table(
        self,
        table_element: Tag,
        idx: int,
        ticker: str,
        fiscal_year: int | None,
        position_in_doc: int = 0,
    ) -> ExtractedTable | None:
        """Convert a single table element to Markdown or stripped HTML.

        Args:
            table_element: The <table> BeautifulSoup Tag.
            idx: Table index in the document.
            ticker: Company ticker.
            fiscal_year: Filing fiscal year.
            position_in_doc: Character offset in source cleaned HTML.

        Returns:
            ExtractedTable or None if the table has no meaningful content.
        """
        # Count rows
        rows = table_element.find_all("tr")
        if len(rows) < 2:  # Need at least header + 1 data row
            return None

        # Detect complexity
        has_complex = self._needs_html_fallback(table_element)

        # Extract title from preceding heading
        title = self._find_preceding_title(table_element)

        # Convert to normalized Markdown table if possible
        norm_md = normalize_html_table_to_markdown(str(table_element))
        if norm_md:
            content = norm_md
            format_type = "markdown"
            header_text = norm_md.split("\n")[0].strip("| ").strip()
        elif has_complex:
            content = self._to_stripped_html(table_element)
            format_type = "html_fallback"
            header_text = self._extract_header_text(table_element)
        else:
            content = self._to_markdown(table_element)
            format_type = "markdown"
            header_text = self._extract_header_text(table_element)

        # Skip tables with very little content
        text_content = table_element.get_text(strip=True)
        if len(text_content) < 20:
            return None

        year_str = str(fiscal_year) if fiscal_year else "unknown"
        table_id = f"{ticker}_10K_{year_str}_table_{idx}"

        return ExtractedTable(
            table_id=table_id,
            title=title,
            content=content,
            format_type=format_type,
            header_row=header_text,
            row_count=len(rows),
            has_complex_structure=has_complex,
            position_in_doc=position_in_doc,
        )

    def _needs_html_fallback(self, table_element: Tag) -> bool:
        """Check if a table has colspan/rowspan requiring HTML fallback.

        Args:
            table_element: The <table> Tag.

        Returns:
            True if the table has complex structure.
        """
        for cell in table_element.find_all(["td", "th"]):
            for attr in ("colspan", "rowspan"):
                val = cell.get(attr)
                if val is not None:
                    try:
                        s = val[0] if isinstance(val, list) else val
                        if int(s) > 1:
                            return True
                    except (ValueError, TypeError):
                        pass
        return False

    def _to_markdown(self, table_element: Tag) -> str:
        """Convert a simple table to Markdown pipe table format.

        Args:
            table_element: A <table> Tag with no colspan/rowspan.

        Returns:
            Markdown table string.
        """
        rows = table_element.find_all("tr")
        if not rows:
            return ""

        # Build a 2D array of cell values
        grid: list[list[str]] = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            row_data = [self._clean_cell_text(cell) for cell in cells]
            grid.append(row_data)

        if not grid:
            return ""

        # Normalize column count (pad short rows)
        max_cols = max(len(row) for row in grid)
        for row in grid:
            while len(row) < max_cols:
                row.append("")

        # Calculate column widths
        col_widths = [
            max(len(row[col]) for row in grid)
            for col in range(max_cols)
        ]
        col_widths = [max(w, 3) for w in col_widths]  # Minimum width of 3

        # Format header row
        lines: list[str] = []
        header = grid[0]
        header_line = "| " + " | ".join(
            h.ljust(w) for h, w in zip(header, col_widths)
        ) + " |"
        lines.append(header_line)

        # Separator
        sep_line = "| " + " | ".join(
            "-" * w for w in col_widths
        ) + " |"
        lines.append(sep_line)

        # Data rows
        for row in grid[1:]:
            data_line = "| " + " | ".join(
                val.ljust(w) for val, w in zip(row, col_widths)
            ) + " |"
            lines.append(data_line)

        return "\n".join(lines)

    def _to_stripped_html(self, table_element: Tag) -> str:
        """Convert a complex table to semantic-stripped HTML.

        Preserves only: <table>, <tr>, <td>, <th>, <thead>, <tbody>,
        <caption>, colspan, rowspan attributes.

        Args:
            table_element: A <table> Tag with complex structure.

        Returns:
            Minimal HTML string preserving table structure.
        """
        # Deep copy to avoid modifying the original
        from copy import copy

        table_copy = table_element.__copy__()

        # For each element, keep only structural attributes
        for tag in table_copy.find_all(True) if isinstance(table_copy, Tag) else []:
            # Keep only table-related tags
            if tag.name not in (
                "table", "thead", "tbody", "tfoot",
                "tr", "td", "th", "caption", "b", "strong",
            ):
                tag.unwrap()
                continue

            # Remove all attributes except colspan/rowspan
            attrs_to_remove = [
                attr
                for attr in list(tag.attrs.keys())
                if attr not in ("colspan", "rowspan")
            ]
            for attr in attrs_to_remove:
                del tag[attr]

        result = str(table_copy)
        # Clean up extra whitespace in HTML
        result = re.sub(r"\s+", " ", result)
        result = result.replace("> <", ">\n<")
        return result.strip()

    def _extract_header_text(self, table_element: Tag) -> str:
        """Extract the header row text from a table.

        Looks for <thead> first, then falls back to first <tr> with <th> elements.

        Args:
            table_element: The <table> Tag.

        Returns:
            Header row text string (pipe-separated cell values).
        """
        header_rows = self._get_header_rows(table_element)
        if not header_rows:
            return ""

        # Use the first header row
        cells = header_rows[0].find_all(["th", "td"])
        header_values = [self._clean_cell_text(cell) for cell in cells]
        return " | ".join(header_values)

    def _get_header_rows(self, table_element: Tag) -> list[Tag]:
        """Get header rows from a table element.

        Args:
            table_element: The <table> Tag.

        Returns:
            List of header <tr> elements.
        """
        # Check for <thead>
        thead = table_element.find("thead")
        if thead:
            return thead.find_all("tr")

        # Fall back to first row if it contains <th> elements
        rows = table_element.find_all("tr")
        if rows and rows[0].find("th"):
            return [rows[0]]

        return []

    def _find_preceding_title(self, table_element: Tag) -> str:
        """Find a heading element or statement title that precedes this table.

        Uses document-order backward traversal to find table captions, headings,
        or financial statement titles.

        Args:
            table_element: The <table> Tag.

        Returns:
            Title text or empty string.
        """
        # Check for <caption> inside the table
        caption = table_element.find("caption")
        if caption:
            return self._clean_cell_text(caption)

        curr: Tag | None = table_element
        for _ in range(25):
            if curr is None:
                break
            prev = curr.find_previous(["h1", "h2", "h3", "h4", "h5", "h6", "b", "strong", "p", "div"])
            if not prev:
                break
            text = self._clean_cell_text(prev)
            if not text:
                curr = prev
                continue

            text_lower = text.lower()
            # Prioritize standard financial statement and table titles
            if any(k in text_lower for k in ["consolidated statement", "balance sheet", "statements of", "schedule", "note "]):
                if len(text) < 150:
                    return text
            # Or bold headings
            if prev.name in ("h1", "h2", "h3", "h4", "h5", "h6") or prev.find(["b", "strong"]):
                if 4 < len(text) < 120 and not text_lower.startswith("item") and "table of contents" not in text_lower:
                    return text

            curr = prev

        return ""

    def _clean_cell_text(self, element: Tag) -> str:
        """Clean and normalize text from a table cell or heading element.

        Args:
            element: A BeautifulSoup Tag.

        Returns:
            Cleaned text string.
        """
        text = element.get_text(strip=True)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
