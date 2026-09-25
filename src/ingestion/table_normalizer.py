"""Table normalizer for converting raw SEC HTML tables into clean Markdown tables.

SEC EDGAR tables frequently contain dozens of visual spacer columns (empty <td>),
multi-row headers, split currency symbols ($ in separate <td>), and colspan/rowspan tags.
This module parses the HTML grid, removes visual noise, merges split symbols,
and outputs standardized, token-efficient Markdown tables.
"""

from __future__ import annotations

import re
from typing import Any
from bs4 import BeautifulSoup, Tag


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip non-breaking spaces."""
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_span(val: object) -> int:
    """Safely parse integer span attribute from BeautifulSoup tag."""
    if val is None:
        return 1
    if isinstance(val, list):
        val = val[0] if val else 1
    try:
        return int(str(val))
    except (ValueError, TypeError):
        return 1


def html_table_to_raw_grid(table_element: Tag) -> list[list[str]]:
    """Expand HTML table into a full 2D grid matrix of cell texts.
    For cells with colspan > 1, the text is placed in the first column
    and subsequent columns in the span are assigned empty string "".
    """
    rows = table_element.find_all("tr")
    if not rows:
        return []

    grid: list[list[str]] = []
    # Track occupied cells due to rowspan: (col_idx, remaining_rows, cell_text)
    active_rowspans: list[dict[str, Any]] = []

    for row in rows:
        grid_row: list[str] = []
        col_idx = 0

        # Decrement and place active rowspans from previous rows
        new_active_rowspans = []
        for span in active_rowspans:
            while col_idx < span["col"]:
                grid_row.append("")
                col_idx += 1
            grid_row.append(span["text"])
            col_idx += 1
            span["rows_left"] -= 1
            if span["rows_left"] > 0:
                new_active_rowspans.append(span)
        active_rowspans = new_active_rowspans

        cells = row.find_all(["td", "th"])
        if not cells and not grid_row:
            continue

        for cell in cells:
            while col_idx < len(grid_row) and grid_row[col_idx] != "":
                col_idx += 1

            text = _clean_text(cell.get_text(separator=" ", strip=True))

            colspan = _parse_span(cell.get("colspan"))
            rowspan = _parse_span(cell.get("rowspan"))

            if rowspan > 1:
                active_rowspans.append({
                    "col": col_idx,
                    "rows_left": rowspan - 1,
                    "text": text,
                })

            grid_row.append(text)
            col_idx += 1
            for _ in range(colspan - 1):
                grid_row.append("")
                col_idx += 1

        grid.append(grid_row)

    max_cols = max((len(r) for r in grid), default=0)
    for r in grid:
        if len(r) < max_cols:
            r.extend([""] * (max_cols - len(r)))

    return grid


def find_date_header_row(grid: list[list[str]]) -> tuple[int, list[int]]:
    """Locate the primary date/column header row and the start column indices of each date.
    
    Returns:
        (header_row_index, list_of_column_start_indices)
    """
    year_re = re.compile(r"\b(19\d\d|20\d\d)\b")
    date_word_re = re.compile(r"(september|december|march|june|years?\s+ended|months?\s+ended|as\s+of)", re.IGNORECASE)

    best_row_idx = -1
    best_starts: list[int] = []

    for r_idx in range(min(6, len(grid))):
        row = grid[r_idx]
        starts = [c_idx for c_idx, val in enumerate(row) if val and (year_re.search(val) or date_word_re.search(val))]
        # If this row contains multiple date/year headers (e.g. 2023, 2022, 2021)
        if len(starts) >= 2 and len(starts) > len(best_starts):
            best_row_idx = r_idx
            best_starts = starts

    # If no multi-date row found, fallback to first row with any year
    if best_row_idx == -1:
        for r_idx in range(min(5, len(grid))):
            row = grid[r_idx]
            starts = [c_idx for c_idx, val in enumerate(row) if val and year_re.search(val)]
            if starts:
                best_row_idx = r_idx
                best_starts = starts
                break

    return best_row_idx, best_starts


def collapse_slots_to_markdown_rows(grid: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Segment 2D grid columns into semantic slots and collapse into a clean table."""
    if not grid or not grid[0]:
        return [], []

    num_cols = len(grid[0])
    header_row_idx, date_starts = find_date_header_row(grid)

    # If we detected distinct date column start positions
    if header_row_idx != -1 and date_starts:
        # Determine slot boundary intervals:
        # Slot 0 is [0 ... first_date_start - 1] (The Line Item column)
        slots: list[tuple[int, int]] = []
        first_date_col = date_starts[0]
        if first_date_col > 0:
            slots.append((0, first_date_col))

        # Each date gets a slot up to the next date start
        for i in range(len(date_starts)):
            start = date_starts[i]
            end = date_starts[i + 1] if i + 1 < len(date_starts) else num_cols
            slots.append((start, end))

        # Extract headers from each slot
        header_labels: list[str] = []
        for slot_idx, (start, end) in enumerate(slots):
            if slot_idx == 0 and first_date_col > 0:
                header_labels.append("Line Item")
                continue

            # Combine text across header rows in this slot range
            slot_header_parts = []
            for r in range(header_row_idx + 1):
                val = " ".join(grid[r][c] for c in range(start, end) if grid[r][c]).strip()
                if val and val not in slot_header_parts:
                    slot_header_parts.append(val)

            label = " ".join(slot_header_parts).strip()
            # Normalize year, e.g. "September 30, 2023" -> extract year if possible or keep label
            clean_label = re.sub(r"Years\s+ended\s+", "", label, flags=re.IGNORECASE).strip()
            header_labels.append(clean_label or f"Col {slot_idx}")

        # Extract data rows by collapsing values inside each slot
        data_rows: list[list[str]] = []
        for r_idx in range(header_row_idx + 1, len(grid)):
            row = grid[r_idx]
            collapsed_row = []

            # Line item cell
            if first_date_col > 0:
                line_item = " ".join(row[c] for c in range(slots[0][0], slots[0][1]) if row[c]).strip()
                collapsed_row.append(line_item)

            # Data column cells
            start_slot = 1 if first_date_col > 0 else 0
            for start, end in slots[start_slot:]:
                # Gather all tokens in slot (e.g. '$' + '96,995' -> '$96,995')
                tokens = [row[c] for c in range(start, end) if row[c].strip()]
                # Merge currency symbols
                if len(tokens) >= 2 and tokens[0] in ("$", "%", "€", "£"):
                    cell_val = f"{tokens[0]}{tokens[1]}"
                    if len(tokens) > 2:
                        cell_val += " " + " ".join(tokens[2:])
                else:
                    cell_val = "".join(tokens) if len(tokens) == 2 and tokens[0] in ("$", "(") else " ".join(tokens)
                collapsed_row.append(cell_val.strip())

            # Only include if at least one cell has content
            if any(cell for cell in collapsed_row):
                data_rows.append(collapsed_row)

        return header_labels, data_rows

    # Generic Fallback: Prune empty columns and use row 0 as header
    cols_with_data = [c for c in range(num_cols) if any(grid[r][c].strip() for r in range(len(grid)))]
    if not cols_with_data:
        return [], []

    pruned = [[grid[r][c] for c in cols_with_data] for r in range(len(grid))]
    headers = [pruned[0][c] or f"Col {c+1}" for c in range(len(pruned[0]))]
    if headers and not headers[0].strip():
        headers[0] = "Line Item"
    data = [[pruned[r][c] for c in range(len(pruned[r]))] for r in range(1, len(pruned)) if any(pruned[r])]
    return headers, data


def normalize_html_table_to_markdown(html_content: str) -> str | None:
    """End-to-end normalization of an SEC HTML <table> into clean GitHub-Flavored Markdown.

    Returns:
        Clean Markdown table string or None if table has no meaningful data.
    """
    soup = BeautifulSoup(html_content, "lxml")
    table_tag = soup.find("table")
    if not table_tag:
        return None

    # Step 1: 2D Grid with colspan/rowspan resolution
    grid = html_table_to_raw_grid(table_tag)
    if not grid or len(grid) < 2:
        return None

    # Step 2: Slot segmentation and collapsing
    headers, data_rows = collapse_slots_to_markdown_rows(grid)
    if not headers or not data_rows:
        return None

    # Format to Markdown table
    header_line = "| " + " | ".join(h.replace("|", "/") for h in headers) + " |"
    separator_line = "| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |"

    lines = [header_line, separator_line]
    for row in data_rows:
        # Clean cell text for markdown pipes
        clean_row = [cell.replace("|", "/").strip() for cell in row]
        while len(clean_row) < len(headers):
            clean_row.append("")
        lines.append("| " + " | ".join(clean_row[:len(headers)]) + " |")

    return "\n".join(lines)
