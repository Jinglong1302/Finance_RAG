"""SEC 10-K section splitter.

Splits cleaned 10-K filing HTML into SEC-mandated sections
(Item 1 through Item 15) using regex patterns on heading elements.
Each section is tagged with its type and content category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from src.ingestion.table_extractor import ExtractedTable
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FilingSection:
    """A single section extracted from a 10-K filing."""

    section_id: str  # e.g., "item7_mda"
    section_name: str  # e.g., "Management's Discussion and Analysis"
    content_type: str  # "prose" | "financial_statements" | "notes"
    html_content: str  # Raw HTML content of this section
    tables: list[ExtractedTable] = field(default_factory=list)
    start_position: int = 0
    end_position: int = 0


# SEC 10-K section patterns and metadata
# Each entry: (pattern, section_id, section_name, content_type)
_SECTION_DEFINITIONS: list[tuple[str, str, str, str]] = [
    (
        r"item\s*1[.\s:\-–—]+\s*business",
        "item1_business",
        "Business",
        "prose",
    ),
    (
        r"item\s*1a[.\s:\-–—]+\s*risk\s*factors",
        "item1a_risk_factors",
        "Risk Factors",
        "prose",
    ),
    (
        r"item\s*1b[.\s:\-–—]+\s*unresolved\s*staff",
        "item1b_unresolved",
        "Unresolved Staff Comments",
        "prose",
    ),
    (
        r"item\s*1c[.\s:\-–—]+\s*cybersecurity",
        "item1c_cybersecurity",
        "Cybersecurity",
        "prose",
    ),
    (
        r"item\s*2[.\s:\-–—]+\s*properties",
        "item2_properties",
        "Properties",
        "prose",
    ),
    (
        r"item\s*3[.\s:\-–—]+\s*legal\s*proceedings",
        "item3_legal",
        "Legal Proceedings",
        "prose",
    ),
    (
        r"item\s*4[.\s:\-–—]+\s*mine\s*safety",
        "item4_mine_safety",
        "Mine Safety Disclosures",
        "prose",
    ),
    (
        r"item\s*5[.\s:\-–—]+\s*market",
        "item5_market",
        "Market for Registrant's Common Equity",
        "prose",
    ),
    (
        r"item\s*6[.\s:\-–—]+\s*(?:selected|reserved)",
        "item6_selected_data",
        "Selected Financial Data",
        "financial_statements",
    ),
    (
        r"item\s*7[.\s:\-–—]+\s*management",
        "item7_mda",
        "Management's Discussion and Analysis",
        "prose",
    ),
    (
        r"item\s*7a[.\s:\-–—]+\s*quantitative",
        "item7a_market_risk",
        "Quantitative and Qualitative Disclosures About Market Risk",
        "prose",
    ),
    (
        r"item\s*8[.\s:\-–—]+\s*financial\s*statements",
        "item8_financial_statements",
        "Financial Statements and Supplementary Data",
        "financial_statements",
    ),
    (
        r"item\s*9[.\s:\-–—]+\s*changes?\s*in\s*and\s*disagreements?",
        "item9_changes",
        "Changes in and Disagreements with Accountants",
        "prose",
    ),
    (
        r"item\s*9a[.\s:\-–—]+\s*controls?\s*and\s*procedures?",
        "item9a_controls",
        "Controls and Procedures",
        "prose",
    ),
    (
        r"item\s*9b[.\s:\-–—]+\s*other\s*information",
        "item9b_other",
        "Other Information",
        "prose",
    ),
    (
        r"item\s*10[.\s:\-–—]+\s*directors",
        "item10_directors",
        "Directors, Executive Officers and Corporate Governance",
        "prose",
    ),
    (
        r"item\s*11[.\s:\-–—]+\s*executive\s*compensation",
        "item11_compensation",
        "Executive Compensation",
        "prose",
    ),
    (
        r"item\s*12[.\s:\-–—]+\s*security\s*ownership",
        "item12_ownership",
        "Security Ownership",
        "prose",
    ),
    (
        r"item\s*13[.\s:\-–—]+\s*certain\s*relationships",
        "item13_relationships",
        "Certain Relationships and Related Transactions",
        "prose",
    ),
    (
        r"item\s*14[.\s:\-–—]+\s*principal\s*account",
        "item14_fees",
        "Principal Accountant Fees and Services",
        "prose",
    ),
    (
        r"item\s*15[.\s:\-–—]+\s*exhibit",
        "item15_exhibits",
        "Exhibits and Financial Statement Schedules",
        "prose",
    ),
]

# Pattern for detecting individual Notes within Item 8
_NOTE_PATTERN = re.compile(
    r"note\s+(\d+)\s*[.\s:\-–—]+\s*(.+)",
    re.IGNORECASE,
)


class SectionSplitter:
    """Splits cleaned 10-K filing HTML into SEC-mandated sections.

    Uses regex patterns to detect section boundaries from heading elements.
    Assigns each section a content type (prose, financial_statements, notes)
    and associates extracted tables with their parent sections.
    """

    def __init__(self) -> None:
        # Compile regex patterns for efficiency
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), sid, sname, stype)
            for pattern, sid, sname, stype in _SECTION_DEFINITIONS
        ]

    def split(
        self,
        clean_html: str,
        tables: list[ExtractedTable] | None = None,
    ) -> list[FilingSection]:
        """Split a cleaned 10-K filing into sections.

        Args:
            clean_html: Cleaned HTML string (XBRL stripped).
            tables: Optional list of pre-extracted tables to associate
                    with their parent sections.

        Returns:
            List of FilingSection objects, ordered by position in document.
        """
        # Find all section boundary positions
        boundaries = self._find_section_boundaries(clean_html)

        if not boundaries:
            logger.warning(
                "No section boundaries found. "
                "Treating entire document as a single section."
            )
            return [
                FilingSection(
                    section_id="full_document",
                    section_name="Full Document",
                    content_type="prose",
                    html_content=clean_html,
                    start_position=0,
                    end_position=len(clean_html),
                )
            ]

        # Sort boundaries by position
        boundaries.sort(key=lambda x: x[0])

        # Extract section content between boundaries
        sections: list[FilingSection] = []

        # If there is content before the first section, capture it as cover page / header
        if boundaries and boundaries[0][0] > 0:
            first_pos = boundaries[0][0]
            header_content = clean_html[:first_pos].strip()
            if header_content:
                sections.append(
                    FilingSection(
                        section_id="cover_page",
                        section_name="Document Header and Cover Page",
                        content_type="prose",
                        html_content=clean_html[:first_pos],
                        start_position=0,
                        end_position=first_pos,
                    )
                )

        for i, (start_pos, section_id, section_name, content_type) in enumerate(
            boundaries
        ):
            # End position is the start of the next section or end of document
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][0]
            else:
                end_pos = len(clean_html)

            html_content = clean_html[start_pos:end_pos]

            section = FilingSection(
                section_id=section_id,
                section_name=section_name,
                content_type=content_type,
                html_content=html_content,
                start_position=start_pos,
                end_position=end_pos,
            )
            sections.append(section)

        # Check for Notes section within Item 8
        sections = self._split_notes_from_item8(sections)

        # Associate tables with sections
        if tables:
            self._associate_tables(sections, tables)

        logger.info(
            f"Split filing into {len(sections)} sections: "
            + ", ".join(s.section_id for s in sections)
        )
        return sections

    def _find_section_boundaries(
        self, html: str
    ) -> list[tuple[int, str, str, str]]:
        """Find section boundary positions in the HTML text.

        Searches through the cleaned HTML text for heading patterns that match
        SEC section definitions.

        Args:
            html: The cleaned HTML string.

        Returns:
            List of (position, section_id, section_name, content_type) tuples.
        """
        boundaries: list[tuple[int, str, str, str]] = []

        for pattern, section_id, section_name, content_type in self._compiled_patterns:
            matches = list(pattern.finditer(html))
            if matches:
                # If there are multiple matches (e.g. Table of Contents vs Section body),
                # pick the last occurrence which corresponds to the actual section body.
                match = matches[-1]
                boundaries.append(
                    (match.start(), section_id, section_name, content_type)
                )

        return boundaries

    def _split_notes_from_item8(
        self, sections: list[FilingSection]
    ) -> list[FilingSection]:
        """Split the Notes to Financial Statements from Item 8.

        Item 8 begins with primary financial statements followed by
        Notes to Consolidated Financial Statements. We anchor the split
        to Note 1 (or preceding Notes header) to avoid prematurely splitting
        on Item 8 index tables or footnote references.

        Args:
            sections: List of sections (may be modified in-place).

        Returns:
            Updated list of sections with notes split out.
        """
        result: list[FilingSection] = []

        for section in sections:
            if section.section_id != "item8_financial_statements":
                result.append(section)
                continue

            # Anchor on Note 1 first to avoid matching Item 8 TOC or index tables
            note1_match = re.search(
                r"\bnote\s+1[.\s:\-–—]+",
                section.html_content,
                re.IGNORECASE,
            )

            notes_start: int | None = None
            if note1_match:
                search_window = 1000
                pre_window = section.html_content[
                    max(0, note1_match.start() - search_window) : note1_match.start()
                ]
                notes_headers = list(
                    re.finditer(
                        r"notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements",
                        pre_window,
                        re.IGNORECASE,
                    )
                )
                if notes_headers:
                    notes_start = (
                        max(0, note1_match.start() - search_window)
                        + notes_headers[-1].start()
                    )
                else:
                    notes_start = note1_match.start()
            else:
                # Fallback to general pattern if Note 1 is absent
                notes_match = re.search(
                    r"notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements",
                    section.html_content,
                    re.IGNORECASE,
                )
                if notes_match:
                    notes_start = notes_match.start()

            if notes_start is not None:
                # Split: Item 8 (financial statements) and Notes
                item8_content = section.html_content[:notes_start]
                notes_content = section.html_content[notes_start:]

                original_end = section.end_position

                # Update Item 8 section
                section.html_content = item8_content
                section.end_position = section.start_position + notes_start
                result.append(section)

                # Create Notes section
                notes_section = FilingSection(
                    section_id="item8_notes",
                    section_name="Notes to Consolidated Financial Statements",
                    content_type="notes",
                    html_content=notes_content,
                    start_position=section.start_position + notes_start,
                    end_position=original_end,
                )
                result.append(notes_section)
            else:
                result.append(section)

        return result

    def _associate_tables(
        self,
        sections: list[FilingSection],
        tables: list[ExtractedTable],
    ) -> None:
        """Associate extracted tables with their parent sections.

        Uses exact position overlap (position_in_doc) when available,
        falling back to title matching and content fingerprinting.

        Args:
            sections: List of sections to associate tables with.
            tables: List of extracted tables.
        """
        # 0. Exact data-table-placeholder matching if present in section HTML
        for table in tables:
            placeholder_tag = f'data-table-placeholder="{table.table_id}"'
            for section in sections:
                if placeholder_tag in section.html_content:
                    if table not in section.tables:
                        section.tables.append(table)
                    break

        # Check which tables are already matched
        matched_table_ids = {t.table_id for s in sections for t in s.tables}
        remaining_tables = [t for t in tables if t.table_id not in matched_table_ids]

        for table in remaining_tables:
            # 1. Exact position matching if position_in_doc is known
            if table.position_in_doc > 0:
                matched = False
                for section in sections:
                    if (
                        section.start_position
                        <= table.position_in_doc
                        < section.end_position
                    ):
                        section.tables.append(table)
                        matched = True
                        break
                if matched:
                    continue

            # 2. Match by title in section HTML content
            if table.title:
                matched = False
                for section in sections:
                    if table.title in section.html_content:
                        section.tables.append(table)
                        matched = True
                        break
                if matched:
                    continue

            # 3. Match by content fingerprint
            fingerprint = table.content[:100] if table.content else ""
            if fingerprint:
                table_text = re.sub(r"[|\-\s]+", "", fingerprint)[:50]
                matched = False
                for section in sections:
                    section_text = re.sub(
                        r"<[^>]+>", "", section.html_content[:10000]
                    )
                    if table_text in section_text:
                        section.tables.append(table)
                        matched = True
                        break
                if matched:
                    continue

            # 4. Fallback: assign to the last financial_statements section
            for section in reversed(sections):
                if section.content_type == "financial_statements":
                    section.tables.append(table)
                    break
