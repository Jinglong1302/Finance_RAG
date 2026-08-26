"""HTML parser for SEC filings.

Parses raw SEC filing HTML, strips XBRL wrapper tags while preserving
inner text, removes CSS/style attributes, and normalizes whitespace.
Produces a clean HTML DOM suitable for table extraction and section splitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

from src.ingestion.downloader import FilingMetadata
from src.ingestion.xbrl_extractor import XBRLConcept, XBRLExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CleanedFiling:
    """Result of parsing and cleaning a raw SEC filing."""

    ticker: str
    fiscal_year: int | None
    clean_html: str
    xbrl_concepts: list[XBRLConcept]
    metadata: FilingMetadata


# XBRL namespaces and tag prefixes to strip
_XBRL_TAG_PREFIXES = [
    "ix:",
    "xbrli:",
    "xbrldi:",
    "link:",
    "xlink:",
]

# HTML attributes to remove (non-semantic styling)
_ATTRS_TO_STRIP = {
    "style",
    "class",
    "width",
    "height",
    "bgcolor",
    "background",
    "cellpadding",
    "cellspacing",
    "border",
    "valign",
    "align",
    "color",
    "face",
    "size",
    "font",
    "nowrap",
}

# Tags to preserve (semantic HTML)
_SEMANTIC_TAGS = {
    "html",
    "head",
    "body",
    "div",
    "span",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "td",
    "th",
    "caption",
    "b",
    "strong",
    "i",
    "em",
    "u",
    "br",
    "hr",
    "a",
    "ul",
    "ol",
    "li",
    "sup",
    "sub",
}

# Attributes to keep on table elements (needed for complex table structure)
_TABLE_ATTRS_TO_KEEP = {"colspan", "rowspan"}


class HTMLParser:
    """Parser that cleans SEC filing HTML for downstream processing.

    Pipeline:
    1. Extract XBRL concepts (before stripping tags)
    2. Strip all XBRL wrapper tags (preserve inner text)
    3. Remove non-semantic attributes (style, class, width, etc.)
    4. Normalize whitespace and &nbsp; sequences
    5. Return clean HTML string

    Args:
        xbrl_extractor: XBRLExtractor instance for GAAP tag extraction.
    """

    def __init__(self, xbrl_extractor: XBRLExtractor | None = None) -> None:
        self.xbrl_extractor = xbrl_extractor or XBRLExtractor()

    def parse(self, filing: FilingMetadata) -> CleanedFiling:
        """Parse and clean a raw SEC filing HTML file.

        Args:
            filing: FilingMetadata with file_path pointing to the raw HTML.

        Returns:
            CleanedFiling with stripped HTML and extracted XBRL concepts.

        Raises:
            FileNotFoundError: If the filing HTML file doesn't exist.
            ValueError: If the HTML cannot be parsed.
        """
        if filing.file_path is None or not filing.file_path.exists():
            raise FileNotFoundError(
                f"Filing HTML not found: {filing.file_path}"
            )

        raw_html = filing.file_path.read_text(encoding="utf-8", errors="replace")
        logger.info(
            f"Parsing {filing.ticker} {filing.filing_type} "
            f"({len(raw_html):,} chars)"
        )

        # Step 1: Extract XBRL concepts BEFORE stripping tags
        xbrl_concepts = self.xbrl_extractor.extract(raw_html)

        # Step 2-4: Clean the HTML
        clean_html = self.clean_html(raw_html)

        return CleanedFiling(
            ticker=filing.ticker,
            fiscal_year=filing.fiscal_year,
            clean_html=clean_html,
            xbrl_concepts=xbrl_concepts,
            metadata=filing,
        )

    def clean_html(self, raw_html: str) -> str:
        """Clean raw SEC filing HTML by stripping XBRL and non-semantic content.

        Args:
            raw_html: The raw HTML string from EDGAR.

        Returns:
            Cleaned HTML string with only semantic markup.
        """
        soup = BeautifulSoup(raw_html, "lxml")

        # Step 2: Strip XBRL wrapper tags (preserve inner text)
        self._strip_xbrl_tags(soup)

        # Step 3: Remove non-semantic attributes
        self._strip_non_semantic_attrs(soup)

        # Step 4: Remove empty tags and normalize whitespace
        self._remove_empty_tags(soup)

        # Convert back to string and normalize whitespace
        html_str = str(soup)
        html_str = self._normalize_whitespace(html_str)

        logger.info(f"Cleaned HTML: {len(html_str):,} chars")
        return html_str

    def _strip_xbrl_tags(self, soup: BeautifulSoup) -> None:
        """Strip all XBRL wrapper tags while preserving their inner content.

        Handles tags like <ix:nonFraction>, <xbrli:context>, etc.
        The inner text/children of each tag are preserved in-place.
        """
        # Find all tags with XBRL namespace prefixes
        xbrl_tags = []
        for tag in soup.find_all(True):
            tag_name = tag.name.lower() if tag.name else ""
            if any(tag_name.startswith(prefix) for prefix in _XBRL_TAG_PREFIXES):
                xbrl_tags.append(tag)

        # Unwrap each XBRL tag (replaces tag with its children)
        for tag in xbrl_tags:
            tag.unwrap()

        # Also remove standalone XBRL elements with no visible content
        # (e.g., <xbrli:context>, <xbrli:unit> definitions)
        for tag in soup.find_all(
            re.compile(r"^(xbrli|xbrldi|link|xlink):", re.IGNORECASE)
        ):
            tag.decompose()

        # Remove xmlns attributes from root elements
        for tag in soup.find_all(True):
            attrs_to_remove = [
                attr
                for attr in tag.attrs
                if attr.startswith("xmlns") or attr.startswith("xsi:")
            ]
            for attr in attrs_to_remove:
                del tag[attr]

    def _strip_non_semantic_attrs(self, soup: BeautifulSoup) -> None:
        """Remove non-semantic HTML attributes (style, class, width, etc.).

        Preserves colspan/rowspan on table cells as they carry structural meaning.
        """
        for tag in soup.find_all(True):
            attrs_to_remove = []
            for attr in list(tag.attrs.keys()):
                attr_lower = attr.lower()
                # Keep colspan/rowspan on td/th elements
                if (
                    tag.name in ("td", "th")
                    and attr_lower in _TABLE_ATTRS_TO_KEEP
                ):
                    continue
                # Remove all styling/presentation attributes
                if attr_lower in _ATTRS_TO_STRIP:
                    attrs_to_remove.append(attr)
                # Remove data-* attributes
                elif attr_lower.startswith("data-"):
                    attrs_to_remove.append(attr)
                # Remove id attributes (often XBRL-related)
                elif attr_lower == "id":
                    attrs_to_remove.append(attr)

            for attr in attrs_to_remove:
                del tag[attr]

    def _remove_empty_tags(self, soup: BeautifulSoup) -> None:
        """Remove tags that have no text content and no meaningful children.

        Preserves <br>, <hr>, and table structure tags even if empty.
        """
        preserve_empty = {"br", "hr", "td", "th", "tr", "table", "thead", "tbody"}

        # Multiple passes to handle nested empty tags
        for _ in range(3):
            for tag in soup.find_all(True):
                if tag.name in preserve_empty:
                    continue
                if not tag.get_text(strip=True) and not tag.find_all(
                    preserve_empty
                ):
                    tag.decompose()

    def _normalize_whitespace(self, html: str) -> str:
        """Normalize whitespace and HTML entities in the cleaned HTML.

        - Replaces &nbsp; with regular spaces
        - Collapses multiple spaces into one
        - Removes excessive blank lines
        """
        # Replace &nbsp; and its variants
        html = html.replace("\xa0", " ")
        html = re.sub(r"&nbsp;", " ", html, flags=re.IGNORECASE)

        # Collapse multiple spaces (but not newlines)
        html = re.sub(r"[^\S\n]+", " ", html)

        # Remove excessive blank lines (more than 2 consecutive)
        html = re.sub(r"\n{3,}", "\n\n", html)

        # Strip leading/trailing whitespace on each line
        lines = [line.strip() for line in html.splitlines()]
        html = "\n".join(lines)

        return html.strip()
