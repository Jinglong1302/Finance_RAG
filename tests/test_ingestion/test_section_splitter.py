"""Unit tests for 10-K section splitting."""

import pytest
from src.ingestion.section_splitter import SectionSplitter


class TestSectionSplitter:
    """Tests for SectionSplitter."""

    def setup_method(self) -> None:
        self.splitter = SectionSplitter()

    def test_detects_item7_mda(self) -> None:
        """Test detection of Item 7 (MD&A) section."""
        html = """
        <html><body>
        <h2>Item 7. Management's Discussion and Analysis of Financial Condition
        and Results of Operations</h2>
        <p>Revenue increased 3% year over year.</p>
        <h2>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</h2>
        <p>Interest rate risk discussion.</p>
        </body></html>
        """
        sections = self.splitter.split(html)

        section_ids = [s.section_id for s in sections]
        assert "item7_mda" in section_ids

        mda = next(s for s in sections if s.section_id == "item7_mda")
        assert mda.content_type == "prose"
        assert "Revenue increased" in mda.html_content

    def test_detects_item8_financial_statements(self) -> None:
        """Test detection of Item 8 (Financial Statements)."""
        html = """
        <html><body>
        <h2>Item 8. Financial Statements and Supplementary Data</h2>
        <p>Consolidated balance sheets.</p>
        <h2>Item 9. Changes in and Disagreements with Accountants</h2>
        <p>None.</p>
        </body></html>
        """
        sections = self.splitter.split(html)

        section_ids = [s.section_id for s in sections]
        assert "item8_financial_statements" in section_ids

        item8 = next(s for s in sections if s.section_id == "item8_financial_statements")
        assert item8.content_type == "financial_statements"

    def test_splits_notes_from_item8(self) -> None:
        """Test that Notes to Financial Statements are split from Item 8."""
        html = """
        <html><body>
        <h2>Item 8. Financial Statements and Supplementary Data</h2>
        <p>Balance sheet data here.</p>
        <h3>Notes to Consolidated Financial Statements</h3>
        <p>Note 1: Summary of Significant Accounting Policies</p>
        <p>Note 2: Revenue Recognition</p>
        <h2>Item 9. Changes in and Disagreements with Accountants</h2>
        </body></html>
        """
        sections = self.splitter.split(html)

        section_ids = [s.section_id for s in sections]
        assert "item8_notes" in section_ids

        notes = next(s for s in sections if s.section_id == "item8_notes")
        assert notes.content_type == "notes"
        assert "Note 1" in notes.html_content

    def test_handles_no_sections(self) -> None:
        """Test fallback when no section headings are found."""
        html = "<html><body><p>Just some text with no headings.</p></body></html>"
        sections = self.splitter.split(html)

        assert len(sections) == 1
        assert sections[0].section_id == "full_document"

    def test_multiple_items(self) -> None:
        """Test detection of multiple different Item sections."""
        html = """
        <html><body>
        <h2>Item 1. Business</h2>
        <p>Company description.</p>
        <h2>Item 1A. Risk Factors</h2>
        <p>Risk discussion.</p>
        <h2>Item 7. Management's Discussion and Analysis</h2>
        <p>Financial review.</p>
        </body></html>
        """
        sections = self.splitter.split(html)

        section_ids = [s.section_id for s in sections]
        assert "item1_business" in section_ids
        assert "item1a_risk_factors" in section_ids
        assert "item7_mda" in section_ids
