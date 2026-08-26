"""Unit tests for table extraction and conversion."""

import pytest
from src.ingestion.table_extractor import TableExtractor


class TestTableExtractor:
    """Tests for TableExtractor."""

    def setup_method(self) -> None:
        self.extractor = TableExtractor()

    def test_simple_table_to_markdown(self, sample_html_table: str) -> None:
        """Test Markdown conversion of a simple table."""
        full_html = f"<html><body>{sample_html_table}</body></html>"
        tables = self.extractor.extract_tables(full_html, ticker="AAPL", fiscal_year=2024)

        assert len(tables) >= 1
        table = tables[0]
        assert table.format_type == "markdown"
        assert "Net sales" in table.content
        assert "$394,328" in table.content
        assert "|" in table.content  # Markdown pipe syntax

    def test_complex_table_to_html(self, sample_complex_html_table: str) -> None:
        """Test HTML fallback for tables with colspan."""
        full_html = f"<html><body>{sample_complex_html_table}</body></html>"
        tables = self.extractor.extract_tables(full_html, ticker="AAPL", fiscal_year=2024)

        assert len(tables) >= 1
        table = tables[0]
        assert table.has_complex_structure is True
        assert table.format_type == "html_fallback"
        assert "Cash and cash equivalents" in table.content

    def test_header_extraction(self, sample_html_table: str) -> None:
        """Test header row extraction for child chunk injection."""
        full_html = f"<html><body>{sample_html_table}</body></html>"
        tables = self.extractor.extract_tables(full_html, ticker="AAPL", fiscal_year=2024)

        assert len(tables) >= 1
        header = tables[0].header_row
        assert "2024" in header
        assert "2023" in header

    def test_merge_multipage_tables(self) -> None:
        """Test merging of consecutive tables with identical headers."""
        html = """
        <html><body>
        <table>
            <tr><th>Metric</th><th>2024</th></tr>
            <tr><td>Revenue</td><td>$394B</td></tr>
        </table>
        <table>
            <tr><th>Metric</th><th>2024</th></tr>
            <tr><td>Net Income</td><td>$93B</td></tr>
        </table>
        </body></html>
        """
        tables = self.extractor.extract_tables(html, ticker="TEST", fiscal_year=2024)

        # Should merge into 1 logical table (identical headers)
        assert len(tables) == 1
        content = tables[0].content
        assert "Revenue" in content
        assert "Net Income" in content

    def test_skips_empty_tables(self) -> None:
        """Test that tables with minimal content are skipped."""
        html = """
        <html><body>
        <table><tr><td></td></tr></table>
        </body></html>
        """
        tables = self.extractor.extract_tables(html, ticker="TEST")
        assert len(tables) == 0

    def test_table_id_generation(self, sample_html_table: str) -> None:
        """Test that table IDs are generated correctly."""
        full_html = f"<html><body>{sample_html_table}</body></html>"
        tables = self.extractor.extract_tables(full_html, ticker="AAPL", fiscal_year=2024)

        assert len(tables) >= 1
        assert tables[0].table_id.startswith("AAPL_10K_2024_table_")

    def test_preceding_title_extraction(self) -> None:
        """Test extraction of table title from preceding heading."""
        html = """
        <html><body>
        <h3>Consolidated Statements of Operations</h3>
        <table>
            <tr><th>Item</th><th>2024</th></tr>
            <tr><td>Revenue</td><td>$394B</td></tr>
            <tr><td>Net Income</td><td>$93B</td></tr>
        </table>
        </body></html>
        """
        tables = self.extractor.extract_tables(html, ticker="AAPL", fiscal_year=2024)

        assert len(tables) >= 1
        assert "Consolidated Statements of Operations" in tables[0].title
