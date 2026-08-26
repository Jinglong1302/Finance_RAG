"""Unit tests for HTML parser (XBRL stripping, attribute cleaning)."""

import pytest
from src.ingestion.html_parser import HTMLParser
from src.ingestion.xbrl_extractor import XBRLExtractor


class TestHTMLParser:
    """Tests for HTMLParser XBRL stripping and cleaning."""

    def setup_method(self) -> None:
        self.parser = HTMLParser(xbrl_extractor=XBRLExtractor())

    def test_strips_xbrl_tags_preserves_text(self) -> None:
        """Test that XBRL wrapper tags are removed but inner text is kept."""
        html = """
        <html><body>
        <p>Revenue was <ix:nonFraction name="us-gaap:Revenues"
            contextRef="ctx1" decimals="-6" unitRef="usd">394328</ix:nonFraction> million.</p>
        </body></html>
        """
        cleaned = self.parser.clean_html(html)

        # XBRL tag should be gone
        assert "ix:nonFraction" not in cleaned
        assert "ix:nonfraction" not in cleaned.lower()

        # But the text value should be preserved
        assert "394328" in cleaned
        assert "Revenue was" in cleaned

    def test_strips_style_attributes(self) -> None:
        """Test removal of style/class/width attributes."""
        html = """
        <html><body>
        <table style="width:100%; border:1px solid" class="financial-table" cellpadding="5">
            <tr>
                <td style="color:red" width="50%">Net sales</td>
                <td>$394,328</td>
            </tr>
        </table>
        </body></html>
        """
        cleaned = self.parser.clean_html(html)

        assert 'style=' not in cleaned
        assert 'class=' not in cleaned
        assert 'cellpadding=' not in cleaned

        # Content should be preserved
        assert "Net sales" in cleaned
        assert "$394,328" in cleaned

    def test_preserves_colspan_rowspan(self) -> None:
        """Test that colspan/rowspan on td/th elements are preserved."""
        html = """
        <html><body>
        <table>
            <tr><th colspan="3" style="font-weight:bold">Balance Sheet</th></tr>
            <tr><td rowspan="2">Assets</td><td>2024</td></tr>
        </table>
        </body></html>
        """
        cleaned = self.parser.clean_html(html)

        assert 'colspan="3"' in cleaned or "colspan" in cleaned
        assert 'rowspan="2"' in cleaned or "rowspan" in cleaned
        assert 'style=' not in cleaned

    def test_normalizes_whitespace(self) -> None:
        """Test whitespace normalization."""
        html = """
        <html><body>
        <p>This   has    excessive     spaces</p>
        <p>&nbsp;&nbsp;&nbsp;And non-breaking spaces</p>
        </body></html>
        """
        cleaned = self.parser.clean_html(html)

        # Multiple spaces should be collapsed
        assert "   " not in cleaned
        # &nbsp; should be replaced
        assert "&nbsp;" not in cleaned

    def test_removes_xmlns_attributes(self) -> None:
        """Test removal of xmlns declarations."""
        html = """
        <html xmlns="http://www.w3.org/1999/xhtml"
              xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
        <body><p>Content</p></body></html>
        """
        cleaned = self.parser.clean_html(html)
        assert "xmlns" not in cleaned
