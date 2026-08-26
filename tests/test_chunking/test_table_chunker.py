"""Unit tests for table chunker."""

import pytest
from src.chunking.table_chunker import TableChunker
from src.ingestion.table_extractor import ExtractedTable


class TestTableChunker:
    """Tests for TableChunker."""

    def setup_method(self) -> None:
        self.chunker = TableChunker(max_child_tokens=200, max_parent_tokens=500)
        self.filing_meta = {
            "company_ticker": "AAPL",
            "company_name": "Apple Inc.",
            "filing_type": "10-K",
            "fiscal_year": 2024,
            "filing_date": "2024-11-01",
        }

    def test_small_table_single_chunk(self) -> None:
        """Test that a small table produces a single child + parent chunk."""
        md_content = (
            "| Metric | 2024 | 2023 |\n"
            "| --- | --- | --- |\n"
            "| Revenue | $394B | $383B |\n"
            "| Net Income | $93B | $97B |"
        )
        table = ExtractedTable(
            table_id="AAPL_10K_2024_table_0",
            title="Income Statement",
            content=md_content,
            format_type="markdown",
            header_row="Metric | 2024 | 2023",
            row_count=3,
            has_complex_structure=False,
            position_in_doc=0,
        )

        chunks = self.chunker.chunk(
            table=table,
            section_id="item8_financial_statements",
            section_name="Financial Statements",
            filing_meta=self.filing_meta,
        )

        # Should produce parent + child
        assert len(chunks) == 2
        parents = [c for c in chunks if c.metadata.chunk_type == "parent"]
        children = [c for c in chunks if c.metadata.chunk_type == "child"]
        assert len(parents) == 1
        assert len(children) == 1

        # Child should reference parent
        assert children[0].metadata.parent_chunk_id == parents[0].chunk_id

    def test_large_table_splits_into_children(self) -> None:
        """Test that a large table is split into multiple children."""
        # Create a table large enough to exceed max_parent_tokens
        rows = "\n".join(
            f"| Line item {i} | ${i * 1000} | ${i * 900} |"
            for i in range(1, 50)
        )
        md_content = (
            "| Item | 2024 | 2023 |\n"
            "| --- | --- | --- |\n"
            + rows
        )
        table = ExtractedTable(
            table_id="AAPL_10K_2024_table_1",
            title="Detailed Breakdown",
            content=md_content,
            format_type="markdown",
            header_row="Item | 2024 | 2023",
            row_count=50,
            has_complex_structure=False,
            position_in_doc=0,
        )

        chunks = self.chunker.chunk(
            table=table,
            section_id="item8_financial_statements",
            section_name="Financial Statements",
            filing_meta=self.filing_meta,
        )

        parents = [c for c in chunks if c.metadata.chunk_type == "parent"]
        children = [c for c in chunks if c.metadata.chunk_type == "child"]

        assert len(parents) == 1
        assert len(children) > 1  # Should be split

        # All children should reference the parent
        for child in children:
            assert child.metadata.parent_chunk_id == parents[0].chunk_id

    def test_table_metadata(self) -> None:
        """Test that table chunks carry correct metadata."""
        table = ExtractedTable(
            table_id="AAPL_10K_2024_table_5",
            title="Balance Sheet",
            content="| Assets | $100B |\n| --- | --- |\n| Cash | $29B |",
            format_type="markdown",
            header_row="Assets | Amount",
            row_count=2,
            has_complex_structure=False,
            position_in_doc=0,
        )

        chunks = self.chunker.chunk(
            table=table,
            section_id="item8_financial_statements",
            section_name="Financial Statements",
            filing_meta=self.filing_meta,
        )

        for chunk in chunks:
            assert chunk.metadata.content_type == "table"
            assert chunk.metadata.table_id == "AAPL_10K_2024_table_5"
            assert chunk.metadata.table_format == "markdown"
            assert chunk.metadata.company_ticker == "AAPL"
