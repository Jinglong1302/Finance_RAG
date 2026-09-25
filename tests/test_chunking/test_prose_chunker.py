"""Unit tests for prose chunker."""

import pytest
from src.chunking.prose_chunker import ProseChunker


class TestProseChunker:
    """Tests for ProseChunker."""

    def setup_method(self) -> None:
        self.chunker = ProseChunker(target_tokens=100, overlap_ratio=0.1)
        self.filing_meta = {
            "company_ticker": "AAPL",
            "company_name": "Apple Inc.",
            "filing_type": "10-K",
            "fiscal_year": 2024,
            "filing_date": "2024-11-01",
        }

    def test_chunks_prose_section(self, sample_prose_section: str) -> None:
        """Test chunking of a prose MD&A section."""
        chunks = self.chunker.chunk(
            section_html=sample_prose_section,
            section_id="item7_mda",
            section_name="Management's Discussion and Analysis",
            filing_meta=self.filing_meta,
        )

        assert len(chunks) >= 1
        # All chunks should have correct metadata
        for chunk in chunks:
            assert chunk.metadata.company_ticker == "AAPL"
            assert chunk.metadata.filing_type == "10-K"
            assert chunk.metadata.section == "item7_mda"
            assert chunk.metadata.chunk_type == "child"
            assert chunk.metadata.content_type == "prose"
            assert chunk.metadata.token_count > 0

    def test_detects_note_references(self, sample_prose_section: str) -> None:
        """Test detection of footnote references (See Note 2, Note 12)."""
        chunks = self.chunker.chunk(
            section_html=sample_prose_section,
            section_id="item7_mda",
            section_name="MD&A",
            filing_meta=self.filing_meta,
        )

        # The sample prose mentions "Note 2" and "Note 12"
        all_refs: list[str] = []
        for chunk in chunks:
            all_refs.extend(chunk.metadata.referenced_notes)

        assert "Note 2" in all_refs
        assert "Note 12" in all_refs

    def test_chunk_ids_are_unique(self) -> None:
        """Test that chunk IDs are unique within a section."""
        html = "<p>" + "</p><p>".join([f"Paragraph {i} with enough text to be meaningful." for i in range(20)]) + "</p>"
        chunks = self.chunker.chunk(
            section_html=html,
            section_id="item1_business",
            section_name="Business",
            filing_meta=self.filing_meta,
        )

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_respects_target_size(self) -> None:
        """Test that chunks are approximately target_tokens in size."""
        # Create long content that needs multiple chunks
        paragraphs = [f"<p>This is paragraph {i} with enough content to make the chunker split this into multiple chunks for our test purposes.</p>" for i in range(30)]
        html = "\n".join(paragraphs)

        chunks = self.chunker.chunk(
            section_html=html,
            section_id="item1_business",
            section_name="Business",
            filing_meta=self.filing_meta,
        )

        # Should produce more than 1 chunk
        assert len(chunks) > 1

    def test_empty_section(self) -> None:
        """Test chunking of empty or near-empty section."""
        html = "<div></div>"
        chunks = self.chunker.chunk(
            section_html=html,
            section_id="item4_mine_safety",
            section_name="Mine Safety",
            filing_meta=self.filing_meta,
        )

        assert len(chunks) == 0

    def test_chunks_div_based_paragraphs_with_header(self) -> None:
        """Test that prose chunker captures <div> paragraphs even when heading is in <p>."""
        html = """
        <p>Item 1A. Risk Factors</p>
        <div><span>First major risk factor about global macroeconomic conditions affecting consumer spending and demand.</span></div>
        <div><span>Second major risk factor about supply chain disruptions affecting manufacturing components and logistics.</span></div>
        """
        chunks = self.chunker.chunk(
            section_html=html,
            section_id="item1a_risk_factors",
            section_name="Risk Factors",
            filing_meta=self.filing_meta,
        )

        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "Item 1A. Risk Factors" in all_text
        assert "macroeconomic conditions" in all_text
        assert "supply chain disruptions" in all_text

    def test_filters_running_headers(self) -> None:
        """Test that running headers like 'Apple Inc. | 2023 Form 10-K | 12' are filtered out."""
        html = """
        <p>Item 1A. Risk Factors</p>
        <div>Apple Inc. | 2023 Form 10-K | 12</div>
        <div>The Company's business could be adversely affected by international trade tensions.</div>
        """
        chunks = self.chunker.chunk(
            section_html=html,
            section_id="item1a_risk_factors",
            section_name="Risk Factors",
            filing_meta=self.filing_meta,
        )

        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "2023 Form 10-K | 12" not in all_text
        assert "international trade tensions" in all_text

    def test_checkbox_statements_preserved(self) -> None:
        """Test that checkbox statements and answers (e.g. Yes ☒ No ☐) are preserved."""
        html = """
        <p>Indicate by check mark if the Registrant is a well-known seasoned issuer, as defined in Rule 405 of the Securities Act.</p>
        <div><span>Yes </span><span>&#9746;</span><span> No </span><span>&#9744;</span></div>
        <p>Indicate by check mark whether the Registrant is a shell company (as defined in Rule 12b-2 of the Act).</p>
        <div><span>Yes </span><span>&#9744;</span><span> No </span><span>&#9746;</span></div>
        """
        chunks = self.chunker.chunk(
            section_html=html,
            section_id="cover_page",
            section_name="Document Header and Cover Page",
            filing_meta=self.filing_meta,
        )

        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "well-known seasoned issuer" in all_text
        assert "\u2612" in all_text or "☒" in all_text
        assert "\u2610" in all_text or "☐" in all_text
        assert "shell company" in all_text
