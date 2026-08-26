"""Unit tests for XBRL tag extraction."""

import pytest
from src.ingestion.xbrl_extractor import XBRLConcept, XBRLExtractor


class TestXBRLExtractor:
    """Tests for XBRLExtractor."""

    def test_extract_revenue_concept(self, sample_xbrl_html: str) -> None:
        """Test extraction of a whitelisted revenue concept."""
        extractor = XBRLExtractor()
        concepts = extractor.extract(sample_xbrl_html)

        assert len(concepts) >= 1
        revenue_concepts = [
            c for c in concepts
            if "Revenue" in c.concept
        ]
        assert len(revenue_concepts) >= 1

        concept = revenue_concepts[0]
        assert concept.concept == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        assert "394328" in concept.value
        assert concept.context_ref == "FD2024Q4YTD"
        assert concept.unit_ref == "usd"

    def test_extract_filters_by_whitelist(self) -> None:
        """Test that non-whitelisted concepts are skipped."""
        html = """
        <div>
            <ix:nonFraction name="custom:SomeRandomMetric"
                contextRef="ctx1" decimals="0" unitRef="usd">12345</ix:nonFraction>
        </div>
        """
        extractor = XBRLExtractor()
        concepts = extractor.extract(html)
        assert len(concepts) == 0

    def test_extract_with_custom_whitelist(self) -> None:
        """Test extraction with a custom whitelist."""
        html = """
        <div>
            <ix:nonFraction name="custom:MyMetric"
                contextRef="ctx1" decimals="0" unitRef="usd">999</ix:nonFraction>
        </div>
        """
        extractor = XBRLExtractor(whitelist={"custom:MyMetric"})
        concepts = extractor.extract(html)
        assert len(concepts) == 1
        assert concepts[0].concept == "custom:MyMetric"
        assert concepts[0].value == "999"

    def test_get_concept_names_for_chunk(self) -> None:
        """Test matching concepts to chunk text."""
        extractor = XBRLExtractor()
        concepts = [
            XBRLConcept(
                concept="us-gaap:Revenues",
                value="394328000000",
                context_ref="ctx1",
                unit_ref="usd",
                decimals="-6",
            ),
            XBRLConcept(
                concept="us-gaap:NetIncomeLoss",
                value="93736000000",
                context_ref="ctx2",
                unit_ref="usd",
                decimals="-6",
            ),
        ]

        chunk_text = "Total net revenue was 394328000000 for the period."
        found = extractor.get_concept_names_for_chunk(concepts, chunk_text)
        assert "us-gaap:Revenues" in found
        assert "us-gaap:NetIncomeLoss" not in found

    def test_empty_html(self) -> None:
        """Test extraction from HTML with no XBRL tags."""
        extractor = XBRLExtractor()
        concepts = extractor.extract("<div>No XBRL here</div>")
        assert concepts == []
