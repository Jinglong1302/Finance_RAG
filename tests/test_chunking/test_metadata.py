"""Unit tests for chunk metadata schema."""

import pytest
from src.chunking.metadata import Chunk, ChunkMetadata


class TestChunkMetadata:
    """Tests for Chunk and ChunkMetadata models."""

    def test_chunk_creation(self) -> None:
        """Test basic Chunk creation with all required fields."""
        metadata = ChunkMetadata(
            company_ticker="AAPL",
            filing_type="10-K",
            fiscal_year=2024,
            section="item7_mda",
            chunk_type="child",
            chunk_id="AAPL_10K_2024_item7_prose_0",
        )
        chunk = Chunk(
            chunk_id="AAPL_10K_2024_item7_prose_0",
            text="Revenue increased 3%.",
            metadata=metadata,
        )

        assert chunk.chunk_id == "AAPL_10K_2024_item7_prose_0"
        assert chunk.metadata.company_ticker == "AAPL"
        assert chunk.metadata.fiscal_year == 2024

    def test_metadata_defaults(self) -> None:
        """Test that optional fields have correct defaults."""
        metadata = ChunkMetadata(
            company_ticker="MSFT",
            filing_type="10-Q",
            fiscal_year=2024,
            section="item1_business",
            chunk_type="parent",
            chunk_id="test_chunk_1",
        )

        assert metadata.company_name == ""
        assert metadata.parent_chunk_id is None
        assert metadata.referenced_notes == []
        assert metadata.xbrl_concepts == []
        assert metadata.token_count == 0
        assert metadata.content_type == "prose"

    def test_metadata_serialization(self) -> None:
        """Test JSON serialization of metadata (for Qdrant payload)."""
        metadata = ChunkMetadata(
            company_ticker="AAPL",
            filing_type="10-K",
            fiscal_year=2024,
            section="item8_notes",
            chunk_type="child",
            chunk_id="test_note_1",
            referenced_notes=["Note 2", "Note 5"],
            xbrl_concepts=["us-gaap:Revenues"],
            content_type="note",
            note_id="Note 12",
        )

        payload = metadata.model_dump()
        assert isinstance(payload, dict)
        assert payload["company_ticker"] == "AAPL"
        assert payload["referenced_notes"] == ["Note 2", "Note 5"]
        assert payload["note_id"] == "Note 12"

    def test_chunk_str_representation(self) -> None:
        """Test string representation of a chunk."""
        metadata = ChunkMetadata(
            company_ticker="AAPL",
            filing_type="10-K",
            fiscal_year=2024,
            section="item7_mda",
            chunk_type="child",
            chunk_id="test_1",
            token_count=256,
        )
        chunk = Chunk(chunk_id="test_1", text="test text", metadata=metadata)
        s = str(chunk)
        assert "test_1" in s
        assert "child" in s
        assert "256" in s
