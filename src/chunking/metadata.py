"""Chunk metadata schema.

Pydantic models defining the metadata contract between ingestion,
chunking, embedding, and retrieval layers. Every chunk stored in
Qdrant carries this metadata.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk stored in Qdrant.

    Fields are divided into indexed (used for pre-filtering in Qdrant)
    and unindexed (carried as payload for generation context).
    """

    # === Indexed fields (used for Qdrant pre-filtering) ===
    company_ticker: str = Field(description="Stock ticker symbol, e.g. 'AAPL'")
    filing_type: str = Field(description="SEC filing type, e.g. '10-K'")
    fiscal_year: int | None = Field(default=None, description="Fiscal year of the filing")
    section: str = Field(description="Section ID, e.g. 'item8_financial_statements'")
    section_type: str = Field(
        default="other",
        description="Standardized section category: balance_sheet, income_statement, cash_flow_statement, mda, notes, risk_factors, cover_page, other",
    )
    chunk_type: str = Field(description="'child' or 'parent'")

    # === Unindexed payload fields ===
    company_name: str = Field(default="", description="Full company name")
    filing_date: str = Field(default="", description="Filing date (ISO format)")
    section_title: str = Field(default="", description="Human-readable section title")
    page_number: int | None = Field(default=None, description="Approximate page number")
    table_id: str | None = Field(default=None, description="Table identifier")
    parent_chunk_id: str | None = Field(
        default=None, description="Parent chunk ID for child-to-parent linking"
    )
    chunk_id: str = Field(description="Unique chunk identifier")
    chunk_index: int = Field(default=0, description="Sequential document order index")
    referenced_notes: list[str] = Field(
        default_factory=list,
        description="Note references found in text, e.g. ['Note 2', 'Note 12']",
    )
    xbrl_concepts: list[str] = Field(
        default_factory=list,
        description="US-GAAP concepts present in chunk, e.g. ['us-gaap:Revenues']",
    )
    table_format: str | None = Field(
        default=None, description="'markdown' or 'html_fallback'"
    )
    token_count: int = Field(default=0, description="Token count of chunk text")
    content_type: str = Field(
        default="prose", description="'prose', 'table', or 'note'"
    )
    note_id: str | None = Field(
        default=None, description="Note identifier, e.g. 'Note 12'"
    )
    note_title: str | None = Field(
        default=None, description="Note title, e.g. 'Segment Reporting'"
    )


class Chunk(BaseModel):
    """A self-contained text chunk with metadata, ready for embedding.

    Represents the atomic unit of information in the RAG system.
    Each chunk has text content and a ChunkMetadata instance describing
    its origin, type, and relationships.
    """

    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk text content")
    metadata: ChunkMetadata = Field(description="Chunk metadata")

    def __str__(self) -> str:
        return (
            f"Chunk({self.chunk_id}, "
            f"type={self.metadata.chunk_type}, "
            f"tokens={self.metadata.token_count})"
        )


def classify_section_type(
    section_id: str,
    section_name: str,
    title: str = "",
    text: str = "",
) -> str:
    """Classify a chunk into a standardized section_type.

    Categories:
    - balance_sheet
    - income_statement
    - cash_flow_statement
    - equity_statement
    - mda
    - notes
    - risk_factors
    - cover_page
    - other
    """
    s_id = (section_id or "").lower()
    s_name = (section_name or "").lower()
    t_title = (title or "").lower()

    # Notes sections
    if "notes" in s_id or s_id == "item8_notes" or "notes to" in s_name or t_title.startswith("note "):
        return "notes"

    # MD&A sections
    if "mda" in s_id or "management" in s_name:
        return "mda"

    # Risk Factors
    if "risk" in s_id or "risk factors" in s_name:
        return "risk_factors"

    # Cover page
    if "cover" in s_id or "cover_page" in s_id:
        return "cover_page"

    # Only primary financial statements (Item 8 / Item 1) get statement-specific types
    if "financial_statements" in s_id or s_id in ("item8", "item1", "item8_financial_statements", "item1_financial_statements"):
        combined_title = f"{t_title} {text[:200].lower()}"
        if "balance sheet" in combined_title or "financial position" in combined_title:
            return "balance_sheet"
        if "cash flow" in combined_title or "cash flows" in combined_title:
            return "cash_flow_statement"
        if any(
            k in combined_title
            for k in [
                "statement of income",
                "statements of income",
                "statement of operations",
                "statements of operations",
                "statement of earnings",
                "statements of earnings",
                "income statement",
            ]
        ):
            return "income_statement"
        if "equity" in combined_title:
            return "equity_statement"
        return "financial_statements"

    return "other"

