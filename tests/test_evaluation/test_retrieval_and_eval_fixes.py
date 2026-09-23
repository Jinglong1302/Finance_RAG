"""Unit tests for retrieval filter expansion, answer cleaning, and evaluation context fixes."""

import pytest
from qdrant_client.models import FieldCondition, MatchAny

from src.evaluation.ragas_eval import (
    _clean_answer_for_ragas,
    _format_contexts_for_ragas,
    _is_refusal,
)
from src.retrieval.hybrid_search import HybridSearcher


class TestFilterExpansion:
    """Tests for fiscal year filter expansion in HybridSearcher._build_filter."""

    def test_single_fiscal_year_expanded_to_three_years(self):
        """A single fiscal year Y should expand to [Y, Y+1, Y+2] to match 3-year 10-K comparative tables."""
        # Instantiate a dummy HybridSearcher without network calls
        searcher = object.__new__(HybridSearcher)
        flt = searcher._build_qdrant_filter({"company_ticker": "AAPL", "fiscal_year": 2022})

        assert flt is not None
        assert flt.must is not None
        assert len(flt.must) == 2

        # Check fiscal_year condition
        fy_cond = next(c for c in flt.must if isinstance(c, FieldCondition) and c.key == "fiscal_year")
        assert isinstance(fy_cond.match, MatchAny)
        assert fy_cond.match.any == [2022, 2023, 2024]

    def test_list_fiscal_year_preserved(self):
        """A list of fiscal years should match exactly the specified years."""
        searcher = object.__new__(HybridSearcher)
        flt = searcher._build_qdrant_filter({"fiscal_year": [2023, 2024]})

        assert flt is not None
        fy_cond = next(c for c in flt.must if isinstance(c, FieldCondition) and c.key == "fiscal_year")
        assert isinstance(fy_cond.match, MatchAny)
        assert fy_cond.match.any == [2023, 2024]


class TestAnswerCleaningForRagas:
    """Tests for _clean_answer_for_ragas."""

    def test_strip_json_and_sources(self):
        raw_answer = (
            "Apple's total net sales in FY2024 were $391,035 million [1].\n\n"
            "Sources:\n"
            '[1] AAPL 10-K (FY2024), Financial Statements - "Total net sales $ 391,035"\n\n'
            "```json\n"
            '{\n  "metrics": [{"name": "Total Net Sales", "value": 391035}]\n}\n'
            "```"
        )
        cleaned = _clean_answer_for_ragas(raw_answer)
        assert cleaned == "Apple's total net sales in FY2024 were $391,035 million [1]."

    def test_strip_verification_notes(self):
        raw_answer = (
            "Operating expenses were $241,764 million.\n\n"
            "⚠️ **Verification Notes:**\n"
            "- [UNVERIFIED] R&D expenses not found"
        )
        cleaned = _clean_answer_for_ragas(raw_answer)
        assert cleaned == "Operating expenses were $241,764 million."

    def test_clean_empty_or_plain_string(self):
        assert _clean_answer_for_ragas("") == ""
        assert _clean_answer_for_ragas("Direct plain answer.") == "Direct plain answer."


class TestRefusalDetection:
    """Tests for _is_refusal."""

    def test_identifies_refusals(self):
        assert _is_refusal("I could not find sufficient evidence in the available SEC filings.")
        assert _is_refusal("Insufficient evidence to verify this claim.")
        assert _is_refusal("There is not enough information to calculate net sales.")

    def test_does_not_flag_valid_answers(self):
        assert not _is_refusal("Apple's net sales in FY2024 were $391,035 million.")
        assert not _is_refusal("Gross margin increased by 5.2% year-over-year.")


class TestContextFormattingForRagas:
    """Tests for _format_contexts_for_ragas."""

    def test_formats_header_child_and_parent(self):
        ctxs = [
            {
                "child_text": "Total net sales: $391,035 million",
                "parent_text": "Item 8. Consolidated Statements of Operations ... Full table",
                "metadata": {
                    "company_ticker": "AAPL",
                    "fiscal_year": 2024,
                    "section_title": "Financial Statements",
                    "filing_type": "10-K",
                },
            }
        ]
        formatted = _format_contexts_for_ragas(ctxs)
        assert len(formatted) == 1
        assert "[AAPL 10-K (FY2024) - Financial Statements]" in formatted[0]
        assert "Total net sales: $391,035 million" in formatted[0]
        assert "[Expanded Context]" in formatted[0]
        assert "Consolidated Statements of Operations" in formatted[0]
