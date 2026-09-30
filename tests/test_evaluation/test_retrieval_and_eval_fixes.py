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


class TestSemanticSpanMatch:
    """Tests for span_match, normalize_span, and span_match_rate."""

    def test_tatqa_q1_article_normalization(self):
        from src.evaluation.metrics import span_match
        gt = "before provision for income taxes"
        pred = "before the provision for income taxes"
        assert span_match(pred, gt)

    def test_span_within_full_sentence(self):
        from src.evaluation.metrics import span_match
        gt = "before provision for income taxes"
        pred = "Based on the table, the line item was before the provision for income taxes."
        assert span_match(pred, gt)

    def test_numeric_span_match(self):
        from src.evaluation.metrics import span_match
        assert span_match("$1,577 million", "$1577.00")
        assert span_match("operating margin fell by 1.7%", "1.7%")


class Test4WayClassification:
    """Tests for classify_qa_result."""

    def test_answered_correct(self):
        from src.evaluation.metrics import classify_qa_result
        res = classify_qa_result(
            answer="Capital expenditures were $1,577 million in FY2018.",
            ground_truth="$1577.00",
            is_indexed=True,
            is_refusal=False,
        )
        assert res == "answered-correct"

    def test_answered_incorrect(self):
        from src.evaluation.metrics import classify_qa_result
        res = classify_qa_result(
            answer="Capital expenditures were $9,999 million.",
            ground_truth="$1577.00",
            is_indexed=True,
            is_refusal=False,
        )
        assert res == "answered-incorrect"

    def test_abstained_correctly_real_gap(self):
        from src.evaluation.metrics import classify_qa_result
        res = classify_qa_result(
            answer="[ABSTAIN] Not indexed in available SEC filings.",
            ground_truth="N/A",
            is_indexed=False,
            is_refusal=True,
        )
        assert res == "abstained-correctly-real-gap"

    def test_abstained_incorrectly_indexed_gap(self):
        from src.evaluation.metrics import classify_qa_result
        res = classify_qa_result(
            answer="[ABSTAIN] Could not find evidence.",
            ground_truth="$1577.00",
            is_indexed=True,
            is_refusal=True,
        )
        assert res == "abstained-incorrectly"

    def test_excluded_fiscal_year_not_indexed(self):
        from src.evaluation.metrics import classify_qa_result
        res = classify_qa_result(
            answer="I think the value was $100.",
            ground_truth="$100",
            is_indexed=False,
            is_refusal=False,
        )
        assert res == "excluded: fiscal year not indexed"
