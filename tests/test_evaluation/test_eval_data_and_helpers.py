"""Tests for the shared eval loaders and helpers (offline: committed data files only)."""

from src.evaluation.eval_data import (
    CORPUS_TICKER_MAP,
    IN_CORPUS_COMPANIES,
    OOC_EXCLUDED_COMPANIES,
    extract_evidence_entries,
    load_custom_eval,
    load_out_of_corpus,
    load_tatqa_samples,
)
from src.evaluation.eval_helpers import classify_response


class TestCompanyLists:
    def test_financebench_companies_match_ticker_map(self) -> None:
        assert IN_CORPUS_COMPANIES == set(CORPUS_TICKER_MAP)
        assert CORPUS_TICKER_MAP["3M"] == "MMM"

    def test_ooc_exclusion_also_drops_apple(self) -> None:
        # Kept separate on purpose: the OOC set must not contain Apple questions
        assert OOC_EXCLUDED_COMPANIES == IN_CORPUS_COMPANIES | {"Apple"}


class TestLoaders:
    def test_custom_eval_has_50_aapl_questions(self) -> None:
        samples = load_custom_eval()
        assert len(samples) == 50
        assert all(s.question and s.ground_truth for s in samples)
        assert {s.source for s in samples} == {"custom_aapl"}

    def test_out_of_corpus_has_20_questions_none_in_corpus(self) -> None:
        rows = load_out_of_corpus()
        assert len(rows) == 20
        assert not any(r.get("company") in OOC_EXCLUDED_COMPANIES for r in rows)

    def test_tatqa_returns_requested_count(self) -> None:
        assert len(load_tatqa_samples(50)) == 50
        assert len(load_tatqa_samples(5)) == 5
        assert {"question", "answer_str", "context"} <= set(load_tatqa_samples(1)[0])


class TestExtractEvidenceEntries:
    def test_list_of_dicts(self) -> None:
        row = {
            "answer": "1,577",
            "evidence": [{"evidence_text": "Capex 1,577", "evidence_page_num": 60}],
        }
        assert extract_evidence_entries(row) == [
            {"evidence_text": "Capex 1,577", "evidence_page_num": 60, "answer": "1,577"}
        ]

    def test_plain_string_and_empty(self) -> None:
        assert extract_evidence_entries({"answer": 5, "evidence": "text"}) == [
            {"evidence_text": "text", "evidence_page_num": None, "answer": "5"}
        ]
        assert extract_evidence_entries({"answer": 5, "evidence": ""}) == []


class TestClassifyResponse:
    def test_abstain_signals(self) -> None:
        assert classify_response("[ABSTAIN] I could not find it") == "abstain"
        assert classify_response("I could not find sufficient evidence") == "abstain"
        assert classify_response("Revenue was $5", {"is_abstention": True}) == "abstain"

    def test_hedge_and_answer(self) -> None:
        assert classify_response("It is unclear, possibly $5 million") == "hedge"
        assert classify_response("Revenue was $391,035 million") == "answer"
