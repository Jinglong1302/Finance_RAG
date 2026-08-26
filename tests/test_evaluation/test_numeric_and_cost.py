"""Unit tests for numeric evaluation and cost tracking."""

import pytest
from src.evaluation.numeric_eval import numeric_match, parse_financial_number
from src.evaluation.cost_tracker import CostTracker, BudgetExceededError


class TestParseFinancialNumber:
    """Tests for financial number parsing."""

    def test_plain_integer(self) -> None:
        assert parse_financial_number("394328") == pytest.approx(394328.0)

    def test_with_commas(self) -> None:
        assert parse_financial_number("394,328") == pytest.approx(394328.0)

    def test_with_dollar_sign(self) -> None:
        assert parse_financial_number("$394,328") == pytest.approx(394328.0)

    def test_billions_suffix(self) -> None:
        assert parse_financial_number("$394.3B") == pytest.approx(394.3e9)

    def test_millions_word(self) -> None:
        assert parse_financial_number("$394,328 million") == pytest.approx(394328e6)

    def test_negative_parenthetical(self) -> None:
        assert parse_financial_number("($1,234)") == pytest.approx(-1234.0)

    def test_negative_dash(self) -> None:
        assert parse_financial_number("-$1.5M") == pytest.approx(-1.5e6)

    def test_percentage(self) -> None:
        assert parse_financial_number("2.88%") == pytest.approx(2.88)

    def test_empty_string(self) -> None:
        assert parse_financial_number("") is None

    def test_non_numeric(self) -> None:
        assert parse_financial_number("not a number") is None


class TestNumericMatch:
    """Tests for numeric matching with tolerance."""

    def test_exact_match(self) -> None:
        assert numeric_match("394328", "394328") is True

    def test_format_difference(self) -> None:
        assert numeric_match("$394,328", "394328") is True

    def test_billions_match(self) -> None:
        assert numeric_match("$394.3B", "$394,300 million") is True

    def test_percentage_match(self) -> None:
        assert numeric_match("2.88%", "2.88 percent") is True

    def test_mismatch(self) -> None:
        assert numeric_match("$394B", "$383B") is False

    def test_within_tolerance(self) -> None:
        # 394,328 vs 394,000 = ~0.08% difference
        assert numeric_match("394328", "394000", tolerance=0.01) is True

    def test_outside_tolerance(self) -> None:
        assert numeric_match("400000", "394000", tolerance=0.01) is False


class TestCostTracker:
    """Tests for API cost tracking."""

    def test_log_call(self) -> None:
        tracker = CostTracker(budget_per_query=1.0, budget_per_run=100.0)
        tracker.start_query()
        cost = tracker.log_call("grading", "gpt-4o", 1000, 200)
        assert cost > 0

    def test_query_summary(self) -> None:
        tracker = CostTracker(budget_per_query=1.0)
        tracker.start_query()
        tracker.log_call("decomposition", "gpt-4o", 500, 100)
        tracker.log_call("grading", "gpt-4o", 2000, 500)

        summary = tracker.get_query_summary()
        assert summary["calls"] == 2
        assert "decomposition" in summary["by_stage"]
        assert "grading" in summary["by_stage"]
        assert summary["total_cost_usd"] > 0

    def test_budget_exceeded(self) -> None:
        tracker = CostTracker(budget_per_query=0.001)
        tracker.start_query()
        with pytest.raises(BudgetExceededError):
            tracker.log_call("generation", "gpt-4o", 100000, 50000)

    def test_run_summary(self) -> None:
        tracker = CostTracker()
        tracker.start_query()
        tracker.log_call("decomposition", "gpt-4o", 500, 100)
        tracker.start_query()
        tracker.log_call("decomposition", "gpt-4o", 600, 150)

        summary = tracker.get_run_summary()
        assert summary["total_queries"] == 2
        assert summary["total_calls"] == 2
        assert summary["avg_cost_per_query"] > 0
