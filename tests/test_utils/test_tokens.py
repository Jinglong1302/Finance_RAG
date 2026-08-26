"""Unit tests for token counting and cost estimation utilities."""

import pytest
from src.utils.tokens import count_tokens, truncate_to_tokens, estimate_cost


class TestCountTokens:
    """Tests for token counting."""

    def test_count_simple_text(self) -> None:
        count = count_tokens("Hello, world!")
        assert count > 0
        assert count < 10  # Should be around 4 tokens

    def test_count_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_count_financial_text(self) -> None:
        text = "Apple Inc. reported total net revenue of $394,328 million for fiscal year 2024."
        count = count_tokens(text)
        assert 10 < count < 30

    def test_count_table_markdown(self) -> None:
        text = "| Revenue | $394B | $383B |\n| Net Income | $93B | $97B |"
        count = count_tokens(text)
        assert count > 0


class TestTruncateToTokens:
    """Tests for token truncation."""

    def test_no_truncation_needed(self) -> None:
        text = "Short text"
        result = truncate_to_tokens(text, max_tokens=100)
        assert result == text

    def test_truncation(self) -> None:
        text = "This is a longer piece of text that should be truncated. " * 50
        result = truncate_to_tokens(text, max_tokens=10)
        result_tokens = count_tokens(result)
        assert result_tokens <= 10


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_gpt4o_cost(self) -> None:
        cost = estimate_cost(1000, 500, "gpt-4o")
        # 1000 * 2.5/1M + 500 * 10/1M = 0.0025 + 0.005 = 0.0075
        assert cost == pytest.approx(0.0075, rel=0.01)

    def test_gpt4o_mini_cost(self) -> None:
        cost = estimate_cost(1000, 500, "gpt-4o-mini")
        # 1000 * 0.15/1M + 500 * 0.60/1M = 0.00015 + 0.0003 = 0.00045
        assert cost == pytest.approx(0.00045, rel=0.01)

    def test_zero_tokens(self) -> None:
        cost = estimate_cost(0, 0, "gpt-4o")
        assert cost == 0.0

    def test_unknown_model_defaults_to_gpt4o(self) -> None:
        cost = estimate_cost(1000, 500, "some-future-model")
        expected = estimate_cost(1000, 500, "gpt-4o")
        assert cost == expected
