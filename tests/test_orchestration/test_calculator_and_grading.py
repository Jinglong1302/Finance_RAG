"""Unit tests for the calculator tool and CRAG grading logic."""

import pytest
from src.orchestration.nodes.calculator import calculate
from src.orchestration.nodes.grader import _determine_action


class TestCalculator:
    """Tests for the sandboxed Python calculator."""

    def test_basic_arithmetic(self) -> None:
        result = calculate("2 + 3")
        assert result == "5"

    def test_yoy_growth(self) -> None:
        """Test year-over-year growth calculation."""
        result = calculate("((394328 / 383285) - 1) * 100")
        assert float(result) == pytest.approx(2.88, rel=0.01)

    def test_gross_margin_percentage(self) -> None:
        result = calculate("183976 / 394328 * 100")
        assert float(result) == pytest.approx(46.65, rel=0.01)

    def test_cagr(self) -> None:
        result = calculate("((394328 / 365817) ** (1/3) - 1) * 100")
        assert float(result) > 0

    def test_division_by_zero(self) -> None:
        result = calculate("1 / 0")
        assert "error" in result.lower()

    def test_restricted_builtins(self) -> None:
        """Test that dangerous operations are blocked."""
        result = calculate("__import__('os').system('echo hacked')")
        assert "error" in result.lower()


class TestGradingDecisionMatrix:
    """Tests for the CRAG grading decision matrix."""

    def test_high_confidence_generate(self) -> None:
        conf, action = _determine_action(3, 1, 1, cycle_count=0, max_cycles=2)
        assert conf == "high"
        assert action == "generate"

    def test_medium_confidence_generate(self) -> None:
        conf, action = _determine_action(1, 2, 2, cycle_count=0, max_cycles=2)
        assert conf == "medium"
        assert action == "generate"

    def test_low_confidence_rewrite_cycle0(self) -> None:
        conf, action = _determine_action(0, 3, 2, cycle_count=0, max_cycles=2)
        assert conf == "low"
        assert action == "rewrite"

    def test_low_confidence_generate_at_max_cycles(self) -> None:
        conf, action = _determine_action(0, 3, 2, cycle_count=2, max_cycles=2)
        assert conf == "low"
        assert action == "generate"

    def test_insufficient_rewrite(self) -> None:
        conf, action = _determine_action(0, 0, 5, cycle_count=0, max_cycles=2)
        assert conf == "insufficient"
        assert action == "rewrite"

    def test_insufficient_refuse_at_max_cycles(self) -> None:
        conf, action = _determine_action(0, 0, 5, cycle_count=2, max_cycles=2)
        assert conf == "insufficient"
        assert action == "refuse"
