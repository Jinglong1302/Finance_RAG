"""Unit tests for CRAGState and graph routing."""

import pytest
from src.orchestration.state import CRAGState
from src.orchestration.graph import route_after_grading, route_after_guard


class TestCRAGState:
    """Tests for CRAGState TypedDict usage."""

    def test_partial_initialization(self) -> None:
        """Test that CRAGState can be partially initialized."""
        state: CRAGState = {
            "original_query": "What was Apple's revenue in 2024?",
            "cycle_count": 0,
            "cost_accumulated": 0.0,
        }
        assert state["original_query"] == "What was Apple's revenue in 2024?"
        assert state["cycle_count"] == 0

    def test_state_update(self) -> None:
        """Test state update pattern used by nodes."""
        state: CRAGState = {
            "original_query": "test",
            "cycle_count": 0,
            "cost_accumulated": 0.0,
        }
        update = {"sub_queries": ["test"], "query_type": "factual_numeric"}
        state.update(update)
        assert state["sub_queries"] == ["test"]
        assert state["query_type"] == "factual_numeric"


class TestGraphRouting:
    """Tests for conditional routing functions."""

    def test_route_after_grading_generate(self) -> None:
        state: CRAGState = {"crag_action": "generate"}
        assert route_after_grading(state) == "generate"

    def test_route_after_grading_rewrite(self) -> None:
        state: CRAGState = {"crag_action": "rewrite"}
        assert route_after_grading(state) == "rewrite"

    def test_route_after_grading_refuse(self) -> None:
        state: CRAGState = {"crag_action": "refuse"}
        assert route_after_grading(state) == "refuse"

    def test_route_after_grading_default(self) -> None:
        state: CRAGState = {}
        assert route_after_grading(state) == "generate"

    def test_route_after_guard_pass(self) -> None:
        state: CRAGState = {"hallucination_check": "pass", "cycle_count": 0}
        assert route_after_guard(state) == "end"

    def test_route_after_guard_fail_cycle0(self) -> None:
        state: CRAGState = {"hallucination_check": "fail", "cycle_count": 0}
        assert route_after_guard(state) == "rewrite"

    def test_route_after_guard_fail_at_max(self) -> None:
        state: CRAGState = {"hallucination_check": "fail", "cycle_count": 2}
        assert route_after_guard(state) == "end"

    def test_route_after_guard_error(self) -> None:
        state: CRAGState = {"hallucination_check": "error", "cycle_count": 0}
        assert route_after_guard(state) == "end"
