"""Cost tracker for OpenAI API usage.

Tracks per-query and per-run costs with budget enforcement.
Generates detailed cost breakdowns by pipeline stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils.logging import get_logger
from src.utils.tokens import estimate_cost

logger = get_logger(__name__)


class BudgetExceededError(Exception):
    """Raised when a cost budget is exceeded."""


@dataclass
class APICallRecord:
    """Record of a single API call."""

    stage: str  # e.g., "decomposition", "grading", "generation"
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostTracker:
    """Tracks OpenAI API costs across queries and evaluation runs.

    Enforces per-query and per-run budget caps. Generates detailed
    cost reports broken down by pipeline stage.

    Args:
        budget_per_query: Maximum USD cost per single query. Default: 0.10.
        budget_per_run: Maximum USD cost per evaluation run. Default: 50.0.
    """

    def __init__(
        self,
        budget_per_query: float = 0.10,
        budget_per_run: float = 50.0,
    ) -> None:
        self.budget_per_query = budget_per_query
        self.budget_per_run = budget_per_run

        self._query_cost: float = 0.0
        self._run_cost: float = 0.0
        self._query_records: list[APICallRecord] = []
        self._all_records: list[APICallRecord] = []
        self._query_count: int = 0

    def log_call(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Log an API call and check budget.

        Args:
            stage: Pipeline stage name.
            model: Model used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost of this call in USD.

        Raises:
            BudgetExceededError: If budget is exceeded.
        """
        cost = estimate_cost(input_tokens, output_tokens, model)

        record = APICallRecord(
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        self._query_cost += cost
        self._run_cost += cost
        self._query_records.append(record)
        self._all_records.append(record)

        # Check budgets
        if self._query_cost > self.budget_per_query:
            logger.warning(
                f"Query budget exceeded: ${self._query_cost:.4f} > "
                f"${self.budget_per_query:.2f}"
            )
            raise BudgetExceededError(
                f"Query budget exceeded: ${self._query_cost:.4f}"
            )

        if self._run_cost > self.budget_per_run:
            logger.warning(
                f"Run budget exceeded: ${self._run_cost:.2f} > "
                f"${self.budget_per_run:.2f}"
            )
            raise BudgetExceededError(
                f"Run budget exceeded: ${self._run_cost:.2f}"
            )

        return cost

    def start_query(self) -> None:
        """Reset per-query tracking for a new query."""
        self._query_cost = 0.0
        self._query_records = []
        self._query_count += 1

    def get_query_summary(self) -> dict[str, Any]:
        """Get cost summary for the current query.

        Returns:
            Dict with cost breakdown by stage.
        """
        by_stage: dict[str, float] = {}
        for record in self._query_records:
            by_stage[record.stage] = by_stage.get(record.stage, 0.0) + record.cost_usd

        return {
            "total_cost_usd": self._query_cost,
            "calls": len(self._query_records),
            "by_stage": by_stage,
        }

    def get_run_summary(self) -> dict[str, Any]:
        """Get cost summary for the entire run.

        Returns:
            Dict with aggregate cost statistics.
        """
        by_stage: dict[str, float] = {}
        by_model: dict[str, float] = {}
        total_input = 0
        total_output = 0

        for record in self._all_records:
            by_stage[record.stage] = by_stage.get(record.stage, 0.0) + record.cost_usd
            by_model[record.model] = by_model.get(record.model, 0.0) + record.cost_usd
            total_input += record.input_tokens
            total_output += record.output_tokens

        return {
            "total_cost_usd": self._run_cost,
            "total_calls": len(self._all_records),
            "total_queries": self._query_count,
            "avg_cost_per_query": (
                self._run_cost / self._query_count
                if self._query_count > 0
                else 0.0
            ),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "by_stage": by_stage,
            "by_model": by_model,
        }
