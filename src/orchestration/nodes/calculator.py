"""Python REPL calculator tool for numerical reasoning.

Provides sandboxed execution of Python expressions for financial
calculations like CAGR, growth rates, and margins.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Safe builtins for sandboxed execution
_SAFE_GLOBALS = {
    "__builtins__": {},
    "math": math,
    "Decimal": Decimal,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "float": float,
    "int": int,
}


def calculate(expression: str) -> str:
    """Execute a Python expression for financial calculations.

    Runs in a sandboxed environment with only math, Decimal, and
    basic numeric functions available.

    Args:
        expression: A Python expression string.
            Examples:
            - "((394328 / 383285) - 1) * 100"  # YoY growth
            - "(394328 / 383285) ** (1/3) - 1"  # CAGR
            - "183976 / 394328 * 100"  # Gross margin %

    Returns:
        String representation of the result.
    """
    try:
        result = eval(expression, _SAFE_GLOBALS, {})
        # Format the result
        if isinstance(result, float):
            # Round to reasonable precision
            if abs(result) < 1:
                formatted = f"{result:.4f}"
            elif abs(result) < 100:
                formatted = f"{result:.2f}"
            else:
                formatted = f"{result:,.2f}"
        else:
            formatted = str(result)

        logger.info(f"Calculated: {expression} = {formatted}")
        return formatted

    except Exception as e:
        error_msg = f"Calculation error: {e}"
        logger.warning(error_msg)
        return error_msg
