"""Financial acronym and synonym expansion module.

Maps common financial abbreviations, acronyms, and colloquial financial jargon
to official SEC filing terminology (e.g. "net PPNE" -> "property, plant and equipment, net").
"""

from __future__ import annotations

import re

# Lookup table mapping financial acronyms/abbreviations to filing-text equivalents
FINANCIAL_ACRONYMS: dict[str, str] = {
    r"\bnet\s+pp[&n]?e\b": "property, plant and equipment, net",
    r"\bpp[&n]e\b": "property, plant and equipment",
    r"\bcapex\b": "capital expenditures purchases of property, plant and equipment",
    r"\bcogs\b": "cost of goods sold cost of sales cost of products sold",
    r"\bsg&a\b|\bsga\b": "selling, general and administrative expenses",
    r"\bebitda\b": "operating income depreciation and amortization",
    r"\bebit\b": "operating income",
    r"\bd&a\b": "depreciation and amortization",
    r"\beps\b": "earnings per share net income per share",
    r"\bdiluted\s+eps\b": "diluted earnings per share",
    r"\br&d\b|\brnd\b": "research and development",
    r"\broa\b": "return on assets net income total assets",
    r"\broe\b": "return on equity net income shareholders equity",
    r"\bfcf\b": "free cash flow operating cash flows capital expenditures",
    r"\bp&l\b": "statement of income statement of operations",
    r"\bnwc\b|\bworking\s+capital\b": "current assets current liabilities operating working capital",
}


def expand_financial_query(query: str) -> str:
    """Expand financial acronyms in a query by appending filing terminology.

    Appends expanded terms to preserve the original query semantics while
    ensuring dense and sparse retrieval match exact filing line items.

    Args:
        query: Original user or sub-query string.

    Returns:
        Expanded query string.
    """
    if not query:
        return query

    expansions: list[str] = []
    for pattern, replacement in FINANCIAL_ACRONYMS.items():
        if re.search(pattern, query, re.IGNORECASE):
            # Check if replacement terms aren't already prominently in query
            key_terms = replacement.split()[:2]
            if not all(kt.lower() in query.lower() for kt in key_terms):
                expansions.append(replacement)

    if expansions:
        # Append unique expansions
        unique_expansions = list(dict.fromkeys(expansions))
        return f"{query} {' '.join(unique_expansions)}"
    return query
