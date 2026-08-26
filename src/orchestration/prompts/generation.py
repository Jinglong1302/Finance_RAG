"""Prompt templates for grounded answer generation with citations."""

GENERATION_SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions about SEC filings with precision and citations.

RULES:
1. ONLY use information from the provided context. Do NOT use prior knowledge.
2. Every financial number MUST have a citation [1], [2], etc. referencing the source chunk.
3. If the context does not contain sufficient information, say "I could not find sufficient evidence in the provided filings to answer this question." Do NOT fabricate data.
4. Show calculations step by step when performing arithmetic (e.g., growth rates, margins, ratios).
5. Use the confidence level provided to calibrate your response language.

CITATION FORMAT:
After your answer, provide a Sources section:
[1] {Company} {FilingType} (FY{Year}), {Section} - "{verbatim evidence snippet}"
[2] ...

STRUCTURED METRICS (when applicable):
If your answer includes specific financial metrics, also output them in a JSON block:
```json
{{"metrics": [{{"name": "Total Revenue", "value": 394328, "unit": "millions USD", "period": "FY2024", "company": "Apple Inc.", "citation": 1}}]}}
```"""

GENERATION_USER_PROMPT = """Answer the following question using ONLY the provided context.
Confidence Level: {confidence}

Question: {query}

Context:
{context}"""


def format_context_for_generation(
    enriched_contexts: list[dict],
    include_parent: bool = True,
) -> str:
    """Format enriched contexts for the generation prompt.

    Args:
        enriched_contexts: List of context dicts with child_text,
                          parent_text, and metadata.
        include_parent: Whether to include parent context. Default: True.

    Returns:
        Formatted context string with numbered sources.
    """
    parts = []
    for i, ctx in enumerate(enriched_contexts):
        metadata = ctx.get("metadata", {})
        company = metadata.get("company_ticker", "Unknown")
        section = metadata.get("section_title", "Unknown")
        year = metadata.get("fiscal_year", "Unknown")
        filing = metadata.get("filing_type", "10-K")

        header = f"[Source {i + 1}] {company} {filing} (FY{year}) - {section}"
        parts.append(header)

        # Child text (primary evidence)
        parts.append(ctx.get("child_text", ""))

        # Parent text (expanded context) — truncated to save tokens
        if include_parent and ctx.get("parent_text"):
            parent = ctx["parent_text"]
            if len(parent) > 2000:
                parent = parent[:2000] + "... [truncated]"
            parts.append(f"\n[Expanded Context for Source {i + 1}]")
            parts.append(parent)

        parts.append("")  # Blank line separator

    return "\n".join(parts)
