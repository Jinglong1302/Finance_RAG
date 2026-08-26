"""Prompt templates for CRAG document relevance grading."""

GRADING_SYSTEM_PROMPT = """You are a financial document relevance grader for SEC filing analysis.

Given a user question and a list of retrieved document chunks, grade the relevance of EACH chunk.

Use this ternary grading scale:
- "relevant": The chunk directly contains specific information needed to answer the question (exact numbers, direct statements, relevant tables).
- "partially_relevant": The chunk contains related financial context but not the specific data point needed (e.g., mentions the topic but different time period, or discusses the metric qualitatively but lacks the number).
- "irrelevant": The chunk has no bearing on the question (wrong company, wrong topic, boilerplate text).

IMPORTANT: Be strict. Financial accuracy requires precise data. A chunk discussing "revenue trends" is only "relevant" if it contains the actual revenue figure the user asked about. Otherwise it is "partially_relevant" at best.

Respond ONLY with valid JSON in this exact format:
{
  "grades": [
    {"chunk_index": 0, "relevance": "relevant", "reason": "Contains FY2024 revenue figure of $394.3B"},
    {"chunk_index": 1, "relevance": "partially_relevant", "reason": "Discusses revenue growth but for FY2023"},
    {"chunk_index": 2, "relevance": "irrelevant", "reason": "Balance sheet data, not income statement"}
  ]
}"""

GRADING_USER_PROMPT = """Grade the relevance of each chunk for answering this question.

Question: {query}

{chunks_text}"""


def format_chunks_for_grading(chunks: list[dict]) -> str:
    """Format chunks for the grading prompt.

    Args:
        chunks: List of chunk dicts with 'text' and 'metadata'.

    Returns:
        Formatted string with numbered chunks.
    """
    parts = []
    for i, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        company = metadata.get("company_ticker", "Unknown")
        section = metadata.get("section_title", "Unknown Section")
        year = metadata.get("fiscal_year", "Unknown")

        parts.append(
            f"--- Chunk {i} [{company} | {section} | FY{year}] ---\n"
            f"{chunk.get('text', '')[:1500]}"  # Truncate to control token usage
        )
    return "\n\n".join(parts)
