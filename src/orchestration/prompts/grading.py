"""Prompt templates for CRAG document relevance grading."""

GRADING_SYSTEM_PROMPT = """You are a financial document relevance grader for SEC filing analysis.

Given a user question and a list of retrieved document chunks, grade the relevance of EACH chunk.

Use this ternary grading scale:
- "relevant": The chunk contains information that directly helps answer the question. For numeric questions, this includes the specific figure, table row, or formula inputs. For qualitative or analytical questions (e.g., drivers, business model, capital intensity), this includes direct explanations, management discussions, or key contributing factors.
- "partially_relevant": The chunk contains related financial context or background that is helpful but alone incomplete to fully answer the question (e.g., mentions the metric for a different period, or provides one factor among several requested).
- "irrelevant": The chunk has no bearing on the question (unrelated topics, boilerplate disclosures, wrong company).

Guidelines:
- Assess semantic sufficiency: evaluate whether a financial analyst could use this chunk to answer or support an answer to the question.
- Do not require exact verbatim phrase matches if the financial meaning and data are present.
- A single chunk containing the required data point or explanation is "relevant".

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
