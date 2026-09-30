"""Prompt templates for pooled-context sufficiency judgment.

Part B replacement for per-chunk relevance grading.  Instead of grading
each chunk individually, a single LLM call judges whether the *concatenated*
top-k context is collectively sufficient to answer the question.

Design based on:
  - LangGraph official CRAG example (binary sufficient/insufficient)
  - Google Research "Sufficient Context" autorater (EMNLP 2024 style)

Key differences from the per-chunk grader (grading.py):
  - Holistic view: considers whether chunks *together* contain the answer
  - Uses enriched context (child + parent text) not just truncated child
  - Three-way verdict: SUFFICIENT / PARTIAL / INSUFFICIENT (same action
    semantics but evaluated at context level, not chunk level)
"""

SUFFICIENCY_SYSTEM_PROMPT = """You are a financial analyst evaluating whether retrieved SEC filing context collectively contains sufficient information to answer a specific question.

TASK: Read the question and ALL provided context chunks together, then decide if the information as a whole allows a confident, grounded answer.

VERDICT DEFINITIONS:
- SUFFICIENT: The context clearly contains the data, figures, or facts needed to answer the question completely. A financial analyst could derive a complete answer directly from the provided text.
- PARTIAL: The context contains some relevant information (related figures, discussion of the topic) but is missing one or more key data points needed for a complete answer. An analyst would need additional information.
- INSUFFICIENT: The context clearly lacks any meaningful information to address the question. The retrieved chunks are about unrelated topics or the wrong time period.

IMPORTANT RULES:
- Evaluate holistically: consider whether chunks work TOGETHER to answer the question, not just each chunk individually
- For calculation questions: check if ALL required inputs are present across the chunks (e.g., net income AND total assets for ROA)
- For qualitative questions: check if the discussion or disclosure is present somewhere in the context
- Wrong year or wrong metric = PARTIAL, not SUFFICIENT
- Completely unrelated chunks = INSUFFICIENT

Respond with EXACTLY one word on a single line: SUFFICIENT, PARTIAL, or INSUFFICIENT.
Do not explain. One word only."""

SUFFICIENCY_USER_PROMPT = """Question: {question}

Retrieved Context:
{context}

Verdict:"""


def format_context_for_sufficiency(enriched_contexts: list[dict]) -> str:
    """Format enriched contexts for the sufficiency judgment prompt.

    Uses full child + parent text (unlike per-chunk grader which truncates
    child text to 1500 chars) to give the model the complete picture.

    Args:
        enriched_contexts: List of enriched context dicts with child_text,
            parent_text, and metadata.

    Returns:
        Formatted context string.
    """
    parts = []
    for i, ctx in enumerate(enriched_contexts):
        meta = ctx.get("metadata", {})
        company = meta.get("company_ticker", "Unknown")
        section = meta.get("section_title", "Unknown Section")
        year = meta.get("fiscal_year", "Unknown")

        child = ctx.get("child_text", "") or ctx.get("text", "")
        parent = ctx.get("parent_text", "")

        chunk_text = child
        if parent and parent != child:
            # Include parent context for calculation questions
            chunk_text = f"{child}\n\n[Parent context: {parent[:800]}]"

        parts.append(
            f"--- Chunk {i} [{company} | {section} | FY{year}] ---\n"
            f"{chunk_text[:2000]}"
        )

    return "\n\n".join(parts)
