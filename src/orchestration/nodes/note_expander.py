"""Footnote context expansion node.

Conditional node that fetches referenced Notes from Qdrant when
reranked chunks contain footnote references in their metadata.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.orchestration.state import CRAGState
from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_note_expander_node(client: QdrantClient, collection_name: str):
    """Factory to create a note expander node.

    Args:
        client: QdrantClient instance.
        collection_name: Qdrant collection name.

    Returns:
        A node function compatible with LangGraph.
    """

    def note_expander_node(state: CRAGState) -> dict[str, Any]:
        """Expand enriched contexts with referenced footnote content.

        Scans enriched_contexts for referenced_notes metadata. For each
        referenced Note, fetches its content from Qdrant by note_id
        and appends it to the context.

        Args:
            state: Current CRAG pipeline state.

        Returns:
            State update with expanded enriched_contexts.
        """
        enriched = state.get("enriched_contexts", [])
        if not enriched:
            return {}

        # Collect all unique note references
        all_note_refs: set[str] = set()
        for ctx in enriched:
            refs = ctx.get("metadata", {}).get("referenced_notes", [])
            all_note_refs.update(refs)

        if not all_note_refs:
            logger.debug("No note references found, skipping expansion")
            return {}

        logger.info(f"Expanding {len(all_note_refs)} note references: {all_note_refs}")

        # Fetch each referenced note
        note_texts: dict[str, str] = {}
        for note_ref in all_note_refs:
            text = _fetch_note(client, collection_name, note_ref)
            if text:
                note_texts[note_ref] = text

        if not note_texts:
            logger.debug("No note content found")
            return {}

        # Append note context to enriched contexts
        updated_enriched = list(enriched)
        for note_ref, note_text in note_texts.items():
            # Add as an additional context entry
            updated_enriched.append({
                "child_text": f"[Appended Context: {note_ref}]\n\n{note_text}",
                "parent_text": None,
                "metadata": {
                    "content_type": "note",
                    "note_id": note_ref,
                    "section_title": f"Appended: {note_ref}",
                },
                "rerank_score": 0.0,
                "chunk_id": f"expanded_{note_ref}",
            })

        logger.info(
            f"Added {len(note_texts)} note expansions to context"
        )

        return {"enriched_contexts": updated_enriched}

    return note_expander_node


def _fetch_note(
    client: QdrantClient,
    collection_name: str,
    note_ref: str,
) -> str | None:
    """Fetch a Note chunk from Qdrant by note_id.

    Args:
        client: QdrantClient instance.
        collection_name: Collection name.
        note_ref: Note reference string (e.g., "Note 12").

    Returns:
        Note text content or None if not found.
    """
    try:
        results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="content_type",
                        match=MatchValue(value="note"),
                    ),
                    FieldCondition(
                        key="note_id",
                        match=MatchValue(value=note_ref),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if results:
            text = results[0].payload.get("text", "")
            # Truncate very long notes to save tokens
            if len(text) > 3000:
                text = text[:3000] + "\n... [Note truncated]"
            return text
        return None

    except Exception as e:
        logger.warning(f"Failed to fetch {note_ref}: {e}")
        return None
