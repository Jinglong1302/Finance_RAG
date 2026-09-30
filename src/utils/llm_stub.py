"""Stub-LLM mode for plumbing smoke tests (no API calls, no cost).

Replaces ``Completions.create`` with canned responses that match each
pipeline node's expected output format, so the full CRAG graph can be
exercised end-to-end without any OpenAI calls.

Stub responses by node (detected via system-prompt fingerprint):
- Decomposer  → JSON with sub_queries, structured_filters, query_type, result_count
- Sufficiency → "SUFFICIENT" (one word, triggers generate path)
- Per-chunk grader → "relevant" (triggers generate)
- Generator  → short canned answer
- Guard      → JSON {"result": "pass"}
- Rewriter   → "rewritten query for plumbing test"

Usage
-----
    from src.utils.llm_stub import enable_stub_mode
    enable_stub_mode()          # must be called BEFORE enable_disk_cache()
    # now run any eval script — zero API calls

Stub mode is incompatible with disk cache (they both patch Completions.create).
Call enable_stub_mode() instead of enable_disk_cache() for plumbing tests.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock

log = logging.getLogger(__name__)

_STUB_ACTIVE: bool = False
_ORIG_CREATE: Any = None
_call_log: list[dict] = []


def _detect_node(messages: list[dict]) -> str:
    """Detect which pipeline node is calling based on system prompt content."""
    sys_content = ""
    for m in messages:
        if m.get("role") == "system":
            sys_content = m.get("content", "").lower()
            break
    if "decompose" in sys_content or "sub_queries" in sys_content or "structured_filters" in sys_content:
        return "decomposer"
    # Generator: detected BEFORE sufficiency_grader because generator prompt
    # contains "sufficient information" in rule 3, which would cause a false match.
    # Use the stronger fingerprint "citation format" or "financial analyst assistant".
    if "citation format" in sys_content or (
        "financial analyst assistant" in sys_content and "citation" in sys_content
    ):
        return "generator"
    if "sufficient" in sys_content or "verdict" in sys_content or "partial" in sys_content:
        return "sufficiency_grader"
    if "relevance" in sys_content and "chunk" in sys_content:
        return "grader"
    if "hallucination" in sys_content or "faithfulness" in sys_content or "unsupported" in sys_content:
        return "guard"
    if "rewrite" in sys_content or "reformulate" in sys_content:
        return "rewriter"
    # Fallback: user prompt
    for m in messages:
        if m.get("role") == "user" and "answer the following" in m.get("content","").lower():
            return "generator"
    return "unknown"


def _make_canned_response(node: str, messages: list[dict]) -> "ChatCompletion":
    """Build a fake ChatCompletion object for the given node."""
    import openai.types.chat as chat_types
    import openai.types as ot
    import time

    # Extract query from user message for decomposer
    user_content = ""
    for m in messages:
        if m.get("role") == "user":
            user_content = m.get("content", "")
            break

    if node == "decomposer":
        content = json.dumps({
            "sub_queries": ["stub query for plumbing test"],
            "structured_filters": {},
            "query_type": "factual_numeric",
            "result_count": 5,
        })
        response_format = {"type": "json_object"}
    elif node == "sufficiency_grader":
        content = "SUFFICIENT"
    elif node == "grader":
        content = "relevant"
    elif node == "guard":
        content = json.dumps({"result": "pass", "issues": []})
    elif node == "rewriter":
        content = "rewritten query for plumbing test"
    elif node == "generator":
        content = "STUB ANSWER: This is a plumbing test response with no real content. [1]"
    else:
        content = "STUB RESPONSE"

    # Build minimal ChatCompletion-like object
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = content
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 10
    resp.usage.total_tokens = 110
    resp.usage.prompt_tokens_details = MagicMock()
    resp.usage.prompt_tokens_details.cached_tokens = 0
    resp.usage.completion_tokens_details = MagicMock()
    resp.usage.completion_tokens_details.reasoning_tokens = 0
    resp.model = "stub"
    resp.id = "stub-0000"
    resp.created = int(time.time())
    resp.object = "chat.completion"

    return resp


def enable_stub_mode() -> None:
    """Replace Completions.create with a stub that returns canned responses.

    Idempotent. Must be called before enable_disk_cache() if used together
    (though they are mutually exclusive — don't use both at once).
    """
    global _STUB_ACTIVE, _ORIG_CREATE

    if _STUB_ACTIVE:
        return

    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        log.warning("openai not found; stub mode NOT enabled")
        return

    _ORIG_CREATE = Completions.create

    def _stub_create(self, *args, **kwargs) -> Any:
        messages = kwargs.get("messages", [])
        node = _detect_node(messages)
        resp = _make_canned_response(node, messages)
        _call_log.append({"node": node, "messages_len": len(messages)})
        log.debug("LLM STUB  node=%s call#%d", node, len(_call_log))
        return resp

    Completions.create = _stub_create  # type: ignore[method-assign]
    _STUB_ACTIVE = True
    log.info("LLM stub mode enabled (all LLM calls return canned responses)")


def disable_stub_mode() -> None:
    """Restore the original Completions.create."""
    global _STUB_ACTIVE, _ORIG_CREATE
    if not _STUB_ACTIVE or _ORIG_CREATE is None:
        return
    from openai.resources.chat.completions import Completions
    Completions.create = _ORIG_CREATE  # type: ignore[method-assign]
    _STUB_ACTIVE = False


def stub_call_log() -> list[dict]:
    """Return the log of stub calls made since enable_stub_mode()."""
    return list(_call_log)
