"""Disk cache for OpenAI LLM chat completions.

Patches ``openai.resources.chat.completions.Completions.create`` at the class
level so every ``OpenAI()`` instance — including those created inside pipeline
nodes — benefits from the cache automatically.

Cache key: SHA-256 of (model, messages, temperature, seed, response_format).
Cache value: serialised ``ChatCompletion`` JSON on disk, one file per key.

Usage
-----
Call ``enable_disk_cache()`` once, before any pipeline code imports or runs:

    from src.utils.llm_cache import enable_disk_cache
    enable_disk_cache()          # defaults to results/llm_cache/
    # … now import / run pipeline nodes

The cache is keyed on the deterministic inputs (temp=0, seed=42) so replays
and improvement-loop reruns of unchanged nodes cost nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ORIG_CREATE: Any = None
_CACHE_DIR: Path = Path("results/llm_cache")
_hits: int = 0
_misses: int = 0


def enable_disk_cache(cache_dir: str | Path = "results/llm_cache") -> None:
    """Monkey-patch ``Completions.create`` to persist results to disk.

    Idempotent: calling it more than once is safe.
    """
    global _ORIG_CREATE, _CACHE_DIR

    if _ORIG_CREATE is not None:
        return  # Already patched

    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        log.warning("openai package not found; LLM disk cache NOT enabled")
        return

    _CACHE_DIR = Path(cache_dir)
    _ORIG_CREATE = Completions.create

    def _cached_create(self, *args, **kwargs):  # type: ignore[override]
        global _hits, _misses
        key_parts: dict[str, Any] = {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "temperature": kwargs.get("temperature"),
            "seed": kwargs.get("seed"),
            "response_format": kwargs.get("response_format"),
        }
        raw = json.dumps(key_parts, sort_keys=True, default=str, ensure_ascii=False)
        key = hashlib.sha256(raw.encode()).hexdigest()

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _CACHE_DIR / f"{key}.json"

        if cache_file.exists():
            _hits += 1
            log.debug("LLM cache HIT  %s (total hits=%d)", key[:12], _hits)
            import openai
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return openai.types.chat.ChatCompletion.model_validate(data)

        _misses += 1
        log.debug("LLM cache MISS %s (total misses=%d)", key[:12], _misses)
        resp = _ORIG_CREATE(self, *args, **kwargs)

        try:
            cache_file.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("Failed to write LLM cache entry %s: %s", key[:12], exc)

        return resp

    Completions.create = _cached_create  # type: ignore[method-assign]
    log.info("LLM disk cache enabled → %s", _CACHE_DIR.resolve())


def cache_stats() -> dict[str, int]:
    """Return hit/miss counts since cache was enabled."""
    return {"hits": _hits, "misses": _misses, "total": _hits + _misses}
