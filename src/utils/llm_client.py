"""One place that picks the chat model provider, model name and API client.

Every pipeline step and eval script calls ``chat_completion`` instead of
building ``OpenAI()`` and hard-coding a model. The provider comes from
``LLM_PROVIDER`` in ``.env`` (``gemini`` or ``openai``).

Gemini is reached through its OpenAI-compatible endpoint with the existing
``openai`` SDK, so no extra package is needed. Two Gemini differences are
handled here so callers can stay the same:

- ``seed`` is rejected with HTTP 400, so it is dropped.
- Gemini "thinks" by default and the thinking counts against ``max_tokens``;
  a ``max_tokens=10`` call then returns no text. ``reasoning_effort="none"``
  is sent by default to turn thinking off (override with
  ``LLM_REASONING_EFFORT``).

Calls still go through ``openai.resources.chat.completions.Completions.create``,
so the disk cache (``llm_cache``) and stub mode (``llm_stub``) keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from openai import OpenAI

from config.settings import Settings, get_settings

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class _ProviderDefaults:
    base_url: str | None
    model: str | None  # None = use OPENAI_MODEL
    key_env: str
    reasoning_effort: str | None
    supports_seed: bool
    embedding_model: str


_PROVIDERS: dict[str, _ProviderDefaults] = {
    "gemini": _ProviderDefaults(
        base_url=GEMINI_BASE_URL,
        model="gemini-2.5-flash",
        key_env="GEMINI_API_KEY",
        reasoning_effort="none",
        supports_seed=False,
        embedding_model="gemini-embedding-001",
    ),
    "openai": _ProviderDefaults(
        base_url=None,
        model=None,
        key_env="OPENAI_API_KEY",
        reasoning_effort=None,
        supports_seed=True,
        embedding_model="text-embedding-ada-002",
    ),
}


@dataclass(frozen=True)
class LLMConfig:
    """Resolved chat model settings for one provider."""

    provider: str
    model: str
    api_key: str
    base_url: str | None
    reasoning_effort: str | None
    supports_seed: bool
    max_retries: int
    judge_model: str
    judge_embedding_model: str


def get_llm_config(settings: Settings | None = None) -> LLMConfig:
    """Resolve provider, model, key and URL from settings (``.env`` / environment)."""
    s = settings or get_settings()
    provider = s.llm_provider.strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{s.llm_provider}'. Use one of: {', '.join(_PROVIDERS)}"
        )
    d = _PROVIDERS[provider]
    api_key = s.gemini_api_key if provider == "gemini" else s.openai_api_key
    model = s.llm_model or d.model or s.openai_model
    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=s.llm_base_url or d.base_url,
        reasoning_effort=s.llm_reasoning_effort or d.reasoning_effort,
        supports_seed=d.supports_seed,
        max_retries=s.llm_max_retries,
        judge_model=s.judge_model or model,
        judge_embedding_model=s.judge_embedding_model or d.embedding_model,
    )


def get_llm_model() -> str:
    """The chat model name every pipeline step uses."""
    return get_llm_config().model


@lru_cache(maxsize=8)
def _client(api_key: str, base_url: str | None, max_retries: int) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)


def get_llm_client(config: LLMConfig | None = None) -> OpenAI:
    """Shared OpenAI-SDK client for the configured provider."""
    cfg = config or get_llm_config()
    if not cfg.api_key:
        key_env = _PROVIDERS[cfg.provider].key_env
        raise RuntimeError(
            f"LLM_PROVIDER={cfg.provider} but {key_env} is not set (add it to .env)"
        )
    return _client(cfg.api_key, cfg.base_url, cfg.max_retries)


def chat_completion(messages: list[dict[str, Any]], **kwargs: Any) -> tuple[Any, str]:
    """Call the configured chat model; returns ``(response, model_name)``.

    Accepts the usual ``chat.completions.create`` arguments. ``model`` is
    filled in from settings unless given.
    """
    cfg = get_llm_config()
    kwargs.setdefault("model", cfg.model)
    if not cfg.supports_seed:
        kwargs.pop("seed", None)
    if cfg.reasoning_effort and "reasoning_effort" not in kwargs:
        kwargs["reasoning_effort"] = cfg.reasoning_effort
    response = get_llm_client(cfg).chat.completions.create(messages=messages, **kwargs)
    return response, kwargs["model"]


def get_ragas_judge() -> tuple[Any, Any]:
    """LangChain chat model and embeddings for Ragas, on the configured provider.

    Pass them to ``ragas.evaluate(..., llm=llm, embeddings=embeddings)``;
    without them Ragas silently falls back to OpenAI gpt-4o-mini.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    cfg = get_llm_config()
    if not cfg.api_key:
        get_llm_client(cfg)  # raises the same clear error
    llm_kwargs: dict[str, Any] = {
        "model": cfg.judge_model,
        "api_key": cfg.api_key,
        "base_url": cfg.base_url,
        "temperature": 0,
        "max_retries": cfg.max_retries,
    }
    if cfg.reasoning_effort:
        llm_kwargs["reasoning_effort"] = cfg.reasoning_effort
    if cfg.supports_seed:
        llm_kwargs["seed"] = 42
    llm = ChatOpenAI(**llm_kwargs)
    embeddings = OpenAIEmbeddings(
        model=cfg.judge_embedding_model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        # Gemini expects text, not the token-id lists langchain sends by default
        check_embedding_ctx_length=cfg.provider == "openai",
        max_retries=cfg.max_retries,
    )
    return llm, embeddings
