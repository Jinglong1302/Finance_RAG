"""Tests for provider selection in src.utils.llm_client (offline: the SDK call is captured)."""

import pytest
from openai.resources.chat.completions import Completions

from config.settings import Settings
from src.utils import llm_client
from src.utils.llm_client import GEMINI_BASE_URL, chat_completion, get_llm_config

_ENV_VARS = (
    "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_REASONING_EFFORT", "LLM_MAX_RETRIES",
    "GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_MODEL", "JUDGE_MODEL", "JUDGE_EMBEDDING_MODEL",
)


@pytest.fixture
def env(monkeypatch):
    """Clean LLM env vars and no .env file, so only what the test sets is used."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_client, "get_settings", lambda: Settings(_env_file=None))
    llm_client._client.cache_clear()
    return monkeypatch


@pytest.fixture
def captured(monkeypatch):
    """Replace the SDK call; record kwargs and the client's base URL."""
    calls: list[dict] = []

    def fake_create(self, *args, **kwargs):
        calls.append({"base_url": str(self._client.base_url), **kwargs})
        return "response"

    monkeypatch.setattr(Completions, "create", fake_create)
    return calls


class TestGetLLMConfig:
    def test_default_provider_is_gemini(self, env) -> None:
        env.setenv("GEMINI_API_KEY", "g-key")
        cfg = get_llm_config(Settings(_env_file=None))
        assert cfg.provider == "gemini"
        assert cfg.model == "gemini-2.5-flash"
        assert cfg.base_url == GEMINI_BASE_URL
        assert cfg.api_key == "g-key"
        assert cfg.reasoning_effort == "none"
        assert cfg.supports_seed is False
        assert cfg.judge_model == "gemini-2.5-flash"
        assert cfg.judge_embedding_model == "gemini-embedding-001"

    def test_openai_provider_uses_openai_model(self, env) -> None:
        env.setenv("LLM_PROVIDER", "openai")
        env.setenv("OPENAI_API_KEY", "o-key")
        env.setenv("OPENAI_MODEL", "gpt-4o-mini")
        cfg = get_llm_config(Settings(_env_file=None))
        assert cfg.model == "gpt-4o-mini"
        assert cfg.base_url is None
        assert cfg.api_key == "o-key"
        assert cfg.reasoning_effort is None
        assert cfg.supports_seed is True

    def test_overrides(self, env) -> None:
        env.setenv("LLM_PROVIDER", "Gemini")
        env.setenv("LLM_MODEL", "gemini-3.5-flash")
        env.setenv("LLM_REASONING_EFFORT", "low")
        env.setenv("JUDGE_MODEL", "gemini-2.5-flash")
        cfg = get_llm_config(Settings(_env_file=None))
        assert cfg.model == "gemini-3.5-flash"
        assert cfg.reasoning_effort == "low"
        assert cfg.judge_model == "gemini-2.5-flash"

    def test_unknown_provider_raises(self, env) -> None:
        env.setenv("LLM_PROVIDER", "mistral")
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            get_llm_config(Settings(_env_file=None))


class TestChatCompletion:
    def test_gemini_drops_seed_and_turns_off_thinking(self, env, captured) -> None:
        env.setenv("GEMINI_API_KEY", "g-key")
        resp, model = chat_completion(
            messages=[{"role": "user", "content": "hi"}], temperature=0, seed=42, max_tokens=10
        )
        assert resp == "response"
        assert model == "gemini-2.5-flash"
        call = captured[0]
        assert "seed" not in call
        assert call["reasoning_effort"] == "none"
        assert call["model"] == "gemini-2.5-flash"
        assert call["max_tokens"] == 10
        assert call["base_url"] == GEMINI_BASE_URL

    def test_openai_keeps_seed_and_sends_no_reasoning(self, env, captured) -> None:
        env.setenv("LLM_PROVIDER", "openai")
        env.setenv("OPENAI_API_KEY", "o-key")
        _, model = chat_completion(messages=[{"role": "user", "content": "hi"}], seed=42)
        call = captured[0]
        assert model == "gpt-4o"
        assert call["seed"] == 42
        assert "reasoning_effort" not in call
        assert "api.openai.com" in call["base_url"]

    def test_explicit_model_wins(self, env, captured) -> None:
        env.setenv("GEMINI_API_KEY", "g-key")
        _, model = chat_completion(messages=[], model="gemini-3.5-flash")
        assert model == "gemini-3.5-flash"
        assert captured[0]["model"] == "gemini-3.5-flash"

    def test_missing_key_gives_clear_error(self, env, captured) -> None:
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
            chat_completion(messages=[])
        assert captured == []
