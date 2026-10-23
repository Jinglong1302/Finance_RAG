# tests — rules for this area

- **Every test runs offline**: no API keys, no Qdrant, no model downloads. This is checked by running the suite with `GEMINI_API_KEY` and `OPENAI_API_KEY` unset (`docs/engineering/BASELINE.md`).
- **Fakes, not real services:**
  - LLM: `src/utils/llm_stub.py` (`enable_stub_mode()` / `disable_stub_mode()`), or monkeypatch `openai.resources.chat.completions.Completions.create`.
  - Reranker model: set `CrossEncoderReranker()._model` to an object with `compute_score(pairs)`.
  - Search and expander: small classes with `search()` / `expand()`. Examples: `test_retrieval/test_reranker.py`, `test_orchestration/test_graph_wiring.py`.
- **Isolate settings.** Tests that read settings should clear the LLM env vars with `monkeypatch` and build `Settings(_env_file=None)` so a local `.env` cannot change the result (see `test_utils/test_llm_client.py`).
- **One known network need:** token counting (`src/utils/tokens.py`) downloads tiktoken's `o200k_base` file on first use. On a machine with no internet and no cached copy, 17 chunking/token tests fail (BASELINE.md).
- **A bug fix comes with a test that fails on the old code.** Check that by running the new test against the old code before committing.
- Test folders mirror `src/`: `test_<area>/` with an `__init__.py` (`test_ui/` is still missing one; harmless today).
