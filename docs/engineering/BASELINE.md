# Baseline — how the repo stands on 2026-10-23

Taken on branch `review/engineering-2026-10-23`, starting from commit `1b2664b`.
Every number on this page came from a command run in a cloud VM on that date. The VM had:

- no `.env` file and no `OPENAI_API_KEY`
- no Qdrant running (the Docker CLI was there, but nothing was started)
- outbound network blocked for most hosts (an HTTPS proxy answered `403 Forbidden`)
- Python 3.11.17, 3.12 and 3.13.16 installed; `python3` is 3.13

Rerun the same commands after each later phase and compare against the tables below.

## 1. Install

### What happened

| Step | Result |
|---|---|
| `poetry install` with the default Python 3.13 | **Failed** after 2m23s (exit 1) |
| `poetry env use python3.11` then `poetry install` | **Worked** in 1m49s (exit 0); 225 packages installed plus the project |

Why 3.13 fails: `poetry.lock` pins `tiktoken 0.7.0`. On 3.13 pip had to build it from source, and the build stopped with:

```
error: the configured Python interpreter version (3.13) is newer than PyO3's maximum supported version (3.12)
```

`pyproject.toml` says `python = "^3.11"`, which allows 3.13, so this is a real install bug for anyone whose default Python is 3.13 (see REVIEW.md item 10).

Main installed versions (read from the 3.11 env with `importlib.metadata`):
ragas 0.2.15, openai 1.109.1, langchain-core 0.3.86, langchain-openai 0.2.14, langgraph 0.2.76,
FlagEmbedding 1.4.0, qdrant-client 1.19.0, aiohttp 3.14.3, tiktoken 0.7.0.

## 2. Test suite (`pytest`)

Full run: **113 tests collected, 96 passed, 17 failed, 0 skipped, 0 errors** (8.9 s).

| Folder | Passed | Failed | Skipped | Errors |
|---|---:|---:|---:|---:|
| `tests/test_chunking` | 5 | 11 | 0 | 0 |
| `tests/test_evaluation` | 37 | 0 | 0 | 0 |
| `tests/test_ingestion` | 24 | 0 | 0 | 0 |
| `tests/test_orchestration` | 22 | 0 | 0 | 0 |
| `tests/test_ui` | 4 | 0 | 0 | 0 |
| `tests/test_utils` | 4 | 6 | 0 | 0 |
| **Total** | **96** | **17** | **0** | **0** |

The per-folder numbers come from running each folder on its own; they add up to the full run.

### Which tests need things the VM lacks

**All 17 failures have one cause: network.** None of them need an OpenAI key, Qdrant or Docker.

`src/utils/tokens.py:51` calls `tiktoken.encoding_for_model("gpt-4o")`. The first time this runs on a machine, tiktoken downloads the `o200k_base` token file from the internet. With the network blocked, every test that counts tokens fails with:

```
E   requests.exceptions.ProxyError: HTTPSConnectionPool(host='openaipublic.blob.core.windows.net', port=443):
    Max retries exceeded with url: /encodings/o200k_base.tiktoken
    (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))
```

All 17 tracebacks pass through `src/utils/tokens.py:51` in `get_encoder` (15 via `count_tokens`, 2 via `truncate_to_tokens`). The fallback in `get_encoder` only catches `KeyError`, so a network error is not caught.

The 17 network-dependent tests:

- `tests/test_chunking/test_metadata.py::TestChunkingPipelineOrder::test_interleaved_document_order`
- `tests/test_chunking/test_prose_chunker.py::TestProseChunker::` `test_chunks_prose_section`, `test_detects_note_references`, `test_chunk_ids_are_unique`, `test_respects_target_size`, `test_chunks_div_based_paragraphs_with_header`, `test_filters_running_headers`, `test_checkbox_statements_preserved`
- `tests/test_chunking/test_table_chunker.py::TestTableChunker::` `test_small_table_single_chunk`, `test_large_table_splits_into_children`, `test_table_metadata`
- `tests/test_utils/test_tokens.py::TestCountTokens::` `test_count_simple_text`, `test_count_empty_string`, `test_count_financial_text`, `test_count_table_markdown`
- `tests/test_utils/test_tokens.py::TestTruncateToTokens::` `test_no_truncation_needed`, `test_truncation`

On a machine that can reach `openaipublic.blob.core.windows.net` (or has the file cached in `TIKTOKEN_CACHE_DIR`), these 17 are expected to pass. **Not verified here**: I could not download the file.

**The other 96 tests run fully offline.** No test in the suite calls OpenAI, Qdrant, Hugging Face model downloads or Docker. The UI tests (`tests/test_ui`) start the aiohttp app but only call `/api/health`, `/api/sample-queries`, `/` and an empty `/api/query`, none of which build the pipeline.

What the suite does **not** cover at all (no test file imports these): `src/retrieval/*` (hybrid search, reranker, parent expander), `src/embedding/*`, `src/orchestration/nodes/generator.py`, `sufficiency_grader.py`, `hallucination_guard.py`, `query_decomposer.py`, `query_rewriter.py`, `retriever.py`, `reranker_node.py`, and the compiled graph from `build_crag_graph`. So the 96 passing tests say nothing about the live answer path. See REVIEW.md.

Warnings in the full run: 13 (1 LangChain pending-deprecation warning from langgraph, 12 aiohttp `NotAppKeyWarning` from `src/ui/server.py:38-40`).

## 3. Import check

Every module under `src/` and `config/` imports without error: **59/59 OK** (exit 0).
Importing does not need a key, Qdrant or network (models load lazily).

Other static checks run the same day:

| Check | Result |
|---|---|
| `pytest --collect-only -q` | 113 tests collected, no collection errors |
| `python -m compileall -q scripts src config` | exit 0 (all scripts compile) |
| `ruff check src config scripts tests` (ruff 0.4.10, repo config) | 536 findings; 165 auto-fixable. Biggest: E501 long lines 286, I001 import order 61, F541 48, E402 47, F401 unused imports 43 |
| `mypy src config --ignore-missing-imports` | 41 errors in 17 files (59 files checked) |

Ruff and mypy are not run by any CI today, so these are starting numbers, not failures.

## 4. Things I could not run, and why

| Not run | Why |
|---|---|
| Any eval in `scripts/eval/` | Needs `OPENAI_API_KEY`, a running Qdrant with the 6,971-point index, and model downloads from Hugging Face. None available. |
| `scripts/query.py`, `scripts/serve.py` with a real query | Same reasons. |
| `scripts/ingest.py` | Needs SEC EDGAR network access and model downloads. |
| `docker compose up -d` | Not tried: no network to pull `qdrant/qdrant:latest`, and nothing to index. |
| The 17 tiktoken tests with the file present | Could not download `o200k_base.tiktoken`. |

So the README's evaluation numbers (coverage, precision, Ragas scores) are **not re-checked** by this baseline.

## 5. Commands to rerun this baseline

Run from the repo root.

```bash
# 1. Install (use 3.11; 3.13 fails on tiktoken 0.7.0)
poetry env use python3.11
poetry install

# 2. Full test run with a summary line
poetry run pytest -p no:cacheprovider -q

# 3. Per-folder counts
for d in test_chunking test_evaluation test_ingestion test_orchestration test_ui test_utils; do
  printf "%-20s " "$d"; poetry run pytest -p no:cacheprovider -q "tests/$d" 2>&1 | tail -1
done

# 4. Which failures are the tiktoken download (expect 17 on an offline box, 0 online)
poetry run pytest -p no:cacheprovider -q 2>&1 | grep -c "51: in get_encoder"

# 5. Import check: every module under src/ and config/
poetry run python - <<'EOF'
import importlib, pathlib, sys
sys.path.insert(0, ".")
mods = sorted(".".join(p.with_suffix("").parts).removesuffix(".__init__")
              for d in ("src", "config") for p in pathlib.Path(d).rglob("*.py"))
bad = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f"{m}: {type(e).__name__}: {e}")
print(*bad, f"{len(mods) - len(bad)}/{len(mods)} modules import OK", sep="\n")
sys.exit(bool(bad))
EOF

# 6. Static checks
poetry run pytest -p no:cacheprovider --collect-only -q | tail -1
poetry run python -m compileall -q scripts src config && echo compile OK
poetry run ruff check src config scripts tests --statistics | tail -1
poetry run mypy src config --ignore-missing-imports | tail -1
```

Note for step 4: count the `get_encoder` frame, not the URL. The URL appears twice per failing test (34 lines on 2026-10-23), the frame once (17). If `src/utils/tokens.py` changes, the line number in the pattern may move.
