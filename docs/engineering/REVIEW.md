# Engineering review — 2026-10-23

Read-only review of commit `1b2664b`. Nothing in `src/`, `tests/`, `scripts/` or config was changed.

How each item was checked is marked:

- **[ran]** — I reproduced it with a command (scripts kept outside the repo; the commands are given so you can rerun them).
- **[read]** — found by reading the code; not run, usually because it needs an OpenAI key, Qdrant or network.

Effort: **S** = under an hour, **M** = half a day to a day, **L** = more. "Moves files" = needs `git mv`, so it belongs to phase 3.

## Status (updated 2026-10-23, after phases 2 and 3 started)

| Item | Status | Where |
|---|---|---|
| 1. Reranker never runs | **Fixed**, with tests that fail on the old code | branch `fix/production-grader-and-reranker` (`345bf10`) |
| 2. Wrong grader in production | **Fixed**: Part B wired into the graph, as the owner decided | same commit |
| 8. Settings that do nothing | **Partly**: the model now comes from `LLM_PROVIDER` / `LLM_MODEL`. `RETRIEVAL_TOP_K`, `RERANK_TOP_K*`, `MAX_CRAG_CYCLES` and `BUDGET_*` still do nothing | branch `feat/llm-provider-gemini` (`4423984`) |
| 9. Ragas judge mismatch | **Fixed**: all three `ragas.evaluate` calls pass the configured judge and embeddings | same commit |
| 17. New client per call | **Fixed**: one shared client in `src/utils/llm_client.py` | same commit |
| 21. Docs drift | **Partly**: README tree, docs map and history labels fixed in `docs/organize-structure`; the reranker-name question is answered (owner's `.env` uses `bge-reranker-base`) | branch `docs/organize-structure` |
| 19. Eval scripts import each other | **Fixed**: shared code moved to `src/evaluation/eval_data.py` and `eval_helpers.py`; script-to-script imports 11 → 0. The runners still copy the graph steps instead of calling `build_crag_graph` | branch `refactor/eval-structure` |
| Structure: one-off scripts | **Done**: `eval_ablations`, `eval_oracle`, `run_all_evals`, `generate_3_way_report` moved to `scripts/eval/archive/` | same branch |
| All others | Open | — |

New finding while running evals: the Gemini free tier allows **20 requests per day per model**, too few for any eval run (see `docs/guides/evaluation.md`). Errors from that limit become scored "answers" (item 6), so item 6 matters more now.

## Fix first

### 1. The cross-encoder reranker never runs — **bug, high** [ran]

- **Where:** `src/orchestration/nodes/reranker_node.py:70` calls `reranker.rerank(query, candidates, top_k=len(candidates))`. `src/retrieval/reranker.py:69` returns early with `if len(candidates) <= top_k: return candidates`. Since `top_k` is always `len(candidates)`, the model is never asked for scores.
- **What follows from it:**
  - Results keep their hybrid RRF scores, which are small positive numbers (at most about 0.03 with the weights in `hybrid_search.py:200-219`). The "Stage 1 coarse gate" (`RERANKER_COARSE_THRESHOLD=-2.0`, used in `grader.py:53` and `sufficiency_grader.py:69`) can therefore only fire when there are no results at all.
  - The table-diversity check `c.score > -3.0` (`reranker_node.py:78`) is always true.
  - Order after "reranking" is just the hybrid search order.
  - This line came in with commit `cbac640` (2026-09-28). Every eval script uses `make_reranker_node`, so the v1.0 numbers in README and `results/FINAL.md` were measured **without** the cross-encoder, while the README describes it as part of the pipeline.
- **Proof:** with a fake model that counts calls, the node made **0** `compute_score` calls for 25 candidates and kept scores `[0.03, 0.029, …]`; calling `rerank(..., top_k=5)` directly made 1 call. Script: see "Rerun the checks" at the bottom.
- **Fix:** in `rerank`, return early only when the list is empty (or skip scoring only when `len(candidates) <= 1`). Add an offline test using a fake model, like the check script. Then re-run the evals, because the coarse gate and the chunk order will change. Expect different coverage/precision.
- **Effort:** S for the code and test; the eval rerun needs your key and Qdrant. **Moves files:** no.

### 2. The app runs a different grader than the README calls "production" — **bug / decision, high** [read]

- **Where:** `src/orchestration/graph.py:16` and `:190` wire `grader_node` (the older per-chunk grader). The pooled "Part B" `sufficiency_grader_node` is used only by hand-written loops in eval scripts (`scripts/eval/eval_fb27_no_injection.py:94-114`, `eval_part_b.py`, `eval_part_b_aapl50.py`, `smoke_test_stub.py:74-88`).
- **Why it matters:** `scripts/query.py`, the web UI (`src/ui/server.py:65`) and `eval_fullscale_baselines.py` (which the README calls "production graph") all run the per-chunk grader. README "v1.0 — Part B pooled-context sufficiency grader, production path" does not describe what the app does.
- **Decision (owner, 2026-10-23):** production uses the Part B `sufficiency_grader_node`.
- **Fix:** wire `sufficiency_grader_node` in `graph.py:16` and `:190`, or or add a `grader=` argument to `build_crag_graph` and make the eval scripts use the graph instead of their own loops (see item 19). Add a test that inspects the compiled graph's `grade` node.
- **Effort:** S once decided. **Moves files:** no.

### 3. Web UI renders model output as raw HTML — **security, medium** [read]

- **Where:** `src/ui/static/app.js:301-302` puts `marked.parse(answerText)` into `innerHTML` with no cleaning. `src/ui/static/index.html:11` loads `marked` from jsDelivr with **no version and no integrity hash**.
- **Why it matters:** the answer text comes from the LLM, which quotes SEC filing text and the user's question. Any `<img onerror=…>` or `<script>`-like HTML in it runs in the page. And whatever jsDelivr serves as "latest marked" runs in the page too.
- **Fix:** pin `marked` to an exact version with an `integrity=` hash, and clean the HTML before inserting it (DOMPurify is the usual choice — **that is a new dependency, so ask first**; the no-dependency option is to escape HTML before `marked.parse`).
- **Effort:** S. **Moves files:** no.

### 4. Any website can make the local UI spend OpenAI money — **security, medium** [ran, partly]

- **Where:** `src/ui/server.py:143-152` (`/api/query`) and `:110` (`/api/reload`). No auth, no Origin check, and `request.json()` parses any body.
- **Proof:** a POST with `Content-Type: text/plain` and `Origin: https://evil.example` was parsed and answered `400 {"error": "Query cannot be empty"}`. A browser sends `text/plain` cross-site without asking first, so a page you visit could fire real queries at `127.0.0.1:8000`. (It cannot read the reply. I did not send a non-empty query, since that would build the real pipeline.)
- **Fix:** reject requests whose `Content-Type` is not `application/json`, and reject `Origin` headers that are not the UI's own. Optionally a token in `.env`.
- **Effort:** S. **Moves files:** no.

### 5. A non-commercial dataset is still in the public git history — **licence, decision** [ran]

- **Where:** `data/eval/financebench_in_corpus.jsonl` was added in `e9de99f` (2026-09-26) and deleted in `1b2664b`. The repo is public (checked with the GitHub repo listing). Anyone can still get the file from history.
- **Why it matters:** README says the CC BY-NC 4.0 licence "prohibits commercial redistribution". A public repo with the file in history is still distributing it. Whether that is a problem for non-commercial use is your call, not mine.
- **Fix:** your choice: (a) accept it, since CC BY-NC allows non-commercial sharing with attribution — then make the README wording match; or (b) rewrite history. (b) needs a force-push, which your rules forbid without your OK.
- **Effort:** S for (a), M for (b). **Moves files:** no.

Secrets: I searched all 51 commits for OpenAI (`sk-…`), Google (`AIza…`) and Hugging Face (`hf_…`) key patterns and for any committed `.env`. **None found**; the only hit is the placeholder in `.env.example`. `.env` is in `.gitignore`. [ran]

## Reliability and correctness

### 6. Errors turn into answers — **bug, medium** [read]

- **Where:**
  - `src/orchestration/nodes/generator.py:116-122`: on any exception the "answer" becomes `"Error generating answer: <exception text>"`.
  - `src/orchestration/nodes/hallucination_guard.py:127-133` and `graph.py:72`: if the guard itself fails, the answer passes as if checked.
  - `src/orchestration/nodes/sufficiency_grader.py:126-128`: if the grader fails, the verdict becomes `PARTIAL`, which means "generate".
- **Why it matters:** an expired key, a rate limit or a timeout gives the user an error string presented as the answer, with `confidence` still set. `_is_refusal` (`src/evaluation/ragas_eval.py:84-103`) does not match "error", so evals count it as answered-and-wrong rather than as a failure.
- **Fix:** on an LLM error, set `error`, mark the result as abstained with a clear reason, and route to `refuse`. In evals, count errors separately.
- **Effort:** M. **Moves files:** no.

### 7. LLM-written filters are trusted without checking — **bug, medium** [read]

- **Where:** `query_decomposer.py:283-287` passes the model's `structured_filters` straight on. `src/retrieval/hybrid_search.py:328` does `int(value)` on `fiscal_year`, and `:303-316` turns `company_ticker` into `str(value)`.
- **Why it matters:** `"fiscal_year": "FY2024"` raises `ValueError`; `retriever.py` has no `try`, so the whole graph fails (UI shows 500). `"company_ticker": ["AAPL", "MSFT"]` becomes the string `"['AAPL', 'MSFT']"`, which matches nothing, so a comparison question silently retrieves nothing for that filter.
- **Fix:** validate the decomposer output with a small pydantic model (ticker: str or list of str; year: int or list of int); drop bad values with a log line. Offline tests with hand-written decomposer outputs.
- **Effort:** S–M. **Moves files:** no.

### 8. Settings in `.env` that do nothing — **reliability, medium** [read]

- **Where:** see `LLM_CALL_SITES.md` section 6. `RETRIEVAL_TOP_K`, `RERANK_TOP_K`, `RERANK_TOP_K_EXPANDED`, `MAX_CRAG_CYCLES`, `BUDGET_PER_QUERY`, `BUDGET_PER_EVAL_RUN` are read but never change behaviour. `OPENAI_MODEL` only changes baselines. The max cycle count `2` is written in four places (`grader.py:43`, `sufficiency_grader.py:60`, `hallucination_guard.py:106`, `graph.py:74`). `CostTracker` (`src/evaluation/cost_tracker.py`) is only used by tests, so there is no spending cap at all.
- **Why it matters:** changing `.env` and seeing no change is confusing, and the eval result JSON records these values as if they were used (`eval_crag_holdout22.py:356-358`).
- **Fix:** read them from settings where the hard-coded values are, or remove them from `.env.example`. Phase 2 touches the same files.
- **Effort:** M. **Moves files:** no.

### 9. Ragas judge is gpt-4o-mini, results say gpt-4o — **eval correctness, medium** [ran: read installed package]

- **Where:** `src/evaluation/ragas_eval.py:192`, `scripts/eval/eval_ragas_phase_c.py:126`, `scripts/eval/eval_generation_financebench.py:209` call `ragas.evaluate` with no `llm=`; Ragas 0.2.15 then uses `gpt-4o-mini` and `text-embedding-ada-002`. `eval_ragas_phase_c.py:210` writes `"judge_model": "gpt-4o"`; README says "Judge model: gpt-4o".
- **Fix:** pass `llm=` and `embeddings=` explicitly (phase 2), and correct the README.
- **Effort:** S. **Moves files:** no.

### 10. `poetry install` fails on Python 3.13 — **install, medium** [ran]

- **Where:** `pyproject.toml` `python = "^3.11"` allows 3.13; `poetry.lock` pins `tiktoken 0.7.0`, which cannot build on 3.13 (`PyO3's maximum supported version (3.12)`). Details in BASELINE.md.
- **Fix:** either `python = ">=3.11,<3.13"` (no new package) or update tiktoken in the lock (changes a dependency version — ask). Add the Python version to README "Prerequisites".
- **Effort:** S. **Moves files:** no.

### 11. Token counting needs the internet — **reliability, medium** [ran]

- **Where:** `src/utils/tokens.py:51`. `tiktoken.encoding_for_model("gpt-4o")` downloads `o200k_base` on first use; the fallback only catches `KeyError` (and its fallback, `cl100k_base`, also needs a download). This is why 17 tests fail offline.
- **Also:** chunks are sized with the GPT-4o tokenizer but embedded by BGE-M3, which has a different tokenizer, so `PROSE_CHUNK_SIZE=512` is not 512 BGE tokens. [read]
- **Fix:** for tests, set `TIKTOKEN_CACHE_DIR` to a cached copy, or count tokens with the embedder's own tokenizer (already installed with FlagEmbedding). Mark tests that truly need network with a pytest marker so the offline count is clear.
- **Effort:** M. **Moves files:** no.

### 12. `aiohttp` is used but not declared — **install, low** [ran]

- **Where:** `src/ui/server.py:15-16` imports `aiohttp`; it is not in `pyproject.toml`. It installs today only because other packages pull it in (aiohttp 3.14.3 in the lock).
- **Fix:** add it to `[tool.poetry.dependencies]`. Not a new package, but it is a dependency change — ask first.
- **Effort:** S. **Moves files:** no.

## Test gaps on risky logic

### 13. The answer path has almost no tests — **test gap, high** [ran]

- No test imports `src/retrieval/*`, `src/embedding/*`, `reranker_node.py`, `retriever.py`, `sufficiency_grader.py`, `generator.py`, `hallucination_guard.py`, `query_decomposer.py`, `query_rewriter.py`, or calls `build_crag_graph`. Items 1, 2, 6 and 7 would each have been caught by a small offline test.
- **Fix:** offline tests with fakes (fake reranker model, fake `HybridSearcher`, stubbed `Completions.create` — `src/utils/llm_stub.py` already does the last one):
  1. `reranker_node` calls the model and sorts by its scores (item 1).
  2. `build_crag_graph` uses the chosen grader (item 2).
  3. `_build_qdrant_filter` with good and bad filter values (item 7).
  4. `sufficiency_grader_node` decisions for each verdict and for an exception.
  5. RRF fusion order in `HybridSearcher` with a fake Qdrant client.
  6. One full graph run in stub mode with a fake searcher.
- **Effort:** M. **Moves files:** no.

## Security, lower priority

### 14. The "sandboxed" calculator is not a sandbox — **security, low (unused today)** [ran]

- **Where:** `src/orchestration/nodes/calculator.py:19-31, 51` runs `eval()` with empty builtins. That does not stop escapes: the same globals let `().__class__.__base__.__subclasses__()` reach `BuiltinImporter` (checked), from which `os` can be loaded. `pow(10, 10**10)` would also hang it.
- **Why it matters:** nothing calls it today (README says so too). If it is ever wired to LLM output, that becomes code execution from text in a filing.
- **Fix:** delete it, or rewrite it with `ast` that only allows numbers and arithmetic operators. Its tests are in `tests/test_orchestration/test_calculator_and_grading.py`.
- **Effort:** S. **Moves files:** no.

### 15. Qdrant container is open and unpinned — **security, low** [read]

- **Where:** `docker-compose.yml:3-6`: `qdrant/qdrant:latest`, ports `6333`/`6334` published on all network interfaces, no API key.
- **Why it matters:** on a laptop on shared Wi-Fi, anyone on the network can read or delete the index. `latest` can change under you.
- **Fix:** pin a version; publish as `127.0.0.1:6333:6333`.
- **Effort:** S. **Moves files:** no.

## Performance

### 16. Reranker score cache grows without limit — **performance, low** [read]

- **Where:** `src/retrieval/reranker.py:35, 79-93`. One entry per (query, chunk) forever. Harmless in a script; in the long-running UI server it slowly eats memory. Only matters once item 1 is fixed.
- **Fix:** `functools.lru_cache`-style size limit, or a plain dict with a max size.
- **Effort:** S. **Moves files:** no.

### 17. New HTTP client and settings load on every node call — **performance, low** [read]

- **Where:** each node in `LLM_CALL_SITES.md` section 1 creates `OpenAI()` per call; `grader.py:47-48` and `sufficiency_grader.py:63-64` call `get_settings()` (re-reading `.env`) per call.
- **Why it matters:** a few milliseconds and a new connection each time; up to about 8 LLM calls per question. Small next to GPT-4o latency.
- **Fix:** one shared client from the phase 2 factory.
- **Effort:** S (comes free with phase 2). **Moves files:** no.

## Dead code and duplication

### 18. Two refusal detectors, and private helpers used across scripts — **duplication, low** [read]

- **Where:** `_REFUSAL_PHRASES` and `_is_refusal` exist in `src/evaluation/ragas_eval.py:84-103` and again in `scripts/eval/eval_abstention.py:47-100`, with different lists. The "private" `_is_refusal`, `_clean_answer_for_ragas`, `_format_contexts_for_ragas` are imported by 9 scripts.
- **Fix:** one public `is_refusal()` in `src/evaluation/`; use it everywhere.
- **Effort:** S. **Moves files:** no.

### 19. Eval scripts copy the graph instead of running it — **structure, medium** [read]

- **Where:** `eval_fb27_no_injection.py:94-114`, `smoke_test_stub.py:74-88` and the Part B scripts loop over nodes by hand. 11 scripts import from other scripts (`scripts.eval.eval_retrieval` alone is imported by 7), and 21 files do `sys.path.insert` to make that work.
- **Why it matters:** the evals and the app can drift apart (item 2 is exactly that).
- **Fix:** let `build_crag_graph` take the grader as an argument and have evals call the graph. Move the shared eval helpers (loaders, `_run_crag`, `_run_ragas`, `get_git_info`) from `scripts/eval/` into `src/evaluation/` so scripts stop importing scripts.
- **Effort:** M. **Moves files:** yes, the helper move is phase 3.

### 20. Unused variables and lint debt — **cleanup, low** [ran]

- `ruff`: 536 findings (165 auto-fixable). Real dead code among them (F841): `graph.py:91`, `reporter.py:40`, `numeric_eval.py:182`, `downloader.py:318`, `llm_stub.py:79,89`, `eval_part_b.py:569`, `eval_ragas_phase_c.py:176`. F811 re-imports in `eval_retrieval.py:149,251`. `mypy`: 41 errors in 17 files. Nothing runs these automatically.
- **Fix:** one `ruff check --fix` commit on its own (so the diff is easy to review), then a CI job. Do it **after** phase 3 moves, or the moves become hard to review.
- **Effort:** S–M. **Moves files:** no.

## Docs that do not match the code

### 21. README and `.env.example` drift — **docs, low** [read]

- README says the reranker is `bge-reranker-base` (`README.md:9, 91`); the code default is `bge-reranker-large` (`config/settings.py:35`, `reranker.py:29`). Which one ran for the published numbers depends on your `.env` — I cannot tell. (And per item 1, neither scored anything.)
- README "Project Structure" lists `notebooks/`, which does not exist.
- README "Key limitations" says FB27 coverage is 17/27 = 62.96%; the canonical row above it says 16/27 = 59.3%.
- `.env.example` says `OPENAI_MODEL` controls generation, grading and the guard; it does not (item 8).
- README does not mention the UI dependency on aiohttp or the Python upper limit (items 10, 12).
- **Effort:** S. **Moves files:** no.

## Structure

Checked [ran]: **no circular imports** inside `src/` and `config/` (59 modules, import graph walked with `ast`), and no lower layer (`ingestion`, `chunking`, `embedding`, `retrieval`, `utils`) imports a higher one (`orchestration`, `evaluation`, `ui`). The `src/` package layout is sound; phase 3 should not move `src/` folders around for their own sake.

Real structure problems, for phase 3:

| Problem | Where | Suggestion | Moves files |
|---|---|---|---|
| Shared eval code lives in scripts | `scripts/eval/eval_retrieval.py`, `eval_generation_financebench.py`, `eval_abstention.py` imported by other scripts | Move helpers into `src/evaluation/` (item 19) | yes |
| Large script files | `scripts/eval/run_all_evals.py` 692 lines, `eval_part_b.py` 589, `eval_part_a_consolidated.py` 506; `src/ui/static/app.js` 1,027 | Split only when touched for another reason; size alone is not a bug | maybe |
| Many one-off eval scripts for past experiments | 16 files in `scripts/eval/` | Keep the ones the README tells people to run; move the rest to `scripts/eval/archive/` | yes |
| Unused module kept "for later" | `src/orchestration/nodes/calculator.py` | Delete or fix (item 14) | no |
| `tests/test_ui/` has no `__init__.py` while the other test folders do | `tests/test_ui/` | Add one for consistency (harmless today: all 113 tests collect) | no |
| Prompts live under `orchestration/prompts/` next to nodes | — | Fine as is | no |

## Rerun the checks

```bash
# Item 1: does the reranker node call the model? (expects "compute_score calls: 0" today, >0 after a fix)
poetry run python - <<'EOF'
import sys; sys.path.insert(0, ".")
from types import SimpleNamespace as NS
from src.retrieval.reranker import CrossEncoderReranker
from src.orchestration.nodes.reranker_node import make_reranker_node
class Fake:
    calls = 0
    def compute_score(self, pairs):
        Fake.calls += 1; return [-5.0 - i for i in range(len(pairs))]
class Exp:
    def expand(self, rs): return [NS(child_text=r.text, parent_text="", metadata=r.metadata, rerank_score=r.score, chunk_id=r.chunk_id) for r in rs]
rr = CrossEncoderReranker(); rr._model = Fake()
res = [{"chunk_id": f"c{i}", "text": f"t{i}", "metadata": {}, "score": 0.03 - i/1000} for i in range(25)]
out = make_reranker_node(rr, Exp())({"search_results": res, "original_query": "q", "result_count": 5})
print("compute_score calls:", Fake.calls, "scores:", [round(r["score"], 3) for r in out["reranked_results"]])
EOF

# Item 2: which grader does the graph use?
grep -n 'add_node("grade"' src/orchestration/graph.py

# Item 4: cross-site text/plain body is parsed (expects 400 "Query cannot be empty")
poetry run python - <<'EOF'
import asyncio, sys; sys.path.insert(0, ".")
from aiohttp.test_utils import TestClient, TestServer
from src.ui.server import create_app
async def main():
    async with TestClient(TestServer(create_app())) as c:
        r = await c.post("/api/query", data='{"query": ""}', headers={"Content-Type": "text/plain", "Origin": "https://evil.example"})
        print(r.status, await r.json())
asyncio.run(main())
EOF

# Item 5: dataset still in history
git log --all --format="%h %ad %s" --date=short -- data/eval/financebench_in_corpus.jsonl

# Secrets scan over all history (expects only the .env.example placeholder)
git log --all -p | grep -nE "sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{30,}"

# Item 14: sandbox escape reachable (expects ['BuiltinImporter'])
python3 -I -c "import math; g={'__builtins__':{},'math':math}; print(eval(\"[c.__name__ for c in ().__class__.__base__.__subclasses__() if c.__name__=='BuiltinImporter']\", g, {}))"
```
