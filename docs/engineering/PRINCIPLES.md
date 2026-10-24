# Engineering principles for Finance RAG

Short rules that fit this repo: a question-answering pipeline over SEC filings, where the numbers in the README are the product. Each rule has why it matters here and how to check it. Item numbers like "REVIEW 1" point to `REVIEW.md`.

### 1. The README describes what the code actually runs

- **Why:** the README's eval tables are the main claim of the project. Today it describes a cross-encoder and a pooled grader that the app does not use (REVIEW 1, 2).
- **Check:** when a PR changes a node, the graph wiring or a model, ask: "Which README line does this change?" Quick look: `grep -n 'add_node(' src/orchestration/graph.py` against the README "Architecture" block.

### 2. Evals run the same graph as the app

- **Why:** hand-written node loops in eval scripts drift away from `build_crag_graph`, and then the published numbers measure something users never get.
- **Check:** `grep -L "build_crag_graph" scripts/eval/eval_*.py` lists scripts that build their own pipeline; each one needs a stated reason (for example "ablation: no grader").

### 3. Every setting in `.env.example` changes something, or it is removed

- **Why:** six settings today are read but ignored, so changing them does nothing and results files record values that were never used (REVIEW 8).
- **Check:** for each field in `config/settings.py`, `grep -rn "settings\.<field>" src scripts` must show a use outside result-metadata dicts.

### 4. One place chooses the AI provider and model

- **Why:** phase 2 needs OpenAI/Gemini switching from `.env`; six nodes hard-coding `"gpt-4o"` make that six edits and six chances to miss one.
- **Check:** `grep -rn '"gpt-4o' src/orchestration` should return nothing once phase 2 is done; the only model names live in settings and the price table.

### 5. An error is never shown or scored as an answer

- **Why:** a failed API call currently becomes the text "Error generating answer: …" with a confidence level, and a failed guard counts as passed (REVIEW 6). For financial answers, refusing is safer than guessing.
- **Check:** an offline test that makes the stubbed LLM raise and asserts `is_abstention` is true and `error` is set. Review question: "What does this node return when the call fails?"

### 6. LLM output is untrusted input

- **Why:** the decomposer's JSON goes straight into Qdrant filters and `int()` (REVIEW 7); the answer goes straight into `innerHTML` (REVIEW 3).
- **Check:** every `json.loads(response…)` is followed by validation (a pydantic model or explicit type checks); `grep -n "innerHTML" src/ui/static/app.js` shows only escaped or cleaned values.

### 7. The answer path is tested offline

- **Why:** none of retrieval, reranking, grading, generation or the graph has a test today, and that is where the two biggest bugs are (REVIEW 13). Fakes are cheap: `src/utils/llm_stub.py` already stubs the LLM.
- **Check:** with no key, no Qdrant and no network, `poetry run pytest -q` passes; BASELINE.md lists any test that still needs network and why. New node code comes with a test using a fake model, searcher or client.

### 8. Results record what really ran

- **Why:** comparisons between SHAs only mean something if each result says exactly which code, models and judge produced it. Today the judge is recorded as gpt-4o while Ragas used gpt-4o-mini (REVIEW 9).
- **Check:** each results JSON has git SHA, dirty flag, and model names taken from the objects that made the calls (client, reranker, Ragas `llm`), not copied from settings. Review question: "Could this field be wrong if `.env` changed?"

### 9. LLM calls are repeatable and the cache key covers everything that changes the answer

- **Why:** the whole eval method relies on `temperature=0`, `seed=42` and the disk cache. A key that misses a parameter (today: `max_tokens`, base URL) can replay an answer from a different setup.
- **Check:** a unit test that two calls differing only in provider/base URL or `max_tokens` get different cache keys (`src/utils/llm_cache.py:59-65`).

### 10. No secrets or licence-restricted data in git, ever

- **Why:** the repo is public. The FinanceBench file is still in history (REVIEW 5); keys never have been, and should stay that way.
- **Check:** `git status` before every commit; `.env`, `results/` and `data/eval/financebench_in_corpus.jsonl` must not appear. Occasionally: `git log --all -p | grep -nE "sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}"` returns only the `.env.example` placeholder.

### 11. Every import is a declared dependency, and new ones are asked for first

- **Why:** `aiohttp` works only because another package pulls it in (REVIEW 12); one upstream change and the UI stops installing. New packages add install weight (torch is already heavy) and licence questions.
- **Check:** a review question on any new `import`: "Is this in `pyproject.toml`, and did the owner OK it?" `poetry install` on a fresh Python 3.11 env, then the import check in BASELINE.md (59/59 today).

### 12. Change structure one area at a time, and rerun the baseline after each step

- **Why:** 21 files patch `sys.path` and 11 scripts import other scripts; a careless move breaks imports that no test covers.
- **Check:** after each `git mv`, run the full BASELINE.md command list and compare: same 113 tests collected, same pass count, import check still all OK, `compileall` exit 0.
