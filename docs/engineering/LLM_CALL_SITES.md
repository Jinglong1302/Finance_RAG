# AI provider and model call sites

Every place that picks or calls an AI model, as of commit `45966d4` (2026-10-23).
Found with `grep` over `src/`, `config/`, `scripts/` and `tests/`, then read by hand.
Line numbers are for that commit. This is the input for phase 2 (choose provider and model from `.env`).

> **Snapshot, now partly out of date.** Since branch `feat/llm-provider-gemini` (`34f5ab7`), every chat call in sections 1–3 goes through `chat_completion()` in `src/utils/llm_client.py`. The model comes from `LLM_PROVIDER` / `LLM_MODEL` (default Gemini `gemini-2.5-flash`), and Ragas gets its judge from `get_ragas_judge()`. Sections 4 (local models) and 6 (unused settings, apart from the model) still hold. The file:line references below are for the old commit.

## Short version

- **All live LLM calls use the raw `openai` SDK** (`OpenAI().chat.completions.create`). The app's own code never imports `langchain_openai`.
  `langchain_openai` is only used *inside Ragas* for the eval judge.
- **The six pipeline nodes hard-code `model="gpt-4o"`.** `OPENAI_MODEL` in `.env` does **not** change the model the app answers with. It only changes a few eval scripts, the naive baseline, and the label shown by the UI health check.
- **The Ragas judge is not gpt-4o.** No eval passes `llm=` to `ragas.evaluate`, so Ragas 0.2.15 falls back to `gpt-4o-mini` plus `text-embedding-ada-002` (checked in the installed package, `ragas/evaluation.py:196-204`, `ragas/llms/base.py:382`, `ragas/embeddings/base.py:356`). `scripts/eval/eval_ragas_phase_c.py:210` writes `"judge_model": "gpt-4o"` into results, which does not match.
- **Two local Hugging Face models** (BGE-M3 embeddings, BGE reranker) are already read from `.env`.
- Three helpers are tied to the OpenAI SDK class itself and will need care when a second provider is added: the disk cache, the stub mode, and the price table.

## 1. Pipeline LLM calls (the real answer path)

All six build a fresh `OpenAI()` client on every call. The client reads `OPENAI_API_KEY` (and `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` if set) from the environment; `config/settings.py:83-84` copies the key from `.env` into `os.environ` when `get_settings()` runs.

| File:line (client / call) | Node | Model | Other settings hard-coded | Cost and trace label |
|---|---|---|---|---|
| `src/orchestration/nodes/query_decomposer.py:47` / `:50-51` | Split query, pull ticker/year filters | `"gpt-4o"` | `response_format={"type":"json_object"}`, `temperature=0`, `seed=42` | `cost_from_usage(..., "gpt-4o")` `:68`; trace `:94` |
| `src/orchestration/nodes/grader.py:72` / `:76-77` | Per-chunk relevance grader (**this is the one the graph uses**, see REVIEW.md item 2) | `"gpt-4o"` | `json_object`, `temperature=0`, `seed=42` | `:96`; trace `:169` |
| `src/orchestration/nodes/sufficiency_grader.py:89` / `:93-94` | Pooled "Part B" grader (used by eval scripts only) | `"gpt-4o"` | `temperature=0`, `seed=42`, `max_tokens=10` | `:110`; trace `:152` |
| `src/orchestration/nodes/generator.py:57` / `:61-62` | Writes the answer with citations | `"gpt-4o"` | `temperature=0`, `seed=42` | `:78`; trace `:94` |
| `src/orchestration/nodes/hallucination_guard.py:52` / `:56-57` | Checks the answer against context | `"gpt-4o"` | `json_object`, `temperature=0`, `seed=42` | `:76`; trace `:89` |
| `src/orchestration/nodes/query_rewriter.py:45` / `:48-49` | Rewrites the query for a retry | `"gpt-4o"` | `temperature=0`, `seed=42` | `estimate_cost(..., "gpt-4o")` `:72` (ignores cached-token discount, unlike the others); trace `:98` |

Other numbers that shape LLM behaviour and are hard-coded, not read from `.env`:

| What | Where | Value | `.env.example` has |
|---|---|---|---|
| Max rewrite cycles | `grader.py:43`, `sufficiency_grader.py:60`, `hallucination_guard.py:106`, `graph.py:74` | `2` | `MAX_CRAG_CYCLES=2` (read into settings, never used by these) |
| Candidates fetched per sub-query | `src/orchestration/nodes/retriever.py:57` | `25` | `RETRIEVAL_TOP_K=25` (never used) |
| Chunks kept after rerank | `reranker_node.py:65` takes `result_count` from the decomposer LLM, default `5` | LLM-chosen | `RERANK_TOP_K=5`, `RERANK_TOP_K_EXPANDED=8` (never used) |

## 2. Other raw `openai` calls (baselines and eval scripts)

| File:line (client / call) | Purpose | Model | Settings |
|---|---|---|---|
| `src/evaluation/baseline.py:57` / `:111-112` | Naive RAG baseline generator | `self.openai_model` (default `"gpt-4o"` at `:44`; `settings.openai_model` passed at `:191`) | `temperature=0`, `seed=42`, `max_tokens=512` |
| `scripts/eval/eval_ablations.py:92` / `:59-60` | Ablation generator | `settings.openai_model` | `temperature=0`, `seed=42` |
| `scripts/eval/eval_part_a_consolidated.py:233` / `:89-90` | Part A generator | `settings.openai_model` | `temperature=0`, `seed=42` |
| `scripts/eval/eval_oracle.py:113` / `:79-80` | Oracle-context generator | `settings.openai_model` | `temperature=0`, `seed=42` |
| `scripts/eval/eval_generation_tatqa.py:219` / `:173-174` | Naive TAT-QA generator | `settings.openai_model` (passed at `:241`) | `temperature=0`, `max_tokens=256`, **no seed** |

So with `OPENAI_MODEL=gpt-4o-mini`, these scripts would compare a mini baseline against a gpt-4o pipeline, and record `"model": "gpt-4o-mini"` in the results JSON (e.g. `eval_part_b.py:392`, `eval_fullscale_baselines.py:286`) even though the pipeline still ran gpt-4o.

## 3. LLM-as-judge (Ragas)

| File:line | Call | Judge LLM | Judge embeddings |
|---|---|---|---|
| `src/evaluation/ragas_eval.py:192` | `evaluate(eval_dataset, metrics=...)` | Ragas default: `llm_factory()` → `langchain_openai.ChatOpenAI(model="gpt-4o-mini")` | `embedding_factory()` → `langchain_openai.OpenAIEmbeddings(model="text-embedding-ada-002")` |
| `scripts/eval/eval_ragas_phase_c.py:126` | same, no `llm=` | same | same |
| `scripts/eval/eval_generation_financebench.py:209` (`_run_ragas`, also used by `eval_fullscale_baselines.py:230`) | same, no `llm=` | same | same |

Metrics used everywhere: `faithfulness`, `context_precision`, `answer_relevancy`, `context_recall`. `answer_relevancy` is the one that needs embeddings.

Not verified: whether the disk cache (below) also caches the Ragas calls. It patches the OpenAI SDK class that `langchain_openai` uses underneath, so it probably does, but I did not run Ragas.

## 4. Local models (Hugging Face, via FlagEmbedding)

| File:line | What | Model | Read from `.env`? |
|---|---|---|---|
| `src/embedding/embedder.py:63-66` | `BGEM3FlagModel` dense + sparse embeddings | default `"BAAI/bge-m3"` (`:47`), `use_fp16=True` (`:48`) | Yes, callers pass `settings.embedding_model`. `USE_FP16` is passed only by `scripts/query.py`, `scripts/query_naive.py`, `scripts/ingest.py`, `scripts/ingest_historical_fb.py`, `src/ui/server.py`, `src/evaluation/baseline.py`; every `scripts/eval/` script leaves the default `True`. |
| `src/retrieval/reranker.py:42-45` | `FlagReranker` cross-encoder | default `"BAAI/bge-reranker-large"` (`:29`), `use_fp16=True` (`:30`) | Model yes (`settings.reranker_model`); `use_fp16` never passed from settings. README says `bge-reranker-base`; code default is `-large`. **Note:** the pipeline never actually calls this model's scoring, see REVIEW.md item 1. |
| `src/utils/tokens.py:51` | `tiktoken.encoding_for_model("gpt-4o")` for chunk sizes and truncation | `gpt-4o` tokenizer (`o200k_base`) | No. Used to size chunks that are embedded by BGE-M3, which has its own tokenizer. |

Callers that build these: `scripts/query.py:67,99`, `src/ui/server.py:49-60`, `scripts/ingest.py:101`, `scripts/ingest_historical_fb.py:118`, `scripts/query_naive.py:109`, `src/evaluation/baseline.py:185`, and every `scripts/eval/eval_*.py` (see `grep -n "BGEEmbedder(\|CrossEncoderReranker(" -r scripts`).

## 5. Helpers tied to the OpenAI SDK

| File:line | What it does | Phase 2 impact |
|---|---|---|
| `src/utils/llm_cache.py:49-90` | Replaces `openai.resources.chat.completions.Completions.create` with a disk cache. Key = model, messages, temperature, seed, response_format (`:59-65`). Not in key: `max_tokens`, base URL. | Works for any provider reached *through the openai SDK*. Add base URL / provider to the key, or two providers with the same model name would share entries. |
| `src/utils/llm_stub.py:136,161` | Stub mode: same patch, canned replies picked by system-prompt keywords (`:37-66`). | Same as above. |
| `src/utils/tokens.py:19-30, 112` | Price table with only `gpt-4o` and `gpt-4o-mini`. Unknown models are **silently priced as gpt-4o** (`:112`). | Gemini or any other model would get wrong costs with no warning. |
| `src/ui/static/app.js:500` | Shows `node.model || "gpt-4o"`. | Cosmetic. |

## 6. Env vars: what `.env.example` lists vs what the code reads

`config/settings.py` (pydantic-settings, names are not case-sensitive) reads every field below. "Used" means some code path changes behaviour because of it.

| Variable | In `.env.example` | Read by settings | Actually used |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | yes | yes (copied to `os.environ`, then read by every `OpenAI()` and by Ragas) |
| `OPENAI_MODEL` | yes ("used for generation, sufficiency grading, and hallucination guard") | yes | **only** baseline / eval-script generators and UI health label. The comment in `.env.example` is wrong: those nodes hard-code gpt-4o. |
| `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` | yes | yes | yes |
| `EMBEDDING_MODEL` | yes | yes | yes |
| `EMBEDDING_DIM` | yes | yes | yes (indexer) |
| `USE_FP16` | yes | yes | partly (see section 4) |
| `RERANKER_MODEL` | yes | yes | yes (but see REVIEW.md item 1) |
| `RETRIEVAL_TOP_K`, `RERANK_TOP_K`, `MAX_CRAG_CYCLES` | yes | yes | **no** — only copied into result JSON by `eval_crag_holdout22.py:356-358` |
| `RERANK_TOP_K_EXPANDED` | yes | yes | **no** |
| `RERANKER_COARSE_THRESHOLD` | yes | yes | yes (grader nodes) — but it never triggers, see REVIEW.md item 1 |
| `PROSE_CHUNK_SIZE`, `PROSE_CHUNK_OVERLAP`, `MAX_TABLE_CHUNK_TOKENS` | yes | yes | yes (ingest scripts) |
| `SEC_USER_AGENT_COMPANY`, `SEC_USER_AGENT_EMAIL` | yes | yes | yes |
| `BUDGET_PER_QUERY`, `BUDGET_PER_EVAL_RUN` | yes | yes | **no** — `CostTracker` is only used by tests |
| `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` | no | no | read directly by the openai SDK if present (`openai/_client.py:133-160` in the installed package) |

No variable today selects a provider. There is no Gemini / Google library installed (`site-packages/google/` only holds `protobuf` and `_upb`).

## 7. Notes for phase 2

These are suggestions to check, not decisions.

1. **One place builds the client and picks the model.** Add something like `src/llm/client.py` with `get_chat_client()` and `model_for(role)`; replace the six `OpenAI()` + `"gpt-4o"` pairs in section 1. Keep `temperature=0, seed=42` there too.
2. **Gemini without a new dependency (unverified here, no network):** Google offers an OpenAI-compatible endpoint, so the existing `openai` SDK can talk to Gemini with `base_url` + a Gemini key. Things to test before trusting it: whether `seed` and `response_format={"type":"json_object"}` are accepted, and whether `usage.prompt_tokens_details.cached_tokens` is filled in. If any of those fail, the native Google SDK would be a **new dependency — ask the owner first.**
3. **Per-role model settings** make sense: decomposer, grader, generator, guard, rewriter, judge. One `LLM_MODEL` default plus optional overrides keeps `.env` short.
4. **Judge settings**: pass `llm=` and `embeddings=` to all three `ragas.evaluate` calls, built from `.env`, so the recorded judge model is the real one. `langchain_openai.ChatOpenAI` accepts `base_url`, so the same OpenAI-compatible route works for the judge.
5. **Costs**: make unknown models log a warning (or cost `None`) instead of using gpt-4o prices.
6. **Cache key**: add provider/base URL so cached gpt-4o answers are never replayed for another provider.
7. **Check against the baseline**: the 96 offline tests and the 59/59 import check must stay green; add offline tests for the client factory (it can be tested with a fake base URL and no network).
