# Finance RAG — working notes

Question answering over SEC 10-K filings: hybrid search (BGE-M3 + Qdrant) → cross-encoder rerank → Part B sufficiency grader → LLM answer with citations → hallucination guard, wired as a LangGraph graph.

## Where things are

| Path | What | Own rules |
|---|---|---|
| `config/settings.py` | All settings, read from `.env` (template: `.env.example`) | |
| `src/ingestion/`, `src/chunking/`, `src/embedding/` | SEC download → parse → chunk → embed → index | |
| `src/retrieval/` | Hybrid search, reranker, parent expansion | `src/retrieval/CLAUDE.md` |
| `src/orchestration/` | Graph, pipeline steps (`nodes/`), prompts | `src/orchestration/CLAUDE.md` |
| `src/utils/llm_client.py` | The only place that picks the AI provider and model | |
| `src/evaluation/`, `scripts/eval/` | Metrics and eval runners | `scripts/eval/CLAUDE.md` |
| `src/ui/`, `scripts/serve.py` | Local web dashboard | |
| `tests/` | Offline unit tests | `tests/CLAUDE.md` |
| `docs/README.md` | Map of every doc (guides, engineering, history) | |

## Commands

```bash
poetry env use python3.11 && poetry install     # 3.13 fails: locked tiktoken 0.7.0 cannot build
poetry run pytest -q                            # offline, no keys needed
poetry run python scripts/query.py "question"   # needs Qdrant + LLM key + models
```

Full baseline checks (tests, import check, lint): `docs/engineering/BASELINE.md`.

## House rules

- Plain everyday wording in docs. Verify before stating a fact; mark anything unverified and say why.
- Never commit `.env`, API keys, connection strings, `results/` or `data/eval/financebench_in_corpus.jsonl` (CC BY-NC). Run `git status` before every commit.
- Ask the owner before adding or upgrading a dependency.
- Commit message: `type(scope): plain-words summary`, then one verb-led `- ` line per change, a blank line, and `Verified: <checks that actually ran, with counts>`. No Co-Authored-By trailer.
- Never force-push or rebase. No pull requests or GitHub comments without the owner's OK.
- Move files with `git mv`, one area at a time, and rerun the baseline after each move.
- A new doc goes in `docs/guides/` (stays true) or `docs/history/YYYY-MM-DD-name.md` (dated record); add it to `docs/README.md`.
