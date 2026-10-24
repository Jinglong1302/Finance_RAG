# scripts/eval — rules for this area

How to run the evals, cost and quotas: `docs/guides/evaluation.md`.

- **Runs cost real money or quota.** Know the cost before starting (see the guide). On the Gemini free tier (20 requests/day per model) a full eval cannot run.
- **Check the log for API errors before trusting any score.** A failed call becomes an "answer" and gets scored. Use `grep -ciE "429|RESOURCE_EXHAUSTED|quota|Error generating answer"`; it must be 0.
- **Results stay local.** Scripts write to `results/eval/` (gitignored). Only `results/FINAL.md`, `results/README.md` and `results/archive/INDEX.md` are tracked. New published numbers get a row in `results/README.md` (file, SHA, numbers).
- **Never commit FinanceBench data** (`data/eval/financebench_in_corpus.jsonl`, CC BY-NC). Scripts download it from Hugging Face when it is missing.
- **Use the configured model.** Get the model from `get_llm_model()` / `chat_completion()` in `src/utils/llm_client.py`, never `settings.openai_model` or a literal name. Pass `llm=` and `embeddings=` from `get_ragas_judge()` to every `ragas.evaluate` call.
- **Shared code lives in `src/evaluation/`, not in scripts.** Question loaders and company lists are in `eval_data.py`; pipeline-running, Ragas and answer-classifying helpers are in `eval_helpers.py`. A script must not import another script; check with `grep -rn --include="*.py" "from scripts.eval" scripts/eval`, which should print nothing.
- **Two company lists, on purpose.** `IN_CORPUS_COMPANIES` (5 FinanceBench companies) and `OOC_EXCLUDED_COMPANIES` (same plus Apple, used to pick out-of-corpus questions). Do not merge them.
- **Retired scripts are in `archive/`** (see `archive/README.md`). They find the project root with `Path(__file__).parents[3]`; a script moved to another depth needs that line changed.
- After any change here, run `poetry run python -m compileall -q scripts` and the BASELINE import check.
- **Most runners copy the pipeline steps** instead of calling `build_crag_graph` (REVIEW.md item 19). If you change a step's order or routing in `src/orchestration/graph.py`, check `eval_fb27_no_injection.py`, `eval_part_b.py` and `eval_part_b_aapl50.py` too.
