# scripts/eval — rules for this area

How to run the evals, cost and quotas: `docs/guides/evaluation.md`.

- **Runs cost real money or quota.** Know the cost before starting (see the guide). On the Gemini free tier (20 requests/day per model) a full eval cannot run.
- **Check the log for API errors before trusting any score.** A failed call becomes an "answer" and gets scored. Use `grep -ciE "429|RESOURCE_EXHAUSTED|quota|Error generating answer"`; it must be 0.
- **Results stay local.** Scripts write to `results/eval/` (gitignored). Only `results/FINAL.md`, `results/README.md` and `results/archive/INDEX.md` are tracked. New published numbers get a row in `results/README.md` (file, SHA, numbers).
- **Never commit FinanceBench data** (`data/eval/financebench_in_corpus.jsonl`, CC BY-NC). Scripts download it from Hugging Face when it is missing.
- **Use the configured model.** Get the model from `get_llm_model()` / `chat_completion()` in `src/utils/llm_client.py`, never `settings.openai_model` or a literal name. Pass `llm=` and `embeddings=` from `get_ragas_judge()` to every `ragas.evaluate` call.
- **Scripts import each other.** Seven scripts import `eval_retrieval.py`, and others import `eval_generation_financebench.py`, `eval_abstention.py` and `eval_generation_tatqa.py`. Renaming or moving one of those breaks the rest. Run `poetry run python -m compileall -q scripts` and the BASELINE import check after any change.
- **Most runners copy the pipeline steps** instead of calling `build_crag_graph` (REVIEW.md item 19). If you change a step's order or routing in `src/orchestration/graph.py`, check `eval_fb27_no_injection.py`, `eval_part_b.py` and `eval_part_b_aapl50.py` too.
