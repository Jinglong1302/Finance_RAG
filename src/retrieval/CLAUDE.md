# src/retrieval — rules for this area

- **Two kinds of score, two scales.** `HybridSearcher` returns RRF scores: small positive numbers, at most about 0.03. `CrossEncoderReranker` returns cross-encoder logits, which can be negative (one real query on 2026-10-23 scored +2.2 down to −4.2). The coarse gate (`RERANKER_COARSE_THRESHOLD=-2.0`) and the table check in `reranker_node.py` (`score > -3.0`) only make sense on reranker scores.
- **The reranker must always score.** `rerank()` may return early only for an empty list. Until 2026-10-23 it skipped the model whenever `top_k >= len(candidates)`, which is exactly how the pipeline calls it. `tests/test_retrieval/test_reranker.py` guards this.
- **The reranker model comes from `.env`** (`RERANKER_MODEL`, owner uses `BAAI/bge-reranker-base`); the code default is `-large`.
- **Score cache grows without limit** (`reranker.py` `_score_cache`). Fine for scripts; for a long-running UI server, see REVIEW.md item 16.
- **Filters from the LLM go straight into Qdrant** (`_build_qdrant_filter`): `int()` on `fiscal_year` raises on values like `"FY2024"` (REVIEW.md item 7).
- Test without Qdrant or model downloads by using fakes: see `tests/test_retrieval/` and `tests/test_orchestration/test_graph_wiring.py`.
