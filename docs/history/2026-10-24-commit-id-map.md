# Commit ID map after the history rewrite

> **Historical record from 2026-10-24.** Kept as written at the time. For current docs start at [docs/README.md](../README.md).

On 2026-10-24 the git history was rewritten to remove `data/eval/financebench_in_corpus.jsonl` (CC BY-NC 4.0) from every commit, on all branches and the `v1.0` tag, and then force-pushed (owner's decision; REVIEW.md item 5). The code in every commit is unchanged apart from that one file.

That gave every commit from 2026-09-26 on a new ID. Current docs were updated to the new IDs. **History docs, old commit messages and your local result files (`results/eval/*.json`, field `git_sha`) still show old IDs**: look them up here. The 11 commits before 2026-09-26 kept their IDs and are not listed.

52 commits, oldest first. Matched by author, author date, commit date and subject; all 63 commits matched exactly once.

| Old ID | New ID | Date | Subject |
|---|---|---|---|
| `e9de99f` | `0064ffc` | 2026-09-26 | feat(eval): align runners with 4-track matrix (FB 27, AAPL 50, TAT-QA 50, Abstention 20) |
| `281ad9f` | `a14ee39` | 2026-09-26 | feat(eval): add stripped-down naive RAG baseline and query CLI |
| `c102411` | `4da4c3b` | 2026-09-26 | feat(eval): add page match OR overlap rule and chunk page tagging |
| `e74ef72` | `c1f6bd6` | 2026-09-26 | feat(crag): add two-stage abstention mechanism and classifier |
| `09ae61b` | `5e3bf50` | 2026-09-26 | feat(ingest): add historical filing discovery and ingestion pipeline |
| `9397566` | `b308512` | 2026-09-26 | fix(eval): use covariant Sequence type for evidence_list in eval_retrieval |
| `8d909f8` | `b534ab1` | 2026-09-26 | feat(telemetry): unify latency and token-cost tracking across CRAG and Naive RAG |
| `9ffe46d` | `48299d2` | 2026-09-26 | feat(eval): add --limit alias to eval_generation_tatqa |
| `d42def9` | `5480bb7` | 2026-09-26 | feat(eval): integrate AAPL evaluation and per-question answer comparison into summary report |
| `eff3039` | `9b7bbfa` | 2026-09-26 | feat(eval): record and display per-question Naive RAG answers in TAT-QA evaluation |
| `62ead20` | `34bb265` | 2026-09-26 | chore(eval): remove legacy evaluate.py and query_baseline.py, consolidate query_naive.py |
| `78493bb` | `9b99758` | 2026-09-26 | feat(eval): display complete questions, retrieved chunks comparison, and Naive answers in Slices 1, 2, and 5 |
| `b63bee2` | `ab862a5` | 2026-09-26 | feat(eval): record and display per-question Ragas scores in Slice 2 summary |
| `e09bdfa` | `c9fb8da` | 2026-09-26 | feat(eval): wire distinct generation paths in TAT-QA and enhance numeric parsing |
| `8f0f6dc` | `86f1f79` | 2026-09-26 | feat(eval): add --no-reranker support across pipeline and evaluation harness |
| `c35ae4d` | `c01e525` | 2026-09-26 | fix(eval): accept --no-reranker flag in TAT-QA eval |
| `63f1fcb` | `e20e409` | 2026-09-27 | feat(eval): add 3-way comparative evaluation report across Naive, CRAG+Reranker, and CRAG-NoReranker |
| `9b62366` | `f393d67` | 2026-09-27 | feat(eval): compile full granular 3-pipeline comparison report |
| `cc77014` | `6cc0219` | 2026-09-27 | fix(eval): add missing return type annotation to main in generate_3_way_report.py |
| `907cff9` | `1b76dd6` | 2026-09-27 | fix(eval): address fiscal-year gap, page matching, TAT-QA span metrics, and 4-way classification |
| `cc7d1b1` | `54be88a` | 2026-09-27 | fix(eval): update summary aggregation and generate_3_way_report with granular flags |
| `081838e` | `9c5d1fa` | 2026-09-27 | fix(types): fix NoneType payload access in tag_chunk_pages and add return type to reindex_mmm |
| `a51143f` | `d12f863` | 2026-09-27 | fix(crag): fix grading threshold, prompt strictness, candidate accumulation, and reranker caching |
| `7c045da` | `dd6437d` | 2026-09-27 | fix(eval): fix numeric unit normalization, pin deterministic LLM sampling, ship weighted RRF |
| `5ae7f75` | `922d113` | 2026-09-28 | chore(git): ignore local qdrant database files |
| `5554556` | `f738d67` | 2026-09-28 | fix(eval): switch gold matching to containment and run real retrieval path |
| `ae06e08` | `06fe3b7` | 2026-09-28 | fix(eval): scope numeric accuracy to numeric questions and remove stale ragas fallback |
| `5cbc72d` | `8821c98` | 2026-09-28 | feat(retrieval): record dense, sparse, and RRF ranks with query expansion |
| `cbac640` | `3d40581` | 2026-09-28 | feat(orchestration): improve query preservation, table diversity, and fiscal guardrails |
| `3122ccd` | `d0bade1` | 2026-09-28 | feat(chunking): add section tagging and table normalization improvements |
| `94fdd2f` | `d4e92ea` | 2026-09-28 | chore(git): ignore *.snapshot files |
| `bb3321d` | `f9a7449` | 2026-09-29 | fix(eval): uniform refusal detector covering naive baseline phrase and expanded GPT-4o formulations |
| `78e05dd` | `31b93e9` | 2026-09-29 | fix(eval): raise overlap_match threshold 0.50->0.65; add consolidated Part A eval script (ablations a/b/c + oracle + dev-5) |
| `e3d3e42` | `e3c1871` | 2026-09-30 | feat(eval): add LLM disk cache, full-CRAG holdout-22 eval, and PROGRESS log |
| `b630a6b` | `dc20780` | 2026-09-30 | feat(eval): add Part B pooled-context sufficiency grader and Tier 1/2 eval |
| `b5f9c2e` | `4cfad9d` | 2026-09-30 | chore(eval): ship Part B — FINAL.md, AAPL50 eval, reproducible README |
| `8893b0a` | `be31255` | 2026-10-01 | feat(eval): Phase A/B/C eval — production path FB27, stub-LLM, Ragas |
| `19a0a8a` | `1912f56` | 2026-10-01 | fix(eval): correct swapped context_precision/answer_relevancy in Ragas baseline |
| `645a0b2` | `66d0863` | 2026-10-01 | merge(eval/part-a-repair): ship v1.0 — Part B sufficiency grader, Phase A–C evals |
| `1b2664b` | `45966d4` | 2026-10-01 | chore(repo): tidy-up — MIT license, data license notes, archive stale results, move PROGRESS.md |
| `a1ce7fb` | `585a6be` | 2026-10-23 | docs(engineering): add phase 1 review, baseline, AI call map and principles |
| `736a6ca` | `89461b0` | 2026-10-23 | docs(engineering): record owner decision that production uses the Part B grader |
| `345bf10` | `fee402c` | 2026-10-23 | fix(orchestration): use Part B grader in production graph and make reranker score every list |
| `4423984` | `34f5ab7` | 2026-10-23 | feat(llm): switch AI provider from .env and use Gemini by default |
| `28c2271` | `0b47e05` | 2026-10-23 | merge(review/engineering-2026-10-23): bring phase 1 review docs into the docs reorganisation |
| `421ce1c` | `2d12325` | 2026-10-23 | docs(structure): move guides to docs/guides and dated logs to docs/history |
| `ca5dfdb` | `491ceb8` | 2026-10-23 | docs(structure): add docs map, CLAUDE.md index and area rules, eval guide, status updates |
| `8ccc6fa` | `b933348` | 2026-10-23 | refactor(eval): move shared FinanceBench helpers from eval_retrieval.py into src/evaluation |
| `e892d82` | `d4cfb27` | 2026-10-23 | refactor(eval): use shared OOC loader and answer classifier from src/evaluation |
| `40145ce` | `cde5ed9` | 2026-10-23 | refactor(eval): use shared AAPL50 and FinanceBench loaders, run_crag and run_ragas from src/evaluation |
| `64ff2ba` | `a1373d4` | 2026-10-23 | refactor(eval): use shared TAT-QA loader and context generator, so no eval script imports another |
| `e547891` | `0e25dd4` | 2026-10-23 | refactor(eval): archive four superseded eval scripts and update docs for the new layout |
