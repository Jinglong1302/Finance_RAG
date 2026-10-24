# src/orchestration — rules for this area

- **Production grader is Part B.** `graph.py` wires `sufficiency_grader_node` into the `grade` step (owner decision, 2026-10-23). `nodes/grader.py`, the per-chunk grader, is kept only for old eval comparisons. `tests/test_orchestration/test_graph_wiring.py` fails if the wiring changes.
- **No provider or model names in node code.** Call `chat_completion()` from `src/utils/llm_client.py`, and use the model name it returns for cost and trace. It handles Gemini vs OpenAI differences (no `seed` on Gemini, thinking turned off).
- **Keep `temperature=0`, and keep passing `seed=42`.** The client drops `seed` for providers that reject it.
- **Short replies need thinking off.** `sufficiency_grader.py` asks for `max_tokens=10`. On Gemini that only works with `reasoning_effort="none"`, which is the default; otherwise the reply comes back empty and every question is refused.
- **A changed prompt changes the stub.** `src/utils/llm_stub.py` picks its fake reply by keywords in the system prompt (`prompts/*.py`). If you reword a system prompt, run the graph test.
- **Max rewrite cycles is written as `2` in four places** (`graph.py`, `grader.py`, `sufficiency_grader.py`, `hallucination_guard.py`). Change all of them, or better, read `MAX_CRAG_CYCLES` from settings (REVIEW.md item 8).
- Known open issues here: errors become answers (REVIEW.md item 6); LLM-written filters are not checked (item 7).
