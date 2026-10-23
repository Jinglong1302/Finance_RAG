"""End-to-end graph wiring test (offline: stub LLM, fake search, fake reranker model)."""

import pytest

from src.orchestration.graph import build_crag_graph, run_query
from src.retrieval.hybrid_search import SearchResult
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.llm_stub import disable_stub_mode, enable_stub_mode


class FakeSearcher:
    def search(self, query, filters=None, top_k=25):
        return [
            SearchResult(chunk_id=f"c{i}", text=f"t{i}", metadata={}, score=0.03 - i / 1000)
            for i in range(6)
        ]


class FakeModel:
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    def compute_score(self, pairs):
        self.calls += 1
        return [self.score] * len(pairs)


class FakeExpander:
    def expand(self, results):
        from src.retrieval.parent_expander import EnrichedContext

        return [
            EnrichedContext(
                child_text=r.text,
                parent_text=None,
                metadata=r.metadata,
                rerank_score=r.score,
                chunk_id=r.chunk_id,
            )
            for r in results
        ]


@pytest.fixture
def stub_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    enable_stub_mode()
    yield
    disable_stub_mode()


def _graph(reranker_score: float):
    reranker = CrossEncoderReranker()
    model = FakeModel(reranker_score)
    reranker._model = model
    graph = build_crag_graph(
        searcher=FakeSearcher(),
        reranker=reranker,
        expander=FakeExpander(),
        qdrant_client=None,  # note expander only queries Qdrant when chunks reference notes
    )
    return graph, model


def _trace_nodes(result) -> list[str]:
    return [t.get("node") for t in result.get("pipeline_trace", [])]


def test_production_graph_uses_sufficiency_grader(stub_llm) -> None:
    graph, model = _graph(reranker_score=1.0)
    result = run_query(graph, "What was revenue in FY2024?")

    nodes = _trace_nodes(result)
    assert "sufficiency_grader" in nodes
    assert "grader" not in nodes
    assert model.calls >= 1
    assert result["final_answer"].startswith("STUB ANSWER")
    assert not result.get("is_abstention")


def test_low_rerank_score_abstains_at_coarse_gate(stub_llm) -> None:
    graph, model = _graph(reranker_score=-5.0)
    result = run_query(graph, "What was revenue in FY2024?")

    assert model.calls >= 1
    assert result["is_abstention"] is True
    assert result["abstention_stage"] == 1
    assert "generator" not in _trace_nodes(result)
