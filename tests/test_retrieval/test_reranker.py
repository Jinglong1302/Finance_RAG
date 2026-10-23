"""Tests for the cross-encoder reranker and the reranker node (offline, fake model)."""

from types import SimpleNamespace

from src.orchestration.nodes.reranker_node import make_reranker_node
from src.retrieval.hybrid_search import SearchResult
from src.retrieval.reranker import CrossEncoderReranker


class FakeModel:
    """Stands in for FlagReranker: scores each pair by a fixed lookup on the text."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self.scores_by_text = scores_by_text
        self.calls = 0

    def compute_score(self, pairs: list[list[str]]) -> list[float]:
        self.calls += 1
        return [self.scores_by_text[text] for _, text in pairs]


class FakeExpander:
    """Stands in for ParentExpander: no parents, passes results through."""

    def expand(self, results: list[SearchResult]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                child_text=r.text,
                parent_text=None,
                metadata=r.metadata,
                rerank_score=r.score,
                chunk_id=r.chunk_id,
            )
            for r in results
        ]


def _candidates(n: int) -> list[SearchResult]:
    # Hybrid (RRF) order: c0 first, small positive scores like real RRF output
    return [
        SearchResult(chunk_id=f"c{i}", text=f"t{i}", metadata={}, score=0.03 - i / 1000)
        for i in range(n)
    ]


def _reranker(scores_by_text: dict[str, float]) -> tuple[CrossEncoderReranker, FakeModel]:
    reranker = CrossEncoderReranker()
    model = FakeModel(scores_by_text)
    reranker._model = model
    return reranker, model


class TestCrossEncoderReranker:
    def test_scores_when_top_k_equals_candidate_count(self) -> None:
        # The reranker node asks for top_k=len(candidates); the model must still run
        reranker, model = _reranker({"t0": -1.0, "t1": 3.0, "t2": 0.5})
        result = reranker.rerank("q", _candidates(3), top_k=3)
        assert model.calls == 1
        assert [r.chunk_id for r in result] == ["c1", "c2", "c0"]
        assert [r.score for r in result] == [3.0, 0.5, -1.0]

    def test_scores_when_fewer_candidates_than_top_k(self) -> None:
        reranker, model = _reranker({"t0": -4.0, "t1": 2.0})
        result = reranker.rerank("q", _candidates(2), top_k=5)
        assert model.calls == 1
        assert [r.chunk_id for r in result] == ["c1", "c0"]

    def test_truncates_to_top_k(self) -> None:
        reranker, _ = _reranker({f"t{i}": float(i) for i in range(6)})
        result = reranker.rerank("q", _candidates(6), top_k=2)
        assert [r.chunk_id for r in result] == ["c5", "c4"]

    def test_empty_candidates(self) -> None:
        reranker, model = _reranker({})
        assert reranker.rerank("q", [], top_k=5) == []
        assert model.calls == 0


class TestRerankerNode:
    def test_node_uses_cross_encoder_scores(self) -> None:
        # Reverse the hybrid order and make every score low enough to trip the coarse gate
        scores = {f"t{i}": -10.0 + i for i in range(8)}
        reranker, model = _reranker(scores)
        node = make_reranker_node(reranker, FakeExpander())
        search_results = [
            {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata, "score": c.score}
            for c in _candidates(8)
        ]

        out = node({"search_results": search_results, "original_query": "q", "result_count": 3})

        assert model.calls == 1
        assert [r["chunk_id"] for r in out["reranked_results"]] == ["c7", "c6", "c5"]
        assert out["reranked_results"][0]["score"] == -3.0
        assert [e["rerank_score"] for e in out["enriched_contexts"]] == [-3.0, -4.0, -5.0]
