"""
tests/test_reranker.py — Unit tests for the re-ranking module.
"""

import pytest
from unittest.mock import MagicMock, patch
from database.vector_store import SearchResult


def _make_results(n: int = 10) -> list[SearchResult]:
    return [
        SearchResult(
            id=f"chunk-{i}",
            text=f"This is test chunk number {i} with some relevant content.",
            metadata={"source": "test.pdf", "chunk_index": i},
            score=round(0.9 - i * 0.05, 2),
        )
        for i in range(n)
    ]


class TestNoOpReranker:
    def test_returns_top_n(self):
        from core.reranker import NoOpReranker
        reranker = NoOpReranker()
        results = _make_results(10)
        ranked = reranker.rerank("test query", results, top_n=3)
        assert len(ranked) == 3

    def test_rank_is_sequential(self):
        from core.reranker import NoOpReranker
        reranker = NoOpReranker()
        ranked = reranker.rerank("test", _make_results(5), top_n=5)
        assert [r.rank for r in ranked] == [1, 2, 3, 4, 5]

    def test_vector_score_equals_rerank_score(self):
        from core.reranker import NoOpReranker
        reranker = NoOpReranker()
        results = _make_results(3)
        ranked = reranker.rerank("test", results, top_n=3)
        for r in ranked:
            assert r.vector_score == r.rerank_score

    def test_empty_results(self):
        from core.reranker import NoOpReranker
        reranker = NoOpReranker()
        assert reranker.rerank("test", [], top_n=5) == []


class TestLocalReranker:
    @pytest.mark.skipif(
        not pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed"),
        reason="sentence-transformers required"
    )
    def test_reranks_results(self):
        from core.reranker import LocalReranker
        reranker = LocalReranker()
        results = _make_results(5)
        ranked = reranker.rerank("test query about chunks", results, top_n=3)
        assert len(ranked) == 3
        assert all(r.rerank_score is not None for r in ranked)
        # Results should be sorted by rerank_score descending
        scores = [r.rerank_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_results(self):
        from core.reranker import LocalReranker
        with patch("sentence_transformers.CrossEncoder") as MockCE:
            MockCE.return_value = MagicMock()
            reranker = LocalReranker.__new__(LocalReranker)
            reranker._model = MockCE.return_value
            assert reranker.rerank("test", [], top_n=5) == []


class TestGetReranker:
    def test_returns_noop_when_mode_is_none(self):
        with patch("core.reranker.settings") as mock_settings:
            mock_settings.RERANKER_MODE = "none"
            from core.reranker import get_reranker, NoOpReranker
            reranker = get_reranker()
            assert isinstance(reranker, NoOpReranker)

    def test_falls_back_to_local_when_cohere_key_missing(self):
        with patch("core.reranker.settings") as mock_settings:
            mock_settings.RERANKER_MODE = "cohere"
            mock_settings.COHERE_API_KEY = ""
            mock_settings.LOCAL_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            with patch("sentence_transformers.CrossEncoder"):
                from core.reranker import get_reranker, LocalReranker
                reranker = get_reranker()
                assert isinstance(reranker, LocalReranker)
