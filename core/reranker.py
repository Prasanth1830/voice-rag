"""
core/reranker.py — Cross-encoder re-ranking module.

After vector similarity search returns top-K candidates, the reranker
scores each (query, chunk) pair with a more powerful cross-encoder model
and returns only the top-N highest-scoring chunks.

Modes (controlled via RERANKER_MODE env var):
  - "cohere" : Cohere Rerank API  (fast, cloud, requires COHERE_API_KEY)
  - "local"  : sentence-transformers cross-encoder  (slower, fully local)
  - "none"   : skip reranking, return raw vector results unchanged
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

from config import settings
from database.vector_store import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """A chunk after re-ranking with both vector score and rerank score."""
    id: str
    text: str
    metadata: dict
    vector_score: float      # cosine similarity from vector search (0–1)
    rerank_score: float      # cross-encoder relevance score (higher = better)
    rank: int                # 1-indexed position after reranking


class CohereReranker:
    """
    Re-ranker powered by Cohere's Rerank API.
    Requires COHERE_API_KEY in environment.
    """

    def __init__(self):
        import cohere
        self._client = cohere.Client(settings.COHERE_API_KEY)
        self._model = settings.COHERE_RERANK_MODEL
        logger.info(f"Cohere reranker initialised (model: {self._model})")

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[RankedResult]:
        if not results:
            return []

        documents = [r.text for r in results]
        response = self._client.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            model=self._model,
        )

        ranked: list[RankedResult] = []
        for rank_idx, hit in enumerate(response.results, start=1):
            original = results[hit.index]
            ranked.append(
                RankedResult(
                    id=original.id,
                    text=original.text,
                    metadata=original.metadata,
                    vector_score=original.score,
                    rerank_score=round(hit.relevance_score, 4),
                    rank=rank_idx,
                )
            )
        logger.debug(f"Cohere reranked {len(results)} → {len(ranked)} results")
        return ranked


class LocalReranker:
    """
    Re-ranker using a locally downloaded cross-encoder model via
    sentence-transformers. Fully offline, no API key required.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - ~86 MB download on first use
      - Fast on CPU, excellent retrieval quality
    """

    def __init__(self):
        from sentence_transformers import CrossEncoder
        model_name = settings.LOCAL_RERANKER_MODEL
        logger.info(f"Loading local cross-encoder: {model_name}")
        self._model = CrossEncoder(model_name, max_length=512)
        logger.info("Local reranker ready")

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[RankedResult]:
        if not results:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, r.text) for r in results]
        scores: list[float] = self._model.predict(pairs).tolist()

        # Zip scores with original results and sort descending
        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)

        ranked: list[RankedResult] = []
        for rank_idx, (score, original) in enumerate(scored[:top_n], start=1):
            ranked.append(
                RankedResult(
                    id=original.id,
                    text=original.text,
                    metadata=original.metadata,
                    vector_score=original.score,
                    rerank_score=round(float(score), 4),
                    rank=rank_idx,
                )
            )
        logger.debug(f"Local reranked {len(results)} → {len(ranked)} results")
        return ranked


class NoOpReranker:
    """
    Pass-through reranker — returns raw vector results unchanged.
    Used when RERANKER_MODE=none.
    """

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[RankedResult]:
        ranked: list[RankedResult] = []
        for rank_idx, r in enumerate(results[:top_n], start=1):
            ranked.append(
                RankedResult(
                    id=r.id,
                    text=r.text,
                    metadata=r.metadata,
                    vector_score=r.score,
                    rerank_score=r.score,   # same as vector score
                    rank=rank_idx,
                )
            )
        return ranked


def get_reranker() -> CohereReranker | LocalReranker | NoOpReranker:
    """
    Factory: returns the reranker configured via RERANKER_MODE.

    Falls back gracefully:
      cohere  → if COHERE_API_KEY is missing → falls back to local
      local   → requires sentence-transformers (auto-downloaded on first use)
      none    → no reranking
    """
    mode = settings.RERANKER_MODE

    if mode == "cohere":
        if not settings.COHERE_API_KEY:
            logger.warning("RERANKER_MODE=cohere but COHERE_API_KEY is not set — falling back to local reranker")
            return LocalReranker()
        return CohereReranker()

    if mode == "local":
        return LocalReranker()

    logger.info("Reranking disabled (RERANKER_MODE=none)")
    return NoOpReranker()
