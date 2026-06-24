"""
database/vector_store.py — Vector store abstraction supporting ChromaDB (local)
and Pinecone (cloud). Selected via VECTOR_STORE env var.
"""

from __future__ import annotations
import logging
from typing import Optional
from dataclasses import dataclass, field

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single result from a vector similarity search."""
    id: str
    text: str
    metadata: dict
    score: float = 0.0


class ChromaVectorStore:
    """Local vector store backed by ChromaDB. No API key required."""

    def __init__(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB initialised — collection: {settings.CHROMA_COLLECTION_NAME}")

    # ── Public API ──────────────────────────────────────────────────────────

    def upsert(self, ids: list[str], embeddings: list[list[float]], texts: list[str], metadatas: list[dict]) -> None:
        """Insert or update vectors in the collection."""
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.debug(f"Upserted {len(ids)} vectors into ChromaDB")

    def query(self, query_embedding: list[float], top_k: int = 20) -> list[SearchResult]:
        """Retrieve top-k nearest neighbours for the query embedding."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        for i, doc_id in enumerate(results["ids"][0]):
            # Chroma returns cosine *distance* (0 = identical); convert to similarity
            distance = results["distances"][0][i]
            score = 1.0 - distance
            search_results.append(
                SearchResult(
                    id=doc_id,
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    score=round(score, 4),
                )
            )
        return search_results

    def get_all_chunks(self, limit: int = 200, offset: int = 0) -> list[SearchResult]:
        """Fetch all stored chunks (paginated)."""
        total = self._collection.count()
        if total == 0:
            return []

        results = self._collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        return [
            SearchResult(
                id=results["ids"][i],
                text=results["documents"][i],
                metadata=results["metadatas"][i],
            )
            for i in range(len(results["ids"]))
        ]

    def delete(self, chunk_id: str) -> None:
        """Delete a chunk by its ID."""
        self._collection.delete(ids=[chunk_id])
        logger.debug(f"Deleted chunk {chunk_id}")

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks belonging to a document source."""
        results = self._collection.get(where={"source": source}, include=[])
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunks for source: {source}")
        return len(ids)

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._collection.count()


class PineconeVectorStore:
    """Cloud vector store backed by Pinecone."""

    def __init__(self):
        import pinecone

        pinecone.init(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT,
        )
        self._index = pinecone.Index(settings.PINECONE_INDEX_NAME)
        logger.info(f"Pinecone initialised — index: {settings.PINECONE_INDEX_NAME}")

    def upsert(self, ids: list[str], embeddings: list[list[float]], texts: list[str], metadatas: list[dict]) -> None:
        vectors = [
            (ids[i], embeddings[i], {**metadatas[i], "text": texts[i]})
            for i in range(len(ids))
        ]
        self._index.upsert(vectors=vectors)

    def query(self, query_embedding: list[float], top_k: int = 20) -> list[SearchResult]:
        response = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )
        results = []
        for match in response["matches"]:
            meta = match.get("metadata", {})
            results.append(
                SearchResult(
                    id=match["id"],
                    text=meta.pop("text", ""),
                    metadata=meta,
                    score=round(match.get("score", 0.0), 4),
                )
            )
        return results

    def get_all_chunks(self, limit: int = 200, offset: int = 0) -> list[SearchResult]:
        raise NotImplementedError("Pinecone does not support listing all vectors. Use fetch by ID.")

    def delete(self, chunk_id: str) -> None:
        self._index.delete(ids=[chunk_id])

    def delete_by_source(self, source: str) -> int:
        self._index.delete(filter={"source": {"$eq": source}})
        return -1  # Pinecone doesn't return count

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.get("total_vector_count", 0)


def get_vector_store() -> ChromaVectorStore | PineconeVectorStore:
    """Factory: returns the vector store configured in settings."""
    if settings.VECTOR_STORE == "pinecone":
        return PineconeVectorStore()
    return ChromaVectorStore()
