"""
core/rag_engine.py — Orchestrates the full RAG pipeline:

  Document upload:
    file bytes → DocumentProcessor → chunks → embed → VectorStore.upsert

  Query:
    query text → embed → VectorStore.query (top-K)
               → Reranker.rerank (top-N)
               → build prompt → OpenAI Chat → answer

  Voice query:
    audio bytes → VoiceProcessor → text query → (same as above)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from config import settings
from core.document_processor import DocumentProcessor, DocumentChunk
from core.reranker import get_reranker, RankedResult
from core.voice_processor import VoiceProcessor
from database.vector_store import get_vector_store, SearchResult
from utils.helpers import build_rag_prompt, timeit

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """Full response from a RAG query."""
    query: str
    answer: str
    vector_results: list[SearchResult]          # raw top-K from vector search
    reranked_results: list[RankedResult]         # top-N after reranking
    reranker_mode: str
    tokens_used: int = 0
    model: str = ""


class RAGEngine:
    """
    Central orchestrator for the RAG pipeline.
    Singleton-friendly — initialise once at app startup.
    """

    def __init__(self):
        self._openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._vector_store = get_vector_store()
        self._document_processor = DocumentProcessor()
        self._reranker = get_reranker()
        self._voice_processor: Optional[VoiceProcessor] = None  # lazy-loaded
        logger.info(
            f"RAGEngine ready | vector_store={settings.VECTOR_STORE} "
            f"| reranker={settings.RERANKER_MODE}"
        )

    # ── Document Ingestion ─────────────────────────────────────────────────

    @timeit
    def ingest_document(self, content: bytes, filename: str) -> dict:
        """
        Parse, chunk, embed, and store a document.

        Returns:
            dict with keys: filename, total_chunks, file_hash
        """
        chunks: list[DocumentChunk] = self._document_processor.process(content, filename)

        # Embed all chunks in one batched API call
        embeddings = self._embed_texts([c.text for c in chunks])

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        self._vector_store.upsert(
            ids=[c.id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            texts=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

        logger.info(f"Ingested '{filename}': {len(chunks)} chunks stored")
        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "file_hash": chunks[0].metadata["file_hash"] if chunks else "",
        }

    # ── Query ──────────────────────────────────────────────────────────────

    @timeit
    def query(self, query_text: str) -> QueryResponse:
        """
        Run a full RAG query:
          embed → vector search → rerank → LLM answer

        Args:
            query_text: The user's natural language question.

        Returns:
            QueryResponse with answer and all intermediate results.
        """
        logger.info(f"Query: '{query_text[:80]}'")

        # 1. Embed query
        query_embedding = self._embed_texts([query_text])[0]

        # 2. Vector similarity search (top-K)
        vector_results: list[SearchResult] = self._vector_store.query(
            query_embedding=query_embedding,
            top_k=settings.RETRIEVAL_TOP_K,
        )
        logger.debug(f"Vector search returned {len(vector_results)} results")

        # 3. Re-rank (top-N)
        reranked_results: list[RankedResult] = self._reranker.rerank(
            query=query_text,
            results=vector_results,
            top_n=settings.RERANKER_TOP_N,
        )

        # 4. Build prompt from top-N reranked chunks
        context_chunks = [r.text for r in reranked_results]
        messages = build_rag_prompt(query=query_text, context_chunks=context_chunks)

        # 5. Generate answer via OpenAI Chat
        completion = self._openai.chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        )
        answer = completion.choices[0].message.content.strip()
        tokens_used = completion.usage.total_tokens if completion.usage else 0

        return QueryResponse(
            query=query_text,
            answer=answer,
            vector_results=vector_results,
            reranked_results=reranked_results,
            reranker_mode=settings.RERANKER_MODE,
            tokens_used=tokens_used,
            model=settings.OPENAI_CHAT_MODEL,
        )

    # ── Voice Query ────────────────────────────────────────────────────────

    def voice_query(self, audio_bytes: bytes, filename: str = "audio.wav") -> QueryResponse:
        """
        Transcribe audio then run a RAG query.

        Args:
            audio_bytes: Raw audio content.
            filename: Original filename for format detection.

        Returns:
            QueryResponse (same as text query).
        """
        if self._voice_processor is None:
            self._voice_processor = VoiceProcessor()

        transcription = self._voice_processor.transcribe(audio_bytes, filename)
        logger.info(f"Voice transcription: '{transcription}'")
        return self.query(transcription)

    # ── Vector Store Helpers ───────────────────────────────────────────────

    def get_all_chunks(self, limit: int = 200, offset: int = 0) -> list[SearchResult]:
        return self._vector_store.get_all_chunks(limit=limit, offset=offset)

    def delete_chunk(self, chunk_id: str) -> None:
        self._vector_store.delete(chunk_id)

    def delete_document(self, source: str) -> int:
        return self._vector_store.delete_by_source(source)

    def get_chunk_count(self) -> int:
        return self._vector_store.count()

    # ── Internal ───────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts using the OpenAI Embeddings API.
        Batches automatically handled by the SDK.
        """
        response = self._openai.embeddings.create(
            input=texts,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
        # Sort by index to preserve order
        embeddings = sorted(response.data, key=lambda e: e.index)
        return [e.embedding for e in embeddings]
