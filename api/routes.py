"""
api/routes.py — FastAPI router with all REST endpoints.

Endpoints:
  GET  /health                 — health check
  POST /upload                 — upload & ingest a document
  GET  /chunks                 — list all indexed chunks (paginated)
  DELETE /chunks/{chunk_id}    — delete a single chunk
  DELETE /documents/{source}   — delete all chunks from a document
  POST /query                  — text RAG query
  POST /voice-query            — voice RAG query
  GET  /stats                  — index statistics
"""

from __future__ import annotations
import logging
from typing import Optional, Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from core.rag_engine import RAGEngine, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Shared engine instance ─────────────────────────────────────────────────────
# Initialised once and reused across all requests.
_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


# ── Request / Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None     # override RETRIEVAL_TOP_K
    top_n: Optional[int] = None     # override RERANKER_TOP_N


class RankedResultOut(BaseModel):
    id: str
    text: str
    metadata: dict
    vector_score: float
    rerank_score: float
    rank: int


class VectorResultOut(BaseModel):
    id: str
    text: str
    metadata: dict
    score: float


class QueryResponseOut(BaseModel):
    query: str
    answer: str
    reranker_mode: str
    tokens_used: int
    model: str
    vector_results: list[VectorResultOut]
    reranked_results: list[RankedResultOut]


class ChunkOut(BaseModel):
    id: str
    text: str
    metadata: dict
    score: float = 0.0


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
async def health_check():
    """Returns service health status."""
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "vector_store": settings.VECTOR_STORE,
        "reranker_mode": settings.RERANKER_MODE,
    }


@router.get("/stats", tags=["System"])
async def get_stats():
    """Returns index statistics."""
    engine = get_engine()
    return {
        "total_chunks": engine.get_chunk_count(),
        "vector_store": settings.VECTOR_STORE,
        "reranker_mode": settings.RERANKER_MODE,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "reranker_top_n": settings.RERANKER_TOP_N,
        "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
        "chat_model": settings.OPENAI_CHAT_MODEL,
    }


@router.post("/upload", tags=["Documents"], status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF, TXT, DOCX).
    The document is parsed, chunked, embedded, and stored in the vector store.
    """
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    content = await file.read()

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB} MB",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        engine = get_engine()
        result = engine.ingest_document(content, file.filename or "upload")
        return {"success": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error during document ingestion")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/chunks", tags=["Documents"], response_model=list[ChunkOut])
async def list_chunks(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List all indexed chunks with pagination support."""
    engine = get_engine()
    chunks = engine.get_all_chunks(limit=limit, offset=offset)
    return [
        ChunkOut(id=c.id, text=c.text, metadata=c.metadata, score=c.score)
        for c in chunks
    ]


@router.delete("/chunks/{chunk_id}", tags=["Documents"])
async def delete_chunk(chunk_id: str):
    """Delete a single chunk by its ID."""
    try:
        get_engine().delete_chunk(chunk_id)
        return {"success": True, "deleted_id": chunk_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{source:path}", tags=["Documents"])
async def delete_document(source: str):
    """Delete all chunks belonging to a document (by source filename)."""
    try:
        deleted = get_engine().delete_document(source)
        return {"success": True, "source": source, "deleted_chunks": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", tags=["Query"], response_model=QueryResponseOut)
async def query(request: QueryRequest):
    """
    Run a RAG query.

    Pipeline: embed query → vector search (top-K) → rerank (top-N) → LLM answer.
    Returns the final answer along with both raw vector results and reranked results
    so the dashboard can display the retrieval quality improvement.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Temporarily override top_k / top_n if provided
    if request.top_k:
        settings.RETRIEVAL_TOP_K = request.top_k
    if request.top_n:
        settings.RERANKER_TOP_N = request.top_n

    try:
        response: QueryResponse = get_engine().query(request.query)
        return QueryResponseOut(
            query=response.query,
            answer=response.answer,
            reranker_mode=response.reranker_mode,
            tokens_used=response.tokens_used,
            model=response.model,
            vector_results=[
                VectorResultOut(id=r.id, text=r.text, metadata=r.metadata, score=r.score)
                for r in response.vector_results
            ],
            reranked_results=[
                RankedResultOut(
                    id=r.id, text=r.text, metadata=r.metadata,
                    vector_score=r.vector_score, rerank_score=r.rerank_score, rank=r.rank,
                )
                for r in response.reranked_results
            ],
        )
    except Exception as e:
        logger.exception("Error during query")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/voice-query", tags=["Query"], response_model=QueryResponseOut)
async def voice_query(audio: UploadFile = File(...)):
    """
    Transcribe an audio file and run a RAG query on the transcription.
    Supports: mp3, wav, webm, m4a, ogg, mp4.
    """
    content = await audio.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    try:
        response: QueryResponse = get_engine().voice_query(content, audio.filename or "audio.wav")
        return QueryResponseOut(
            query=response.query,
            answer=response.answer,
            reranker_mode=response.reranker_mode,
            tokens_used=response.tokens_used,
            model=response.model,
            vector_results=[
                VectorResultOut(id=r.id, text=r.text, metadata=r.metadata, score=r.score)
                for r in response.vector_results
            ],
            reranked_results=[
                RankedResultOut(
                    id=r.id, text=r.text, metadata=r.metadata,
                    vector_score=r.vector_score, rerank_score=r.rerank_score, rank=r.rank,
                )
                for r in response.reranked_results
            ],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error during voice query")
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")
