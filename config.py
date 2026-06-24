"""
config.py — Application configuration via environment variables.
Uses Pydantic BaseSettings for type-safe config with .env support.
"""

from pydantic_settings import BaseSettings
from typing import Literal
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────
    APP_NAME: str = "RAG Voice Boilerplate"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── OpenAI ────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"
    OPENAI_WHISPER_MODEL: str = "whisper-1"

    # ── Vector Store ──────────────────────────────────────────────────────
    # Options: "chroma" (local, no API key) | "pinecone" (cloud)
    VECTOR_STORE: Literal["chroma", "pinecone"] = "chroma"

    # ChromaDB (local)
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # Pinecone (cloud, optional)
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = "rag-index"

    # ── Retrieval ─────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 20        # chunks fetched from vector store
    RERANKER_TOP_N: int = 5          # top chunks after reranking

    # ── Re-ranker ─────────────────────────────────────────────────────────
    # Options: "cohere" | "local" | "none"
    RERANKER_MODE: Literal["cohere", "local", "none"] = "local"

    # Cohere Rerank (cloud, optional)
    COHERE_API_KEY: str = ""
    COHERE_RERANK_MODEL: str = "rerank-english-v3.0"

    # Local cross-encoder model (sentence-transformers)
    LOCAL_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Document Processing ───────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE_MB: int = 50

    # ── CORS ──────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


settings = get_settings()
