"""
utils/helpers.py — Common utility functions used across the application.
"""

import hashlib
import logging
import time
from typing import Any
from functools import wraps

logger = logging.getLogger(__name__)


def get_file_hash(content: bytes) -> str:
    """Generate a SHA-256 hash of file content for deduplication."""
    return hashlib.sha256(content).hexdigest()


def chunk_list(lst: list, n: int) -> list[list]:
    """Split a list into chunks of size n."""
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length characters."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def timeit(func):
    """Decorator to measure and log execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters from text."""
    import re
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_rag_prompt(query: str, context_chunks: list[str], system_prompt: str | None = None) -> list[dict[str, Any]]:
    """
    Build the messages list for the OpenAI Chat API.

    Args:
        query: The user's question.
        context_chunks: List of retrieved (and reranked) text chunks.
        system_prompt: Optional override for the system message.

    Returns:
        List of message dicts for openai.chat.completions.create.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question using ONLY "
            "the provided context. If the answer is not found in the context, say "
            "'I don't have enough information to answer that.' Be concise and accurate."
        )

    context_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    user_message = f"""Context:
{context_text}

Question: {query}

Answer:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def format_chunk_metadata(metadata: dict) -> dict:
    """Ensure chunk metadata has all required fields with defaults."""
    return {
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page", 0),
        "chunk_index": metadata.get("chunk_index", 0),
        "file_hash": metadata.get("file_hash", ""),
        "file_type": metadata.get("file_type", ""),
        "total_chunks": metadata.get("total_chunks", 0),
    }
