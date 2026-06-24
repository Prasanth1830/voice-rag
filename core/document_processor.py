"""
core/document_processor.py — Document ingestion pipeline.

Supports: PDF, TXT, DOCX
Steps: parse → clean → chunk → (caller embeds and stores)
"""

from __future__ import annotations
import io
import logging
import uuid
from dataclasses import dataclass, field
from typing import Literal

from config import settings
from utils.helpers import clean_text, get_file_hash

logger = logging.getLogger(__name__)

FileType = Literal["pdf", "txt", "docx", "unknown"]


@dataclass
class DocumentChunk:
    """A single text chunk ready for embedding and storage."""
    id: str
    text: str
    metadata: dict
    embedding: list[float] = field(default_factory=list)


class DocumentProcessor:
    """
    Parses documents from raw bytes, splits them into overlapping chunks,
    and returns a list of DocumentChunk objects ready for embedding.
    """

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    # ── Public ─────────────────────────────────────────────────────────────

    def process(self, content: bytes, filename: str) -> list[DocumentChunk]:
        """
        Full pipeline: raw bytes → list of DocumentChunk.

        Args:
            content: Raw file bytes.
            filename: Original filename (used to detect file type).

        Returns:
            List of DocumentChunk objects (without embeddings yet).
        """
        file_type = self._detect_type(filename)
        file_hash = get_file_hash(content)

        logger.info(f"Processing '{filename}' ({file_type}, {len(content)} bytes)")

        text = self._extract_text(content, file_type)
        text = clean_text(text)

        if not text.strip():
            raise ValueError(f"No readable text found in '{filename}'")

        raw_chunks = self._split_text(text)

        chunks: list[DocumentChunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = str(uuid.uuid4())
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "source": filename,
                        "file_hash": file_hash,
                        "file_type": file_type,
                        "chunk_index": idx,
                        "total_chunks": len(raw_chunks),
                    },
                )
            )

        logger.info(f"Created {len(chunks)} chunks from '{filename}'")
        return chunks

    # ── Private ────────────────────────────────────────────────────────────

    def _detect_type(self, filename: str) -> FileType:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mapping: dict[str, FileType] = {"pdf": "pdf", "txt": "txt", "docx": "docx"}
        return mapping.get(ext, "unknown")

    def _extract_text(self, content: bytes, file_type: FileType) -> str:
        if file_type == "pdf":
            return self._extract_pdf(content)
        elif file_type == "docx":
            return self._extract_docx(content)
        elif file_type == "txt":
            return content.decode("utf-8", errors="ignore")
        else:
            # Attempt UTF-8 decode for unknown types
            try:
                return content.decode("utf-8")
            except Exception:
                raise ValueError(f"Unsupported file type. Supported: pdf, txt, docx")

    def _extract_pdf(self, content: bytes) -> str:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        return "\n\n".join(pages)

    def _extract_docx(self, content: bytes) -> str:
        import docx
        doc = docx.Document(io.BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks using a word-boundary approach.
        Tries to split at sentence boundaries when possible.
        """
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(words):
                break
            start += self.chunk_size - self.chunk_overlap

        return chunks
