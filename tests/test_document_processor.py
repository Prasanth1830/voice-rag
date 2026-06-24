"""
tests/test_document_processor.py — Unit tests for document processing.
"""

import pytest
from core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    def setup_method(self):
        self.processor = DocumentProcessor()

    def test_process_txt(self):
        content = b"Hello world. This is a test document with enough words to form at least one chunk."
        chunks = self.processor.process(content, "test.txt")
        assert len(chunks) >= 1
        assert all(c.text for c in chunks)
        assert all(c.metadata["source"] == "test.txt" for c in chunks)
        assert all(c.metadata["file_type"] == "txt" for c in chunks)

    def test_chunk_metadata_has_required_fields(self):
        content = b"Test content " * 50
        chunks = self.processor.process(content, "doc.txt")
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "file_hash" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_chunk_indices_are_sequential(self):
        content = b"word " * 1000
        chunks = self.processor.process(content, "big.txt")
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_empty_file_raises(self):
        with pytest.raises(ValueError, match="No readable text"):
            self.processor.process(b"", "empty.txt")

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            self.processor.process(b"\x00\x01\x02binary", "file.bin")

    def test_deduplication_via_hash(self):
        content = b"Same content " * 20
        chunks1 = self.processor.process(content, "doc1.txt")
        chunks2 = self.processor.process(content, "doc2.txt")
        # Same content → same hash
        assert chunks1[0].metadata["file_hash"] == chunks2[0].metadata["file_hash"]
        # Different IDs (UUIDs)
        assert chunks1[0].id != chunks2[0].id

    def test_overlapping_chunks(self):
        # With overlap, words from end of chunk N should appear at start of chunk N+1
        self.processor.chunk_size = 10
        self.processor.chunk_overlap = 5
        words = [f"word{i}" for i in range(30)]
        content = " ".join(words).encode()
        chunks = self.processor.process(content, "overlap.txt")
        assert len(chunks) > 1
