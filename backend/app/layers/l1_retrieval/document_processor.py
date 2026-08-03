"""
Layer 1a: RAG Retrieval Pipeline
Handles document processing, embedding, vector storage, and hybrid retrieval.
"""
import io
import re
from dataclasses import dataclass, field
from typing import List, Optional
import structlog
import pypdf
import docx2txt

logger = structlog.get_logger()


@dataclass
class DocumentChunk:
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0
    doc_id: Optional[str] = None


class DocumentProcessor:
    """Process uploaded documents into chunks for embedding."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, file_bytes: bytes, filename: str, doc_id: str) -> List[DocumentChunk]:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            text = self._extract_pdf(file_bytes)
        elif ext in ("docx", "doc"):
            text = self._extract_docx(file_bytes)
        else:
            text = file_bytes.decode("utf-8", errors="ignore")

        text = self._clean_text(text)
        chunks = self._chunk_text(text)

        return [
            DocumentChunk(
                content=chunk,
                metadata={"filename": filename, "doc_id": doc_id, "chunk_index": i},
                chunk_index=i,
                doc_id=doc_id,
            )
            for i, chunk in enumerate(chunks)
        ]

    def _extract_pdf(self, file_bytes: bytes) -> str:
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            return "\n\n".join(pages)
        except Exception as e:
            logger.error("pdf_extraction_failed", error=str(e))
            return ""

    def _extract_docx(self, file_bytes: bytes) -> str:
        try:
            return docx2txt.process(io.BytesIO(file_bytes))
        except Exception as e:
            logger.error("docx_extraction_failed", error=str(e))
            return ""

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end >= len(words):
                break
            start = end - self.chunk_overlap
        return chunks
