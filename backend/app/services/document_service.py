from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.document import Document
from app.layers.l1_retrieval.document_processor import DocumentProcessor
from app.layers.l1_retrieval.embedder import embedder
from app.layers.l1_retrieval.vector_store import vector_store
from app.core.config import settings
import structlog

logger = structlog.get_logger()


class DocumentService:
    def __init__(self):
        self.processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    async def get_document(self, db: AsyncSession, doc_id: int, user_id: int) -> Optional[Document]:
        result = await db.execute(
            select(Document).where(Document.id == doc_id, Document.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def process_document(
        self,
        db: AsyncSession,
        doc_id: int,
        doc_uuid: str,
        file_bytes: bytes,
        original_filename: str,
    ):
        try:
            logger.info("processing_document_started", doc_id=doc_id)
            
            # Step 1: Process into chunks
            chunks = self.processor.process(file_bytes, original_filename, doc_uuid)
            if not chunks:
                raise ValueError("No text extracted from document")

            # Step 2: Embed chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = await embedder.embed_texts(texts)

            # Step 3: Store in vector database
            metadatas = [chunk.metadata for chunk in chunks]
            vector_store.add_documents(doc_uuid, texts, embeddings, metadatas)

            # Step 4: Update database status
            doc = await db.get(Document, doc_id)
            if doc:
                doc.status = "ready"
                doc.chunk_count = len(chunks)
                await db.commit()
                
            logger.info("processing_document_complete", doc_id=doc_id, chunks=len(chunks))

        except Exception as e:
            logger.error("processing_document_failed", doc_id=doc_id, error=str(e))
            doc = await db.get(Document, doc_id)
            if doc:
                doc.status = "error"
                doc.error_message = str(e)[:500]
                await db.commit()

    async def delete_document(self, db: AsyncSession, doc: Document):
        # Remove from vector store
        vector_store.delete_document(doc.filename)
        
        # Remove from relational DB
        await db.delete(doc)
        await db.commit()


document_service = DocumentService()
