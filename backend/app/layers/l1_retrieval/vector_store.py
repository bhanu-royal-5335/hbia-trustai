"""
Layer 1a: ChromaDB Vector Store
Manages vector embeddings storage and similarity search.
"""
try:
    import chromadb
except Exception:
    chromadb = None
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from app.core.config import settings
import structlog

logger = structlog.get_logger()


@dataclass
class RetrievedChunk:
    content: str
    doc_id: str
    filename: str
    chunk_index: int
    relevance_score: float
    page: Optional[int] = None
    metadata: Dict[str, Any] = None


class VectorStore:
    def __init__(self):
        self._client = None
        self._collection = None
        self._is_fallback = False
        self._fallback_chunks = []      # List[str]
        self._fallback_embeddings = []  # List[List[float]]
        self._fallback_metadatas = []   # List[Dict]
        self._fallback_ids = []         # List[str]

    async def initialize(self):
        try:
            import chromadb
            try:
                self._client = chromadb.HttpClient(
                    host=settings.chromadb_host,
                    port=settings.chromadb_port,
                )
                self._collection = self._client.get_or_create_collection(
                    name=settings.chromadb_collection,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("chromadb_initialized", collection=settings.chromadb_collection)
            except Exception as e:
                logger.warning("chromadb_http_unavailable_trying_local", error=str(e))
                self._client = chromadb.Client()
                self._collection = self._client.get_or_create_collection(
                    name=settings.chromadb_collection,
                    metadata={"hnsw:space": "cosine"},
                )
        except Exception as e:
            logger.warning("chromadb_unavailable_using_inmemory_vector_store", error=str(e))
            self._is_fallback = True

    def add_documents(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ):
        if not chunks:
            return
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        if not self._is_fallback and self._collection:
            self._collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            self._fallback_ids.extend(ids)
            self._fallback_chunks.extend(chunks)
            self._fallback_embeddings.extend(embeddings)
            self._fallback_metadatas.extend(metadatas)
        logger.info("chunks_added", doc_id=doc_id, count=len(chunks))

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 6,
        where: Optional[Dict] = None,
    ) -> List[RetrievedChunk]:
        if self._is_fallback or not self._collection:
            import numpy as np
            if not self._fallback_chunks:
                return []
            q_vec = np.array(query_embedding, dtype=np.float32)
            norm_q = float(np.linalg.norm(q_vec)) + 1e-10
            scores = []
            for i, emb in enumerate(self._fallback_embeddings):
                meta = self._fallback_metadatas[i] if i < len(self._fallback_metadatas) else {}
                if where and any(meta.get(k_) != v_ for k_, v_ in where.items()):
                    continue
                e_vec = np.array(emb, dtype=np.float32)
                sim = float(np.dot(q_vec, e_vec) / (norm_q * (float(np.linalg.norm(e_vec)) + 1e-10)))
                scores.append((sim, i))
            scores.sort(key=lambda x: x[0], reverse=True)
            top_k = scores[:k]
            retrieved = []
            for sim, i in top_k:
                meta = self._fallback_metadatas[i] if i < len(self._fallback_metadatas) else {}
                retrieved.append(RetrievedChunk(
                    content=self._fallback_chunks[i],
                    doc_id=meta.get("doc_id", ""),
                    filename=meta.get("filename", "Unknown"),
                    chunk_index=meta.get("chunk_index", i),
                    relevance_score=round(max(0.0, sim), 4),
                    page=meta.get("page"),
                    metadata=meta,
                ))
            return retrieved

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self._collection.count() or 1),
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            query_params["where"] = where

        try:
            results = self._collection.query(**query_params)
        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            return []

        retrieved = []
        if not results["ids"] or not results["ids"][0]:
            return []

        for i, (doc_id_chunk, doc, dist, meta) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )):
            score = max(0.0, 1.0 - dist)
            retrieved.append(RetrievedChunk(
                content=doc,
                doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", "Unknown"),
                chunk_index=meta.get("chunk_index", i),
                relevance_score=round(score, 4),
                page=meta.get("page"),
                metadata=meta,
            ))
        return retrieved

    def delete_document(self, doc_id: str):
        if self._is_fallback or not self._collection:
            indices_to_remove = [
                i for i, meta in enumerate(self._fallback_metadatas)
                if meta.get("doc_id") == doc_id
            ]
            for i in reversed(indices_to_remove):
                del self._fallback_ids[i]
                del self._fallback_chunks[i]
                del self._fallback_embeddings[i]
                del self._fallback_metadatas[i]
            logger.info("document_deleted", doc_id=doc_id, chunks=len(indices_to_remove))
            return
        results = self._collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info("document_deleted", doc_id=doc_id, chunks=len(results["ids"]))

    def get_stats(self) -> Dict:
        count = len(self._fallback_chunks) if (self._is_fallback or not self._collection) else self._collection.count()
        return {
            "total_chunks": count,
            "collection_name": settings.chromadb_collection,
        }


# Singleton instance
vector_store = VectorStore()
