"""
Layer 1a: Hybrid Retriever
Combines dense vector search with BM25 keyword search using Reciprocal Rank Fusion.
"""
from dataclasses import dataclass
from typing import List, Dict
from rank_bm25 import BM25Okapi
from app.layers.l1_retrieval.embedder import embedder
from app.layers.l1_retrieval.vector_store import vector_store, RetrievedChunk
from app.core.config import settings
import structlog

logger = structlog.get_logger()


class RAGRetriever:
    """Hybrid retriever: dense + BM25 with Reciprocal Rank Fusion."""

    async def retrieve(
        self,
        query: str,
        k: int = None,
        strategy: str = "hybrid",
    ) -> List[RetrievedChunk]:
        k = k or settings.max_retrieval_docs
        try:
            if strategy == "semantic":
                return await self._dense_search(query, k)
            elif strategy == "keyword":
                return await self._bm25_search(query, k)
            else:
                return await self._hybrid_search(query, k)
        except Exception as e:
            logger.error("retrieval_failed", error=str(e), strategy=strategy)
            return []

    async def _dense_search(self, query: str, k: int) -> List[RetrievedChunk]:
        query_embedding = await embedder.embed_query(query)
        results = vector_store.similarity_search(query_embedding, k=k)
        logger.info("dense_search_complete", results=len(results))
        return results

    async def _bm25_search(self, query: str, k: int) -> List[RetrievedChunk]:
        """BM25 over all chunks retrieved from dense search (approximate)."""
        # Get a larger pool for BM25 re-ranking
        query_embedding = await embedder.embed_query(query)
        pool = vector_store.similarity_search(query_embedding, k=min(k * 3, 30))
        if not pool:
            return []

        tokenized_corpus = [doc.content.lower().split() for doc in pool]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())

        scored = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
        results = []
        for chunk, score in scored[:k]:
            chunk.relevance_score = min(score / (max(s for _, s in scored) + 1e-9), 1.0)
            results.append(chunk)
        return results

    async def _hybrid_search(self, query: str, k: int) -> List[RetrievedChunk]:
        """RRF (Reciprocal Rank Fusion) combining dense and BM25 results."""
        query_embedding = await embedder.embed_query(query)
        dense_results = vector_store.similarity_search(query_embedding, k=k * 2)

        if not dense_results:
            return []

        # BM25 on the dense pool
        tokenized_corpus = [doc.content.lower().split() for doc in dense_results]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())

        # Build ranked lists
        dense_ranks: Dict[str, int] = {
            f"{c.doc_id}_{c.chunk_index}": i + 1
            for i, c in enumerate(dense_results)
        }
        bm25_ranked = sorted(
            zip(dense_results, bm25_scores), key=lambda x: x[1], reverse=True
        )
        bm25_ranks: Dict[str, int] = {
            f"{c.doc_id}_{c.chunk_index}": i + 1
            for i, (c, _) in enumerate(bm25_ranked)
        }

        # RRF scoring
        rrf_k = 60
        chunk_map = {f"{c.doc_id}_{c.chunk_index}": c for c in dense_results}
        rrf_scores: Dict[str, float] = {}
        for key in chunk_map:
            dr = dense_ranks.get(key, len(dense_results) + 1)
            br = bm25_ranks.get(key, len(dense_results) + 1)
            rrf_scores[key] = 1 / (rrf_k + dr) + 1 / (rrf_k + br)

        top_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

        results = []
        for key in top_keys:
            chunk = chunk_map[key]
            chunk.relevance_score = round(rrf_scores[key] / max_rrf, 4)
            results.append(chunk)

        logger.info("hybrid_retrieval_complete", results=len(results))
        return results


rag_retriever = RAGRetriever()
