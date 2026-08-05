"""
Layer 1a: Embedder
Generates text embeddings using OpenAI, Gemini, or local Ollama embeddings.
"""
import math
import hashlib
from typing import List
import httpx
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None
from app.core.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()


class Embedder:
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        self.model = settings.embedding_model
        self.batch_size = 100
        self.ollama_base_url = settings.ollama_base_url.rstrip("/")
        self.gemini_api_key = settings.gemini_api_key or ""

    def _hash_vector(self, text: str, dim: int = 1536) -> List[float]:
        vec = [0.0] * dim
        words = text.lower().split()
        if not words:
            return vec
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx = h % dim
            val = ((h >> 16) % 1000) / 1000.0 - 0.5
            vec[idx] += val
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec)) + 1e-10
        return [v / norm for v in vec]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in batches."""
        if settings.default_llm_provider.lower() == "ollama":
            try:
                return await self._ollama_embed_texts(texts)
            except Exception as e:
                logger.warning("ollama_embedding_failed_using_fallback", error=str(e))
                return [self._hash_vector(t) for t in texts]

        if not self.client:
            logger.info("using_local_fallback_embeddings", count=len(texts))
            return [self._hash_vector(t) for t in texts]

        try:
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            logger.info("texts_embedded", count=len(texts), model=self.model)
            return all_embeddings
        except Exception as e:
            logger.warning("openai_embedding_failed_using_fallback", error=str(e))
            return [self._hash_vector(t) for t in texts]

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        if settings.default_llm_provider.lower() == "ollama":
            try:
                return await self._ollama_embed_text(query)
            except Exception as e:
                logger.warning("ollama_query_embedding_failed_using_fallback", error=str(e))
                return self._hash_vector(query)

        if not self.client:
            return self._hash_vector(query)

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=[query],
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning("openai_query_embedding_failed_using_fallback", error=str(e))
            return self._hash_vector(query)

    async def _ollama_embed_texts(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            embeddings = []
            for text in texts:
                response = await client.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                payload = response.json()
                embeddings.append(payload["embedding"])
            return embeddings

    async def _ollama_embed_text(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            payload = response.json()
            return payload["embedding"]


embedder = Embedder()
