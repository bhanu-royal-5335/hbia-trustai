"""
Layer 1a: Embedder
Generates text embeddings using OpenAI text-embedding-3-large.
"""
from openai import AsyncOpenAI
from typing import List
from app.core.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()


class Embedder:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.batch_size = 100

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in batches."""
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=[query],
        )
        return response.data[0].embedding


embedder = Embedder()
