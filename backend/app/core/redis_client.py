import redis.asyncio as aioredis
from app.core.config import settings
import structlog
import json
from typing import Optional, Any

logger = structlog.get_logger()

_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
    return _redis_client


async def close_redis():
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
    logger.info("redis_closed")


class RedisCache:
    def __init__(self, redis: aioredis.Redis):
        self.redis = redis

    async def get(self, key: str) -> Optional[Any]:
        value = await self.redis.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        serialized = json.dumps(value) if not isinstance(value, str) else value
        return await self.redis.setex(key, ttl, serialized)

    async def delete(self, key: str) -> bool:
        return bool(await self.redis.delete(key))

    async def exists(self, key: str) -> bool:
        return bool(await self.redis.exists(key))

    async def incr(self, key: str, ttl: int = 60) -> int:
        pipe = self.redis.pipeline()
        await pipe.incr(key)
        await pipe.expire(key, ttl)
        results = await pipe.execute()
        return results[0]

    async def get_cache_key(self, prefix: str, *args) -> str:
        return f"{prefix}:{':'.join(str(a) for a in args)}"
