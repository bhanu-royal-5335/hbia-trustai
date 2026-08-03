from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field
from typing import List
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "HBIA-TrustAI"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    secret_key: str = "dev-secret-key-change-in-production-32chars"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # LLM Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    default_llm_provider: str = "openai"
    default_model: str = "gpt-4o"
    verification_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"

    # Database
    database_url: str = "postgresql+asyncpg://hbia:hbia_secret@localhost:5432/hbia_db"
    sync_database_url: str = "postgresql://hbia:hbia_secret@localhost:5432/hbia_db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ChromaDB
    chromadb_host: str = "localhost"
    chromadb_port: int = 8001
    chromadb_collection: str = "hbia_knowledge"

    # HBIA Configuration
    min_trust_score: float = 75.0
    max_correction_iterations: int = 3
    max_retrieval_docs: int = 6
    chunk_size: int = 512
    chunk_overlap: int = 64

    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:3001"

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    @computed_field
    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.environment == "production"


settings = Settings()
