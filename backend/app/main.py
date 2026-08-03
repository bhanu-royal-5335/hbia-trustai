import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.router import api_router
from app.core.database import init_db

setup_logging()
logger = structlog.get_logger()

app = FastAPI(
    title=settings.project_name,
    version="1.0.0",
    openapi_url=f"{settings.api_v1_str}/openapi.json",
    description="Hierarchical Bounded Intelligence Architecture (HBIA) for Trustworthy Generative AI",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_str)

@app.on_event("startup")
async def on_startup():
    logger.info("Starting up HBIA TrustAI backend...")
    await init_db()

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down HBIA TrustAI backend...")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
