"""
DocMind — AI Document Q&A SaaS
FastAPI application entrypoint.
"""
import sys
from pathlib import Path

# Ensure the backend directory is on the Python path so that
# 'from services.query import ...' works regardless of where
# uvicorn is launched from.
sys.path.insert(0, str(Path(__file__).parent))

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.database import init_db
from models.schemas import HealthResponse
from models import workspace_models as _workspace_models  # noqa: F401 — ensures tables are created
from routers import ingest, query, workspace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    logger.info(f"Starting {settings.app_name} [{settings.app_env}]")

    # Create DB tables
    await init_db()
    logger.info("Database ready")

    # Pre-load the embedding model so the first upload isn't slow
    from services.vector_store import get_embedding_model
    get_embedding_model()

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="DocMind API",
    description="AI-powered document Q&A — RAG pipeline with source citations",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(workspace.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
    )


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": "/health",
    }