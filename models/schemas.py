from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Document schemas ──────────────────────────────────────────────────────────

class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    file_size: int
    mime_type: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    status: str
    status_message: str | None
    page_count: int | None
    chunk_count: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


# ── Ingest schemas ────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str
    message: str
    status: str


# ── Chat / Query schemas ──────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    chunk_index: int
    text: str
    page: int | None = None
    score: float | None = None
    document_id: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_id: str
    session_id: str | None = None
    # Include last N messages for multi-turn context (Day 5 feature)
    chat_history: list[dict[str, Any]] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    session_id: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_model: str
    llm_model: str