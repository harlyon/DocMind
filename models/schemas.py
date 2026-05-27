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


# ── Workspace schemas ─────────────────────────────────────────────────────────

class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    domain: str = "general"


class WorkspaceUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    domain: str | None = None


class WorkspaceDocumentResponse(BaseModel):
    id: str
    document_id: str
    display_name: str | None
    original_filename: str
    status: str
    page_count: int | None
    chunk_count: int | None
    added_at: datetime

    model_config = {"from_attributes": True}


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    description: str | None
    domain: str
    document_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class WorkspaceDetailResponse(WorkspaceResponse):
    documents: list[WorkspaceDocumentResponse]


class AddDocumentToWorkspaceRequest(BaseModel):
    document_id: str
    display_name: str | None = None


# ── Workspace query schemas ────────────────────────────────────────────────────

class WorkspaceQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    workspace_id: str
    session_id: str | None = None
    chat_history: list[dict[str, Any]] = Field(default_factory=list)


class WorkspaceSourceChunk(BaseModel):
    """Like SourceChunk but carries document-level attribution."""
    chunk_index: int
    text: str
    page: int | None = None
    score: float | None = None
    document_id: str
    document_name: str   # e.g. "Contract v2.pdf" — shown in citation panel


# ── Contradiction detection schemas ───────────────────────────────────────────

class ConflictSide(BaseModel):
    """One side of a detected contradiction."""
    document_id: str
    document_name: str
    page: int | None = None
    text: str                   # the exact passage that conflicts


class ConflictResult(BaseModel):
    """
    A single detected contradiction between two documents.
    conflict=False means the topic was found but no contradiction detected.
    """
    topic: str                  # the subject the conflict is about
    conflict: bool
    summary: str                # plain-English explanation of the conflict
    side_a: ConflictSide | None = None
    side_b: ConflictSide | None = None
    confidence: float           # 0.0 – 1.0, how confident the model is


class ContradictionRequest(BaseModel):
    workspace_id: str
    topics: list[str] = Field(
        default_factory=list,
        description=(
            "Specific topics to check for contradictions. "
            "If empty, the model identifies topics automatically."
        ),
    )


class ContradictionResponse(BaseModel):
    workspace_id: str
    topics_analysed: int
    conflicts_found: int
    results: list[ConflictResult]


# ── Due diligence checklist schemas ───────────────────────────────────────────

class ChecklistItem(BaseModel):
    id: str                     # e.g. "termination_clause"
    label: str                  # e.g. "Termination clause present"
    question: str               # what to ask the RAG system
    required: bool = True       # is this item mandatory?


class ChecklistItemResult(BaseModel):
    id: str
    label: str
    status: str                 # "pass" | "fail" | "partial" | "not_found"
    finding: str                # one-sentence summary of what was found
    evidence: list[str]         # cited passages supporting the finding
    pages: list[int]            # page numbers where evidence was found
    required: bool


class ChecklistRequest(BaseModel):
    document_id: str
    template: str | None = Field(
        None,
        description=(
            "Built-in template name: 'mna_contract', 'financial_report', "
            "'employment_contract'. If None, provide custom_items."
        ),
    )
    custom_items: list[ChecklistItem] = Field(default_factory=list)


class ChecklistResponse(BaseModel):
    document_id: str
    template: str | None
    total_items: int
    passed: int
    failed: int
    partial: int
    not_found: int
    results: list[ChecklistItemResult]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_model: str
    llm_model: str
    ocr_available: bool