"""
Ingest router — handles document uploads and ingestion.

POST /api/ingest/upload
  - Accepts a PDF or txt file
  - Saves it to disk
  - Creates a Document record in the DB
  - Kicks off the ingestion pipeline as a BackgroundTask
  - Returns immediately with {document_id, status: "pending"}

GET /api/ingest/status/{document_id}
  - Returns current ingestion status (pending/processing/ready/error)
  - The frontend polls this every 2s until status = "ready"

DELETE /api/ingest/{document_id}
  - Removes the document, its DB record, and its ChromaDB collection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
import uuid
from pathlib import Path

from core.auth import get_current_user
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.database import get_db
from models.db_models import Document
from models.schemas import DocumentListResponse, DocumentResponse, IngestResponse
from services.ingestion import ingest_document
from services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/ingest", tags=["ingest"])

ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    # TODO Day 3: replace hardcoded user_id with Clerk JWT
    user_id: str = Depends(get_current_user),
):
    """Upload a document and start the ingestion pipeline."""

    # ── Validate file type ────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(ALLOWED_TYPES.keys())}",
        )

    # ── Read and validate file size ───────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB",
        )

    # ── Save file to disk ─────────────────────────────────────────────────────
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid.uuid4())
    ext = ALLOWED_TYPES[file.content_type]
    safe_filename = f"{document_id}{ext}"
    file_path = upload_dir / safe_filename

    file_path.write_bytes(contents)
    logger.info(f"Saved upload: {file_path} ({len(contents):,} bytes)")

    # ── Create DB record ──────────────────────────────────────────────────────
    doc = Document(
        id=document_id,
        user_id=user_id,
        filename=safe_filename,
        original_filename=file.filename or "unknown",
        file_path=str(file_path),
        file_size=len(contents),
        mime_type=file.content_type,
        status="pending",
    )
    db.add(doc)
    await db.commit()

    # ── Kick off ingestion as a background task ───────────────────────────────
    # BackgroundTasks run after the response is sent.
    # The frontend polls /status/{id} every 2s to track progress.
    background_tasks.add_task(_run_ingestion, document_id)

    return IngestResponse(
        document_id=document_id,
        message=f"Upload received. Ingestion started for '{file.filename}'.",
        status="pending",
    )


@router.get("/status/{document_id}", response_model=DocumentResponse)
async def get_ingestion_status(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Poll this endpoint to track ingestion progress."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    """List all documents for a user."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return DocumentListResponse(documents=list(docs), total=len(docs))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and its vector store collection."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from ChromaDB
    get_vector_store().delete_document(document_id)

    # Remove file from disk
    try:
        os.remove(doc.file_path)
    except FileNotFoundError:
        pass

    await db.delete(doc)
    await db.commit()


async def _run_ingestion(document_id: str) -> None:
    """
    Wrapper that creates a fresh DB session for the background task.
    BackgroundTasks run outside the request lifecycle so we can't reuse
    the request's session.
    """
    from core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        await ingest_document(document_id, session)