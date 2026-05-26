import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models.db_models import Document
from models.schemas import (
    AddDocumentToWorkspaceRequest,
    WorkspaceCreate,
    WorkspaceDetailResponse,
    WorkspaceDocumentResponse,
    WorkspaceQueryRequest,
    WorkspaceResponse,
    WorkspaceUpdate,
)
from models.workspace_models import Workspace, WorkspaceDocument
from services.rag import query_workspace

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


# ── Helper ────────────────────────────────────────────────────────────────────

def _sse_event(event_type: str, data) -> str:
    return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"


async def _get_workspace_or_404(
    workspace_id: str, db: AsyncSession
) -> Workspace:
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws


# ── CRUD ──────────────────────────────────────────────────────────────────────

@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    body: WorkspaceCreate,
    db: AsyncSession = Depends(get_db),
    user_id: str = "dev-user",
):
    """Create a new workspace."""
    ws = Workspace(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name=body.name,
        description=body.description,
        domain=body.domain,
    )
    db.add(ws)
    await db.commit()
    await db.refresh(ws)

    return WorkspaceResponse(
        id=ws.id,
        name=ws.name,
        description=ws.description,
        domain=ws.domain,
        document_count=0,
        created_at=ws.created_at,
        updated_at=ws.updated_at,
    )


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    db: AsyncSession = Depends(get_db),
    user_id: str = "dev-user",
):
    """List all workspaces for the current user."""
    result = await db.execute(
        select(Workspace)
        .where(Workspace.user_id == user_id)
        .order_by(Workspace.created_at.desc())
    )
    workspaces = result.scalars().all()

    responses = []
    for ws in workspaces:
        count_result = await db.execute(
            select(WorkspaceDocument)
            .where(WorkspaceDocument.workspace_id == ws.id)
        )
        doc_count = len(count_result.scalars().all())
        responses.append(WorkspaceResponse(
            id=ws.id,
            name=ws.name,
            description=ws.description,
            domain=ws.domain,
            document_count=doc_count,
            created_at=ws.created_at,
            updated_at=ws.updated_at,
        ))

    return responses


@router.get("/{workspace_id}", response_model=WorkspaceDetailResponse)
async def get_workspace(
    workspace_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a workspace with its full document list."""
    ws = await _get_workspace_or_404(workspace_id, db)

    result = await db.execute(
        select(WorkspaceDocument)
        .where(WorkspaceDocument.workspace_id == workspace_id)
        .order_by(WorkspaceDocument.added_at)
    )
    wdocs = result.scalars().all()

    doc_responses = []
    for wd in wdocs:
        doc = await db.get(Document, wd.document_id)
        if doc:
            doc_responses.append(WorkspaceDocumentResponse(
                id=wd.id,
                document_id=wd.document_id,
                display_name=wd.display_name or doc.original_filename,
                original_filename=doc.original_filename,
                status=doc.status,
                page_count=doc.page_count,
                chunk_count=doc.chunk_count,
                added_at=wd.added_at,
            ))

    return WorkspaceDetailResponse(
        id=ws.id,
        name=ws.name,
        description=ws.description,
        domain=ws.domain,
        document_count=len(doc_responses),
        created_at=ws.created_at,
        updated_at=ws.updated_at,
        documents=doc_responses,
    )


@router.patch("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: str,
    body: WorkspaceUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update workspace name, description, or domain."""
    ws = await _get_workspace_or_404(workspace_id, db)

    if body.name is not None:
        ws.name = body.name
    if body.description is not None:
        ws.description = body.description
    if body.domain is not None:
        ws.domain = body.domain

    await db.commit()
    await db.refresh(ws)

    count_result = await db.execute(
        select(WorkspaceDocument).where(WorkspaceDocument.workspace_id == ws.id)
    )
    doc_count = len(count_result.scalars().all())

    return WorkspaceResponse(
        id=ws.id,
        name=ws.name,
        description=ws.description,
        domain=ws.domain,
        document_count=doc_count,
        created_at=ws.created_at,
        updated_at=ws.updated_at,
    )


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a workspace and its document associations.
    The underlying documents are NOT deleted — they can belong to other workspaces.
    """
    ws = await _get_workspace_or_404(workspace_id, db)
    await db.delete(ws)
    await db.commit()


# ── Document management ───────────────────────────────────────────────────────

@router.post("/{workspace_id}/documents", status_code=status.HTTP_201_CREATED)
async def add_document(
    workspace_id: str,
    body: AddDocumentToWorkspaceRequest,
    db: AsyncSession = Depends(get_db),
):
    """Add an existing document to a workspace."""
    await _get_workspace_or_404(workspace_id, db)

    doc = await db.get(Document, body.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check not already in workspace
    existing = await db.execute(
        select(WorkspaceDocument).where(
            WorkspaceDocument.workspace_id == workspace_id,
            WorkspaceDocument.document_id == body.document_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail="Document already in this workspace",
        )

    wd = WorkspaceDocument(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        document_id=body.document_id,
        display_name=body.display_name,
    )
    db.add(wd)
    await db.commit()

    return {
        "message": f"Document '{doc.original_filename}' added to workspace",
        "workspace_document_id": wd.id,
    }


@router.delete(
    "/{workspace_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_document(
    workspace_id: str,
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Remove a document from a workspace (does not delete the document itself)."""
    result = await db.execute(
        select(WorkspaceDocument).where(
            WorkspaceDocument.workspace_id == workspace_id,
            WorkspaceDocument.document_id == document_id,
        )
    )
    wd = result.scalar_one_or_none()
    if not wd:
        raise HTTPException(status_code=404, detail="Document not in this workspace")

    await db.delete(wd)
    await db.commit()


# ── Workspace query — SSE streaming ──────────────────────────────────────────

@router.post("/{workspace_id}/query")
async def query_workspace_endpoint(
    workspace_id: str,
    request: WorkspaceQueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question across all documents in a workspace.

    Returns an SSE stream identical in structure to the single-doc query:
      - sources event: list of WorkspaceSourceChunk (includes document_name)
      - token events:  streamed answer tokens
      - done event:    stream complete

    The sources event arrives before the first token so the citation panel
    renders immediately with full document attribution.
    """
    ws = await _get_workspace_or_404(workspace_id, db)

    # Load all documents in the workspace
    result = await db.execute(
        select(WorkspaceDocument)
        .where(WorkspaceDocument.workspace_id == workspace_id)
    )
    wdocs = result.scalars().all()

    if not wdocs:
        raise HTTPException(
            status_code=400,
            detail="This workspace has no documents. Add documents before querying.",
        )

    # Build doc_id list and display name map
    document_ids = []
    doc_name_map: dict[str, str] = {}
    for wd in wdocs:
        doc = await db.get(Document, wd.document_id)
        if doc and doc.status == "ready":
            document_ids.append(wd.document_id)
            doc_name_map[wd.document_id] = wd.display_name or doc.original_filename

    if not document_ids:
        raise HTTPException(
            status_code=400,
            detail="No documents in this workspace are ready for querying yet.",
        )

    async def event_stream():
        try:
            token_stream, sources = await query_workspace(
                question=request.question,
                workspace_id=workspace_id,
                document_ids=document_ids,
                doc_name_map=doc_name_map,
                chat_history=request.chat_history,
            )

            # ── Send sources first so citation panel renders immediately ──────
            sources_data = [
                {
                    "chunk_index": s.chunk_index,
                    "text": s.text,
                    "page": s.page,
                    "score": s.score,
                    "document_id": s.document_id,
                    "document_name": s.document_name,
                }
                for s in sources
            ]
            yield _sse_event("sources", sources_data)

            # ── Stream tokens ─────────────────────────────────────────────────
            async for token in token_stream:
                yield _sse_event("token", token)

            yield _sse_event("done", "")

        except ValueError as exc:
            yield _sse_event("error", str(exc))
        except Exception as exc:
            logger.exception(f"Workspace query error: {exc}")
            yield _sse_event("error", "An error occurred while generating the answer.")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Workspace-Id": workspace_id,
        },
    )