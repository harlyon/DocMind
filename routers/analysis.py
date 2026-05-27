"""
Analysis router — Features 2 and 3.

POST /api/analysis/contradictions
  Detects contradictions across all documents in a workspace.
  Accepts optional topic list — auto-detects topics if none provided.
  Returns structured ConflictResult list with cited evidence from both sides.

POST /api/analysis/checklist
  Runs a due diligence checklist against a single document.
  Accepts a built-in template name OR a custom list of checklist items.
  Returns structured pass/fail/partial results with cited evidence and page numbers.

GET /api/analysis/templates
  Lists all available built-in checklist templates.

Both endpoints are synchronous (not streaming) because:
  - The frontend renders results as a complete table/panel, not token by token
  - Structured JSON output is more important than perceived speed here
  - Results are cached-friendly (same doc + same checklist = same result)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models.db_models import Document
from models.schemas import (
    ChecklistRequest,
    ChecklistResponse,
    ContradictionRequest,
    ContradictionResponse,
)
from models.workspace_models import Workspace, WorkspaceDocument
from services.checklist import CHECKLIST_TEMPLATES, get_template, run_checklist
from services.conflict import detect_contradictions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["analysis"])


# ── Contradiction detection ───────────────────────────────────────────────────

@router.post("/contradictions", response_model=ContradictionResponse)
async def find_contradictions(
    request: ContradictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Detect contradictions across all documents in a workspace.

    Checks every document pair against every topic.
    Topics are auto-detected from document names if not provided.

    Example request:
    {
      "workspace_id": "abc-123",
      "topics": ["Revenue figures", "Termination conditions"]
    }

    Leave topics empty for automatic topic detection:
    {
      "workspace_id": "abc-123",
      "topics": []
    }
    """
    # ── Load workspace ────────────────────────────────────────────────────────
    ws = await db.get(Workspace, request.workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # ── Load ready documents ──────────────────────────────────────────────────
    result = await db.execute(
        select(WorkspaceDocument)
        .where(WorkspaceDocument.workspace_id == request.workspace_id)
    )
    wdocs = result.scalars().all()

    if len(wdocs) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Contradiction detection requires at least 2 documents. "
                f"This workspace has {len(wdocs)}."
            ),
        )

    document_ids = []
    doc_name_map: dict[str, str] = {}

    for wd in wdocs:
        doc = await db.get(Document, wd.document_id)
        if doc and doc.status == "ready":
            document_ids.append(wd.document_id)
            doc_name_map[wd.document_id] = wd.display_name or doc.original_filename

    if len(document_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 documents must be fully ingested before running contradiction analysis.",
        )

    # ── Run contradiction detection ───────────────────────────────────────────
    logger.info(
        f"Starting contradiction analysis: workspace={request.workspace_id}, "
        f"{len(document_ids)} docs, {len(request.topics)} user topics"
    )

    try:
        results = await detect_contradictions(
            workspace_id=request.workspace_id,
            document_ids=document_ids,
            doc_name_map=doc_name_map,
            topics=request.topics,
            domain=ws.domain,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception(f"Contradiction detection error: {exc}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")

    conflicts_found = sum(1 for r in results if r.conflict)

    return ContradictionResponse(
        workspace_id=request.workspace_id,
        topics_analysed=len(results),
        conflicts_found=conflicts_found,
        results=results,
    )


# ── Due diligence checklist ───────────────────────────────────────────────────

@router.post("/checklist", response_model=ChecklistResponse)
async def run_due_diligence(
    request: ChecklistRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Run a due diligence checklist against a document.

    Use a built-in template:
    {
      "document_id": "abc-123",
      "template": "mna_contract"
    }

    Or supply custom checklist items:
    {
      "document_id": "abc-123",
      "custom_items": [
        {
          "id": "my_check",
          "label": "Force majeure clause present",
          "question": "Is there a force majeure clause?",
          "required": true
        }
      ]
    }
    """
    # ── Validate document ─────────────────────────────────────────────────────
    doc = await db.get(Document, request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Document is not ready (status: {doc.status}). Wait for ingestion to complete.",
        )

    # ── Resolve checklist items ────────────────────────────────────────────────
    if request.template:
        try:
            items = get_template(request.template)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    elif request.custom_items:
        items = request.custom_items
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a template name or custom_items.",
        )

    # ── Run checklist ─────────────────────────────────────────────────────────
    logger.info(
        f"Running checklist: document={doc.original_filename}, "
        f"template={request.template or 'custom'}, "
        f"{len(items)} items"
    )

    try:
        results = await run_checklist(
            document_id=request.document_id,
            items=items,
        )
    except Exception as exc:
        logger.exception(f"Checklist error: {exc}")
        raise HTTPException(status_code=500, detail="Checklist analysis failed. Please try again.")

    # ── Aggregate summary counts ──────────────────────────────────────────────
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    partial = sum(1 for r in results if r.status == "partial")
    not_found = sum(1 for r in results if r.status == "not_found")

    return ChecklistResponse(
        document_id=request.document_id,
        template=request.template,
        total_items=len(results),
        passed=passed,
        failed=failed,
        partial=partial,
        not_found=not_found,
        results=results,
    )


# ── Template discovery ────────────────────────────────────────────────────────

@router.get("/templates")
async def list_templates():
    """
    List all available built-in checklist templates.
    Returns template names with item counts and labels.
    """
    return {
        "templates": [
            {
                "name": name,
                "item_count": len(items),
                "labels": [i.label for i in items],
            }
            for name, items in CHECKLIST_TEMPLATES.items()
        ]
    }