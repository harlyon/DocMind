"""
Query router — RAG question answering with Gemini 2.5 Flash streaming.

POST /api/query/ask
  - Accepts {question, document_id, session_id?, chat_history?}
  - Returns a Server-Sent Events (SSE) stream
  - First event:  {"type": "sources", "data": [...]}  ← citation panel
  - Middle events: {"type": "token", "data": "..."}   ← streamed answer tokens
  - Final event:  {"type": "done", "data": ""}        ← stream complete

GET /api/query/sessions/{document_id}
  - Lists all chat sessions for a document

GET /api/query/history/{session_id}
  - Returns full message history for a session

Why SSE over WebSockets?
  SSE is unidirectional (server → client) and works over plain HTTP/1.1.
  No upgrade handshake needed, works through proxies, and is natively
  supported by the browser EventSource API. Perfect for token streaming.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import uuid

from core.auth import get_current_user
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models.db_models import ChatMessage, ChatSession, Document
from models.schemas import QueryRequest, QueryResponse, SourceChunk
from services.rag import query_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/query", tags=["query"])


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_event(event_type: str, data) -> str:
    """Format a single SSE event string."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"


# ── Main query endpoint ───────────────────────────────────────────────────────

@router.post("/ask")
async def ask_question(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    # TODO Day 3: replace with Clerk JWT user_id
    user_id: str = Depends(get_current_user),
):
    """
    Ask a question about a document. Returns an SSE stream.

    Frontend usage (React):
      const source = new EventSource('/api/query/ask');
      source.onmessage = (e) => {
        const { type, data } = JSON.parse(e.data);
        if (type === 'sources') showCitations(data);
        if (type === 'token')   appendToken(data);
        if (type === 'done')    source.close();
      };
    """
    # ── Validate document exists and is ready ────────────────────────────────
    doc = await db.get(Document, request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Document is not ready for querying (status: {doc.status})"
        )

    # ── Resolve or create chat session ───────────────────────────────────────
    session_id = request.session_id
    if session_id:
        session = await db.get(ChatSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        session_id = str(uuid.uuid4())
        session = ChatSession(
            id=session_id,
            document_id=request.document_id,
            user_id=user_id,
            # Use first 60 chars of question as session title
            title=request.question[:60],
        )
        db.add(session)
        await db.commit()

    # ── Save the user message ─────────────────────────────────────────────────
    user_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=request.question,
    )
    db.add(user_msg)
    await db.commit()

    # ── Build the SSE stream ──────────────────────────────────────────────────
    async def event_stream():
        full_answer = []
        sources: list[SourceChunk] = []

        try:
            # Get the token stream and sources from the query service
            token_stream, sources = await query_document(
                question=request.question,
                document_id=request.document_id,
                chat_history=request.chat_history,
            )

            # ── Event 1: send sources immediately so the citation panel
            #             renders before the answer starts streaming
            sources_data = [
                {
                    "chunk_index": s.chunk_index,
                    "text": s.text,
                    "page": s.page,
                    "score": s.score,
                    "document_id": s.document_id,
                }
                for s in sources
            ]
            yield _sse_event("sources", sources_data)

            # ── Events 2…N: stream tokens as they arrive from Gemini
            async for token in token_stream:
                full_answer.append(token)
                yield _sse_event("token", token)

            # ── Final event: signal stream completion
            yield _sse_event("done", "")

        except ValueError as exc:
            # e.g. no relevant chunks found
            yield _sse_event("error", str(exc))
            return

        except Exception as exc:
            logger.exception(f"Streaming error: {exc}")
            yield _sse_event("error", "An error occurred while generating the answer.")
            return

        finally:
            # ── Persist the assistant message to DB ───────────────────────────
            # This runs whether or not the stream completed successfully
            if full_answer:
                import json as _json
                assistant_msg = ChatMessage(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="assistant",
                    content="".join(full_answer),
                    sources=_json.dumps(sources_data) if sources else None,
                )
                # Need a fresh session since we're outside the request lifecycle
                from core.database import AsyncSessionLocal
                async with AsyncSessionLocal() as save_session:
                    save_session.add(assistant_msg)
                    await save_session.commit()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Prevent buffering — critical for streaming to work correctly
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # Nginx: disable proxy buffering
            "X-Session-Id": session_id,   # Expose session ID to the frontend
        },
    )


# ── Chat history endpoints ────────────────────────────────────────────────────

@router.get("/sessions/{document_id}")
async def list_sessions(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    """List all chat sessions for a document."""
    result = await db.execute(
        select(ChatSession)
        .where(
            ChatSession.document_id == document_id,
            ChatSession.user_id == user_id,
        )
        .order_by(ChatSession.created_at.desc())
    )
    sessions = result.scalars().all()
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions
        ]
    }


@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return the full message history for a chat session."""
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    messages = result.scalars().all()

    return {
        "session_id": session_id,
        "document_id": session.document_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": json.loads(m.sources) if m.sources else None,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }