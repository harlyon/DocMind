"""
Query service — the RAG brain of DocMind.

Supports two query modes:
  1. Single-document query  — query_document(question, document_id)
  2. Workspace query        — query_workspace(question, workspace_id, doc_map)

Both modes share the same retrieval → prompt → stream pipeline.
The workspace mode retrieves across multiple ChromaDB collections and
adds document-level attribution to every source chunk.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import AsyncGenerator

from google import genai
from google.genai import types

from core.config import get_settings
from models.schemas import SourceChunk, WorkspaceSourceChunk
from services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Gemini client — initialised once ─────────────────────────────────────────
_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        logger.info(f"Initialising Gemini client, model: {settings.llm_model}")
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are DocMind, a precise and helpful document assistant.

Your rules:
- Answer ONLY using the provided context chunks. Do not use outside knowledge.
- If the answer is not in the context, say: "I couldn't find that in the document."
- Always cite your sources using [1], [2], etc. matching the chunk numbers given.
- Be concise. Lead with the direct answer, then add supporting detail if useful.
- If multiple chunks support an answer, cite all of them: e.g. [1][3].
- Never fabricate page numbers or quotes."""

_WORKSPACE_SYSTEM_PROMPT = """You are DocMind, a precise document intelligence assistant
analysing a workspace containing multiple related documents.

Your rules:
- Answer ONLY using the provided context chunks. Do not use outside knowledge.
- Each chunk is labelled with its source document — always include the document name in citations.
- Citation format: [1] (Document name, Page X) — always include both document and page.
- If the same information appears in multiple documents, cite all relevant chunks.
- If documents contain conflicting information, explicitly note the conflict and cite both sides.
- If the answer is not in any document, say: "I couldn't find that in the workspace."
- Never fabricate page numbers, document names, or quotes."""


def _build_prompt(
    question: str,
    chunks: list[dict],
    chat_history: list[dict],
    workspace_mode: bool = False,
) -> str:
    """
    Assemble the full prompt sent to Gemini.

    In workspace mode each chunk label includes the document name:
      [1] (Contract v2.pdf, Page 4): <chunk text>

    In single-doc mode the document name is omitted:
      [1] (Page 4): <chunk text>
    """
    context_lines = ["CONTEXT"]
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.get("page_number")
        page_label = f"Page {page}" if page else "unknown page"

        if workspace_mode and chunk.get("document_name"):
            label = f"[{i}] ({chunk['document_name']}, {page_label})"
        else:
            label = f"[{i}] ({page_label})"

        context_lines.append(f"{label}: {chunk['text']}")

    context_block = "\n".join(context_lines)

    history_block = ""
    if chat_history:
        lines = ["\nCONVERSATION HISTORY"]
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_block = "\n".join(lines)

    return f"{context_block}{history_block}\n\nQUESTION\n{question}"


# ── Single-document query ─────────────────────────────────────────────────────

async def query_document(
    question: str,
    document_id: str,
    chat_history: list[dict] | None = None,
) -> tuple[AsyncGenerator[str, None], list[SourceChunk]]:
    """
    Query a single document. Behaviour unchanged from original implementation.
    """
    chat_history = chat_history or []
    vector_store = get_vector_store()

    raw_chunks = vector_store.similarity_search(
        query=question,
        document_id=document_id,
        k=settings.top_k_retrieval,
    )

    if not raw_chunks:
        raise ValueError("No relevant content found in this document for your question.")

    logger.info(
        f"[single-doc] Retrieved {len(raw_chunks)} chunks "
        f"(best score: {raw_chunks[0]['score']:.3f})"
    )

    sources = [
        SourceChunk(
            chunk_index=i + 1,
            text=c["text"],
            page=c.get("page_number"),
            score=c.get("score"),
            document_id=c["document_id"],
        )
        for i, c in enumerate(raw_chunks)
    ]

    prompt = _build_prompt(question, raw_chunks, chat_history, workspace_mode=False)
    return _stream_gemini(prompt, _SYSTEM_PROMPT), sources


# ── Workspace query ───────────────────────────────────────────────────────────

async def query_workspace(
    question: str,
    workspace_id: str,
    document_ids: list[str],
    doc_name_map: dict[str, str],
    chat_history: list[dict] | None = None,
) -> tuple[AsyncGenerator[str, None], list[WorkspaceSourceChunk]]:
    """
    Query across all documents in a workspace.

    Steps:
      1. Run similarity_search on every document's ChromaDB collection
      2. Merge all results into one list
      3. Re-rank globally by score, keep top-k
      4. Attach document name to each chunk for attribution
      5. Build prompt with document-aware citation labels
      6. Stream from Gemini

    Args:
      document_ids:  list of document IDs in the workspace
      doc_name_map:  {document_id: display_name} for citation labels
    """
    chat_history = chat_history or []
    vector_store = get_vector_store()

    # ── 1. Retrieve from every document in the workspace ─────────────────────
    all_chunks: list[dict] = []
    per_doc_k = max(2, settings.top_k_retrieval // len(document_ids))

    for doc_id in document_ids:
        try:
            chunks = vector_store.similarity_search(
                query=question,
                document_id=doc_id,
                k=per_doc_k,
            )
            # Attach the human-readable document name
            for c in chunks:
                c["document_name"] = doc_name_map.get(doc_id, doc_id[:8])
            all_chunks.extend(chunks)
        except ValueError as exc:
            # Document not yet ingested — skip gracefully
            logger.warning(f"Skipping document {doc_id}: {exc}")

    if not all_chunks:
        raise ValueError(
            "No relevant content found across any document in this workspace."
        )

    # ── 2. Global re-rank — keep best top_k across all documents ─────────────
    all_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
    top_chunks = all_chunks[: settings.top_k_retrieval]

    logger.info(
        f"[workspace] {len(document_ids)} docs → {len(all_chunks)} raw chunks "
        f"→ {len(top_chunks)} after re-rank "
        f"(best score: {top_chunks[0]['score']:.3f})"
    )

    # ── 3. Build WorkspaceSourceChunk objects ─────────────────────────────────
    sources = [
        WorkspaceSourceChunk(
            chunk_index=i + 1,
            text=c["text"],
            page=c.get("page_number"),
            score=c.get("score"),
            document_id=c["document_id"],
            document_name=c.get("document_name", "Unknown document"),
        )
        for i, c in enumerate(top_chunks)
    ]

    # ── 4. Build prompt with document-aware labels ────────────────────────────
    prompt = _build_prompt(question, top_chunks, chat_history, workspace_mode=True)
    return _stream_gemini(prompt, _WORKSPACE_SYSTEM_PROMPT), sources


# ── Shared Gemini streaming ───────────────────────────────────────────────────

async def _stream_gemini(
    prompt: str,
    system_prompt: str,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields text tokens from Gemini as they arrive.
    Accepts a system_prompt so single-doc and workspace modes can use
    different instructions without duplicating the streaming logic.
    """
    client = get_gemini_client()

    try:
        async for chunk in await client.aio.models.generate_content_stream(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                top_p=0.8,
                max_output_tokens=2048,
            ),
        ):
            if chunk.text:
                yield chunk.text

    except Exception as exc:
        logger.exception(f"Gemini generation error: {exc}")
        yield f"\n\n[Error generating response: {exc}]"