"""
Query service — the RAG brain of DocMind.

Flow per user question:
  1. Embed the query           (same all-MiniLM-L6-v2 model used at ingest)
  2. Retrieve top-k chunks     (ChromaDB cosine similarity)
  3. Build a citation prompt   (chunks labelled [1], [2] … [k])
  4. Stream from Gemini 2.5    (yields text tokens as they arrive)
  5. Return sources alongside  (page numbers + scores for the citation panel)

Uses the new google-genai SDK (google.genai) — the old google.generativeai
package is deprecated as of 2025.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import AsyncGenerator

from google import genai
from google.genai import types

from core.config import get_settings
from models.schemas import SourceChunk
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


def _build_prompt(question: str, chunks: list[dict], chat_history: list[dict]) -> str:
    """
    Assemble the full prompt sent to Gemini.

    Structure:
      CONTEXT
      [1] (Page X): <chunk text>
      [2] (Page Y): <chunk text>
      ...

      CONVERSATION HISTORY (if multi-turn)
      User: ...
      Assistant: ...

      QUESTION
      <user question>
    """
    # ── Context block ─────────────────────────────────────────────────────────
    context_lines = ["CONTEXT"]
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.get("page_number")
        page_label = f"Page {page}" if page else "unknown page"
        context_lines.append(f"[{i}] ({page_label}): {chunk['text']}")

    context_block = "\n".join(context_lines)

    # ── Optional chat history (Day 5 multi-turn) ──────────────────────────────
    history_block = ""
    if chat_history:
        lines = ["\nCONVERSATION HISTORY"]
        for msg in chat_history[-6:]:   # last 3 turns to keep prompt lean
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_block = "\n".join(lines)

    return f"{context_block}{history_block}\n\nQUESTION\n{question}"


# ── Main query function ───────────────────────────────────────────────────────

async def query_document(
    question: str,
    document_id: str,
    chat_history: list[dict] | None = None,
) -> tuple[AsyncGenerator[str, None], list[SourceChunk]]:
    """
    Retrieve relevant chunks and stream an answer from Gemini.

    Returns:
      (token_stream, sources)
      - token_stream: async generator yielding text tokens as they arrive
      - sources: list of SourceChunk for the citation panel (immediately available)
    """
    chat_history = chat_history or []

    # ── 1. Retrieve relevant chunks ───────────────────────────────────────────
    vector_store = get_vector_store()
    raw_chunks = vector_store.similarity_search(
        query=question,
        document_id=document_id,
        k=settings.top_k_retrieval,
    )

    if not raw_chunks:
        raise ValueError("No relevant content found in this document for your question.")

    logger.info(
        f"Retrieved {len(raw_chunks)} chunks for query "
        f"(best score: {raw_chunks[0]['score']:.3f})"
    )

    # ── 2. Build SourceChunk objects for the citation panel ───────────────────
    sources = [
        SourceChunk(
            chunk_index=i + 1,      # 1-indexed to match [1],[2] in the answer
            text=c["text"],
            page=c.get("page_number"),
            score=c.get("score"),
            document_id=c["document_id"],
        )
        for i, c in enumerate(raw_chunks)
    ]

    # ── 3. Build the prompt ───────────────────────────────────────────────────
    prompt = _build_prompt(question, raw_chunks, chat_history)
    logger.debug(f"Prompt length: {len(prompt)} chars")

    # ── 4. Return (stream generator, sources) ─────────────────────────────────
    token_stream = _stream_gemini(prompt)
    return token_stream, sources


async def _stream_gemini(prompt: str) -> AsyncGenerator[str, None]:
    """
    Async generator that yields text tokens from Gemini as they arrive.

    The new google-genai SDK uses client.aio.models.generate_content_stream()
    for async streaming. Each chunk.text is a partial response token.
    """
    client = get_gemini_client()

    try:
        async for chunk in await client.aio.models.generate_content_stream(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.2,        # Low = factual, grounded answers
                top_p=0.8,
                max_output_tokens=2048,
            ),
        ):
            if chunk.text:
                yield chunk.text

    except Exception as exc:
        logger.exception(f"Gemini generation error: {exc}")
        yield f"\n\n[Error generating response: {exc}]"