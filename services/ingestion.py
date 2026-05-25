"""
Ingestion service — the orchestrator for the Parse → Chunk → Embed pipeline.

Called by the /ingest endpoint. Runs as a background task so the HTTP
response returns immediately while processing happens asynchronously.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from models.db_models import Document
from services.chunker import DocumentChunker
from services.parser import get_parser
from services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()


async def ingest_document(document_id: str, db: AsyncSession) -> None:
    """
    Full ingestion pipeline for a single document.
    Updates the Document.status in the DB at each stage.
    """
    # ── 1. Fetch the document record ─────────────────────────────────────────
    doc = await db.get(Document, document_id)
    if not doc:
        logger.error(f"Document {document_id} not found in DB")
        return

    logger.info(f"Starting ingestion for document: {doc.original_filename}")

    try:
        # ── 2. Mark as processing ─────────────────────────────────────────────
        doc.status = "processing"
        doc.status_message = "Parsing document..."
        await db.commit()

        # ── 3. Parse ──────────────────────────────────────────────────────────
        parser = get_parser(doc.mime_type)
        parsed = parser.parse(doc.file_path)

        doc.page_count = parsed.total_pages
        doc.status_message = f"Chunking {parsed.total_pages} pages..."
        await db.commit()

        logger.info(
            f"Parsed '{doc.original_filename}': "
            f"{parsed.total_pages} pages, {parsed.total_chars:,} chars"
        )

        # ── 4. Chunk ──────────────────────────────────────────────────────────
        chunker = DocumentChunker()
        chunks = chunker.chunk(parsed, document_id)

        if not chunks:
            raise ValueError("No text could be extracted from this document")

        doc.chunk_count = len(chunks)
        doc.status_message = f"Embedding {len(chunks)} chunks..."
        await db.commit()

        # ── 5. Embed + store in ChromaDB ──────────────────────────────────────
        # This is the slowest step (~5–30s depending on doc size and hardware)
        vector_store = get_vector_store()
        collection_name = vector_store.embed_and_store(chunks)

        # ── 6. Mark as ready ─────────────────────────────────────────────────
        doc.collection_name = collection_name
        doc.status = "ready"
        doc.status_message = None
        await db.commit()

        logger.info(
            f"Ingestion complete for '{doc.original_filename}': "
            f"{len(chunks)} chunks in collection '{collection_name}'"
        )

    except Exception as exc:
        logger.exception(f"Ingestion failed for document {document_id}: {exc}")
        doc.status = "error"
        doc.status_message = str(exc)[:500]
        await db.commit()
        raise