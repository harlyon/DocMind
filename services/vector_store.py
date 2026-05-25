"""
Vector store service — embeds chunks and stores them in ChromaDB.

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - 80MB download, runs on CPU
  - Strong recall for English documents
  - Free — no API key required

ChromaDB:
  - Persistent (survives restarts)
  - One collection per document (easy to delete/update)
  - cosine similarity by default
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import get_settings
from services.chunker import TextChunk

logger = logging.getLogger(__name__)
settings = get_settings()

# Singleton embedding model — loaded once at startup
_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded")
    return _embedding_model


# Singleton ChromaDB client
_chroma_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Connecting to ChromaDB at: {persist_dir}")
        _chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def collection_name_for(document_id: str) -> str:
    """
    ChromaDB collection names must be 3–63 chars, alphanumeric + hyphens.
    Prefix with 'doc-' so they're easy to identify.
    """
    return f"doc-{document_id[:32]}"


class VectorStoreService:
    """Handles embedding, storing, and retrieving document chunks."""

    def __init__(self):
        self.embedder = get_embedding_model()
        self.client = get_chroma_client()

    def embed_and_store(self, chunks: list[TextChunk]) -> str:
        """
        Embed all chunks and store them in a ChromaDB collection.
        Returns the collection name.
        """
        if not chunks:
            raise ValueError("No chunks to embed")

        doc_id = chunks[0].document_id
        coll_name = collection_name_for(doc_id)

        # Delete existing collection if re-ingesting
        try:
            self.client.delete_collection(coll_name)
            logger.info(f"Deleted existing collection: {coll_name}")
        except Exception:
            pass

        collection = self.client.create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Embed in batches of 64 to avoid OOM on large docs
        batch_size = 64
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start: batch_start + batch_size]

            texts = [c.text for c in batch]
            embeddings = self.embedder.embed_documents(texts)

            collection.add(
                ids=[f"{doc_id}-{c.chunk_index}" for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "chunk_index": c.chunk_index,
                        "page_number": c.page_number,
                        "document_id": c.document_id,
                        "char_count": c.char_count,
                    }
                    for c in batch
                ],
            )
            logger.debug(
                f"Stored batch {batch_start}–{batch_start + len(batch)} "
                f"({len(batch)} chunks)"
            )

        logger.info(
            f"Stored {len(chunks)} chunks in collection '{coll_name}'"
        )
        return coll_name

    def similarity_search(
        self,
        query: str,
        document_id: str,
        k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.
        Returns list of dicts with text, metadata, and similarity score.
        """
        k = k or settings.top_k_retrieval
        coll_name = collection_name_for(document_id)

        try:
            collection = self.client.get_collection(coll_name)
        except Exception:
            raise ValueError(
                f"No vector store found for document {document_id}. "
                "Has it been ingested yet?"
            )

        query_embedding = self.embedder.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score 0→1
            similarity = 1 - (dist / 2)
            chunks.append({
                "text": doc,
                "chunk_index": meta["chunk_index"],
                "page_number": meta["page_number"],
                "document_id": meta["document_id"],
                "score": round(similarity, 4),
            })

        logger.debug(
            f"Retrieved {len(chunks)} chunks for query "
            f"(top score: {chunks[0]['score'] if chunks else 'N/A'})"
        )
        return chunks

    def delete_document(self, document_id: str) -> None:
        """Remove a document's collection from ChromaDB."""
        coll_name = collection_name_for(document_id)
        try:
            self.client.delete_collection(coll_name)
            logger.info(f"Deleted collection: {coll_name}")
        except Exception:
            logger.warning(f"Collection not found for deletion: {coll_name}")


# Singleton service instance
_vector_store: VectorStoreService | None = None


def get_vector_store() -> VectorStoreService:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store