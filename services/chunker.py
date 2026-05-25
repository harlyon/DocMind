"""
Chunking service.

Strategy: RecursiveCharacterTextSplitter from LangChain.

Why recursive over simple fixed-size?
  - Tries to split on paragraphs first, then sentences, then words
  - Results in semantically coherent chunks (a paragraph stays together
    if it fits) rather than cutting mid-sentence
  - Overlap ensures context is not lost at chunk boundaries

Each chunk keeps metadata:
  - page_number  → shown in the citation panel
  - chunk_index  → used to reference [1], [2] in the answer
  - document_id  → ties the chunk back to its document in ChromaDB
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import get_settings
from services.parser import ParsedDocument

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    page_number: int
    document_id: str
    char_count: int


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # Try these separators in order — stops at the first that fits
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, document: ParsedDocument, document_id: str) -> list[TextChunk]:
        """
        Split a parsed document into overlapping chunks.
        Preserves the page number of the page where each chunk starts.
        """
        all_chunks: list[TextChunk] = []
        chunk_index = 0

        for page in document.pages:
            if not page.text.strip():
                continue

            # Split this page's text
            page_chunks = self.splitter.split_text(page.text)

            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue

                all_chunks.append(TextChunk(
                    text=chunk_text.strip(),
                    chunk_index=chunk_index,
                    page_number=page.page_number,
                    document_id=document_id,
                    char_count=len(chunk_text),
                ))
                chunk_index += 1

        logger.info(
            f"Chunked document {document_id}: "
            f"{document.total_pages} pages → {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks