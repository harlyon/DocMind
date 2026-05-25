"""
PDF parsing service using PyMuPDF (fitz).

Why PyMuPDF over pdfplumber or PyPDF2?
  - Fastest of the three for text extraction
  - Best at preserving reading order in multi-column layouts
  - Returns page numbers natively — critical for source citations
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.x
except ImportError:
    import fitz  # PyMuPDF legacy import name

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed
    text: str
    char_count: int


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    total_pages: int
    total_chars: int
    metadata: dict

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


class PDFParser:
    """Extracts text from PDFs, preserving page boundaries for citation."""

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info(f"Parsing PDF: {path.name}")

        pages: list[ParsedPage] = []

        with fitz.open(str(path)) as doc:
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
            }

            for page_num in range(len(doc)):
                page = doc[page_num]

                # extract_text() with "text" mode preserves reading order
                text = page.get_text("text")
                text = self._clean_text(text)

                pages.append(ParsedPage(
                    page_number=page_num + 1,
                    text=text,
                    char_count=len(text),
                ))

        total_chars = sum(p.char_count for p in pages)
        logger.info(f"Parsed {len(pages)} pages, {total_chars:,} characters")

        return ParsedDocument(
            pages=pages,
            total_pages=len(pages),
            total_chars=total_chars,
            metadata=metadata,
        )

    def _clean_text(self, text: str) -> str:
        """Remove common PDF artifacts while keeping structure."""
        # Collapse multiple blank lines to a single blank line
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove hyphenation at line breaks (common in PDFs)
        text = re.sub(r"-\n(\w)", r"\1", text)
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)
        return text.strip()


class TextParser:
    """Fallback parser for plain .txt files."""

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace")

        # Treat every ~2000 chars as a "page" for citation purposes
        chunk_size = 2000
        pages = []
        for i, start in enumerate(range(0, len(text), chunk_size)):
            pages.append(ParsedPage(
                page_number=i + 1,
                text=text[start:start + chunk_size],
                char_count=min(chunk_size, len(text) - start),
            ))

        return ParsedDocument(
            pages=pages or [ParsedPage(1, text, len(text))],
            total_pages=len(pages) or 1,
            total_chars=len(text),
            metadata={},
        )


def get_parser(mime_type: str):
    """Factory: return the right parser for the file type."""
    if mime_type == "application/pdf":
        return PDFParser()
    if mime_type in ("text/plain", "text/markdown"):
        return TextParser()
    raise ValueError(f"Unsupported file type: {mime_type}")