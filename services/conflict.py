"""
Contradiction detection service — Feature 2.

Strategy:
  For each topic (user-supplied or auto-detected):
    1. Retrieve the top-k chunks about that topic from EACH document
    2. Build a comparison prompt showing both sides to Gemini
    3. Ask Gemini: "Do these passages contradict each other?"
    4. Parse the structured JSON response
    5. Return ConflictResult with both sides cited

Why a separate service from rag.py?
  The RAG query answers a question. The conflict detector does something
  fundamentally different — it compares passages across documents and
  reasons about whether they agree. Different prompt structure, different
  output shape, different Gemini instructions.

Why JSON output?
  The frontend renders conflicts as a structured split-view panel —
  left side vs right side with highlighted passages. That requires
  structured data, not prose. We instruct Gemini to return only JSON
  and parse it strictly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging

from google import genai
from google.genai import types

from core.config import get_settings
from models.schemas import ConflictResult, ConflictSide
from services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Prompt ────────────────────────────────────────────────────────────────────

_CONFLICT_SYSTEM_PROMPT = """You are a precise document analyst specialising in
detecting contradictions between legal and financial documents.

You will be given passages about the same topic from two different documents.
Your job is to determine whether the passages contradict each other.

You MUST respond with valid JSON only — no preamble, no markdown fences, no explanation outside the JSON.

Response schema:
{
  "conflict": true | false,
  "summary": "One clear sentence explaining the conflict, or 'No contradiction detected.'",
  "confidence": 0.0 to 1.0,
  "side_a_text": "The specific phrase or sentence from Document A that conflicts",
  "side_b_text": "The specific phrase or sentence from Document B that conflicts"
}

Rules:
- conflict=true only when the documents make directly opposing claims on the same point
- Differences in emphasis, detail level, or scope are NOT contradictions
- Missing information in one document is NOT a contradiction
- confidence reflects how clearly contradictory the passages are (1.0 = undeniable conflict)
- side_a_text and side_b_text should be short, direct quotes from the passages provided
- If conflict=false, set side_a_text and side_b_text to null"""


def _build_conflict_prompt(
    topic: str,
    doc_a_name: str,
    doc_a_chunks: list[dict],
    doc_b_name: str,
    doc_b_chunks: list[dict],
) -> str:
    """Build the comparison prompt for a single topic across two documents."""

    def format_chunks(chunks: list[dict]) -> str:
        return "\n\n".join(
            f"[Page {c.get('page_number', '?')}]: {c['text']}"
            for c in chunks
        )

    return f"""TOPIC: {topic}

DOCUMENT A — {doc_a_name}:
{format_chunks(doc_a_chunks)}

DOCUMENT B — {doc_b_name}:
{format_chunks(doc_b_chunks)}

Do these passages from Document A and Document B contradict each other on the topic "{topic}"?
Respond with JSON only."""


# ── Auto topic detection ──────────────────────────────────────────────────────

_TOPIC_DETECTION_PROMPT = """You are analysing a set of documents to identify
topics that should be checked for contradictions.

Given these document names, suggest 5-8 specific topics that are likely to appear
in all documents and where contradictions would be meaningful and important.

Documents: {doc_names}
Domain: {domain}

Respond with a JSON array of topic strings only. Example:
["Revenue figures", "Termination conditions", "Liability caps", "Payment terms"]

No preamble, no explanation — JSON array only."""


async def detect_topics(
    doc_names: list[str],
    domain: str = "general",
) -> list[str]:
    """
    Use Gemini to suggest relevant contradiction-check topics
    when the user doesn't supply their own.
    """
    client = genai.Client(api_key=settings.google_api_key)
    prompt = _TOPIC_DETECTION_PROMPT.format(
        doc_names=", ".join(doc_names),
        domain=domain,
    )

    response = await client.aio.models.generate_content(
        model=settings.llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=256,
        ),
    )

    try:
        text = response.text.strip()
        # Strip markdown fences if model ignores instructions
        text = text.replace("```json", "").replace("```", "").strip()
        topics = json.loads(text)
        return topics if isinstance(topics, list) else []
    except Exception as exc:
        logger.warning(f"Topic detection parse error: {exc} — using defaults")
        return ["Key figures and numbers", "Dates and deadlines", "Terms and conditions"]


# ── Main contradiction detection function ─────────────────────────────────────

async def detect_contradictions(
    workspace_id: str,
    document_ids: list[str],
    doc_name_map: dict[str, str],
    topics: list[str],
    domain: str = "general",
) -> list[ConflictResult]:
    """
    Detect contradictions across all document pairs in a workspace.

    For N documents this checks every pair: N*(N-1)/2 pairs.
    For each pair, it checks every topic.

    Args:
        document_ids:  list of ready document IDs in the workspace
        doc_name_map:  {document_id: display_name}
        topics:        list of topics to check (auto-detected if empty)
        domain:        workspace domain for better topic suggestions

    Returns:
        list of ConflictResult — one per (topic, doc_pair) combination
        where a conflict was found
    """
    if len(document_ids) < 2:
        raise ValueError(
            "Contradiction detection requires at least 2 documents in the workspace."
        )

    vector_store = get_vector_store()
    client = genai.Client(api_key=settings.google_api_key)

    # ── Auto-detect topics if none supplied ───────────────────────────────────
    if not topics:
        doc_names = [doc_name_map.get(d, d) for d in document_ids]
        topics = await detect_topics(doc_names, domain)
        logger.info(f"Auto-detected topics: {topics}")

    results: list[ConflictResult] = []

    # ── Check every document pair for every topic ─────────────────────────────
    for i in range(len(document_ids)):
        for j in range(i + 1, len(document_ids)):
            doc_a_id = document_ids[i]
            doc_b_id = document_ids[j]
            doc_a_name = doc_name_map.get(doc_a_id, f"Document {i+1}")
            doc_b_name = doc_name_map.get(doc_b_id, f"Document {j+1}")

            logger.info(
                f"Checking contradictions: '{doc_a_name}' vs '{doc_b_name}' "
                f"across {len(topics)} topics"
            )

            for topic in topics:
                result = await _check_topic_pair(
                    client=client,
                    vector_store=vector_store,
                    topic=topic,
                    doc_a_id=doc_a_id,
                    doc_a_name=doc_a_name,
                    doc_b_id=doc_b_id,
                    doc_b_name=doc_b_name,
                )
                if result:
                    results.append(result)

    # Sort — conflicts first, then by confidence descending
    results.sort(key=lambda r: (not r.conflict, -r.confidence))
    logger.info(
        f"Contradiction analysis complete: "
        f"{sum(1 for r in results if r.conflict)} conflicts found "
        f"across {len(results)} topic checks"
    )
    return results


async def _check_topic_pair(
    client: genai.Client,
    vector_store,
    topic: str,
    doc_a_id: str,
    doc_a_name: str,
    doc_b_id: str,
    doc_b_name: str,
) -> ConflictResult | None:
    """
    Check a single topic across a single document pair.
    Returns None if the topic isn't found in either document.
    """
    # Retrieve relevant chunks about this topic from both documents
    try:
        chunks_a = vector_store.similarity_search(
            query=topic, document_id=doc_a_id, k=3
        )
        chunks_b = vector_store.similarity_search(
            query=topic, document_id=doc_b_id, k=3
        )
    except ValueError:
        return None

    # Skip if topic not meaningfully present in either doc (low scores)
    if not chunks_a or not chunks_b:
        return None

    top_score_a = chunks_a[0].get("score", 0)
    top_score_b = chunks_b[0].get("score", 0)
    if top_score_a < 0.3 or top_score_b < 0.3:
        logger.debug(
            f"Skipping topic '{topic}' — low relevance scores "
            f"({top_score_a:.2f}, {top_score_b:.2f})"
        )
        return None

    # Build and send the comparison prompt
    prompt = _build_conflict_prompt(
        topic=topic,
        doc_a_name=doc_a_name,
        doc_a_chunks=chunks_a[:2],
        doc_b_name=doc_b_name,
        doc_b_chunks=chunks_b[:2],
    )

    try:
        response = await client.aio.models.generate_content(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_CONFLICT_SYSTEM_PROMPT,
                temperature=0.1,        # Near-zero — we want deterministic analysis
                max_output_tokens=512,
            ),
        )

        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)

    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse error for topic '{topic}': {exc}")
        return None
    except Exception as exc:
        logger.error(f"Gemini error for topic '{topic}': {exc}")
        return None

    # Build ConflictResult from parsed JSON
    conflict = parsed.get("conflict", False)
    side_a = None
    side_b = None

    if conflict:
        # Find the page number of the most relevant chunk for each side
        page_a = chunks_a[0].get("page_number") if chunks_a else None
        page_b = chunks_b[0].get("page_number") if chunks_b else None

        side_a = ConflictSide(
            document_id=doc_a_id,
            document_name=doc_a_name,
            page=page_a,
            text=parsed.get("side_a_text") or chunks_a[0]["text"][:300],
        )
        side_b = ConflictSide(
            document_id=doc_b_id,
            document_name=doc_b_name,
            page=page_b,
            text=parsed.get("side_b_text") or chunks_b[0]["text"][:300],
        )

    return ConflictResult(
        topic=topic,
        conflict=conflict,
        summary=parsed.get("summary", "Analysis complete."),
        side_a=side_a,
        side_b=side_b,
        confidence=float(parsed.get("confidence", 0.0)),
    )