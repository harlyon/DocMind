"""
Due diligence checklist service — Feature 3.

Strategy:
  For each checklist item:
    1. Use the item's question as a RAG query against the document
    2. Retrieve the top-k most relevant chunks
    3. Ask Gemini: "Based on these passages, does this item pass, fail, or partial?"
    4. Parse structured JSON response with finding + evidence
    5. Aggregate results into a summary

Why structured output matters here:
  This is the feature that makes DocMind look like a real product.
  A law firm doesn't want prose — they want a table with pass/fail/partial
  badges, expandable evidence, and page references they can verify.
  The structured JSON output maps directly to that UI.

Built-in templates:
  - mna_contract       M&A contract review checklist
  - financial_report   Financial report audit checklist
  - employment_contract Employment contract review checklist
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging

from google import genai
from google.genai import types

from core.config import get_settings
from models.schemas import ChecklistItem, ChecklistItemResult
from services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Built-in templates ────────────────────────────────────────────────────────

CHECKLIST_TEMPLATES: dict[str, list[ChecklistItem]] = {

    "mna_contract": [
        ChecklistItem(id="parties", label="Parties clearly identified",
            question="Who are the parties to this agreement?"),
        ChecklistItem(id="consideration", label="Consideration specified",
            question="What is the purchase price or consideration amount?"),
        ChecklistItem(id="closing_date", label="Closing date defined",
            question="What is the closing date or completion date?"),
        ChecklistItem(id="conditions_precedent", label="Conditions precedent listed",
            question="What conditions must be satisfied before closing?"),
        ChecklistItem(id="representations", label="Representations and warranties present",
            question="What representations and warranties are made by each party?"),
        ChecklistItem(id="indemnification", label="Indemnification clause present",
            question="What are the indemnification obligations?"),
        ChecklistItem(id="termination", label="Termination rights defined",
            question="Under what conditions can this agreement be terminated?"),
        ChecklistItem(id="liability_cap", label="Liability cap specified",
            question="Is there a cap on liability? What is the maximum liability amount?"),
        ChecklistItem(id="governing_law", label="Governing law specified",
            question="Which jurisdiction's law governs this agreement?"),
        ChecklistItem(id="dispute_resolution", label="Dispute resolution mechanism",
            question="How are disputes resolved — arbitration, litigation, mediation?"),
        ChecklistItem(id="confidentiality", label="Confidentiality obligations",
            question="What confidentiality or non-disclosure obligations exist?"),
        ChecklistItem(id="non_compete", label="Non-compete or non-solicitation",
            question="Are there any non-compete or non-solicitation restrictions?"),
    ],

    "financial_report": [
        ChecklistItem(id="revenue", label="Revenue figures disclosed",
            question="What are the total revenue figures reported?"),
        ChecklistItem(id="net_income", label="Net income / profit reported",
            question="What is the net income or net profit for the period?"),
        ChecklistItem(id="ebitda", label="EBITDA disclosed",
            question="Is EBITDA or operating profit disclosed?"),
        ChecklistItem(id="cash_flow", label="Cash flow statement present",
            question="What does the cash flow statement show?"),
        ChecklistItem(id="debt", label="Debt obligations disclosed",
            question="What are the total debt obligations and repayment terms?"),
        ChecklistItem(id="risk_factors", label="Risk factors disclosed",
            question="What risk factors are disclosed?"),
        ChecklistItem(id="auditor", label="Auditor opinion present",
            question="What is the auditor's opinion? Is it qualified or unqualified?"),
        ChecklistItem(id="going_concern", label="Going concern assessment",
            question="Is there any going concern doubt or qualification?"),
        ChecklistItem(id="related_party", label="Related party transactions disclosed",
            question="Are related party transactions disclosed?"),
        ChecklistItem(id="guidance", label="Forward guidance provided",
            question="Is there any forward-looking guidance or outlook for future periods?"),
    ],

    "employment_contract": [
        ChecklistItem(id="role", label="Role and responsibilities defined",
            question="What is the employee's role, title, and responsibilities?"),
        ChecklistItem(id="compensation", label="Compensation specified",
            question="What is the base salary or compensation?"),
        ChecklistItem(id="benefits", label="Benefits outlined",
            question="What benefits are provided — health, pension, equity?"),
        ChecklistItem(id="start_date", label="Start date specified",
            question="What is the employment start date?"),
        ChecklistItem(id="probation", label="Probation period defined",
            question="Is there a probation period? How long?"),
        ChecklistItem(id="notice_period", label="Notice period specified",
            question="What notice period is required for termination by either party?"),
        ChecklistItem(id="ip_assignment", label="IP assignment clause present",
            question="Is there an intellectual property assignment clause?"),
        ChecklistItem(id="non_compete_employment", label="Non-compete clause",
            question="Is there a non-compete restriction after employment ends?"),
        ChecklistItem(id="termination_grounds", label="Termination grounds specified",
            question="What are the grounds for termination with and without cause?"),
        ChecklistItem(id="governing_law_employment", label="Governing law specified",
            question="Which jurisdiction governs this employment contract?"),
    ],
}


# ── Prompt ────────────────────────────────────────────────────────────────────

_CHECKLIST_SYSTEM_PROMPT = """You are a precise document analyst performing due diligence.

You will be given context passages from a document and a specific checklist item to verify.
Your job is to determine whether the checklist item is satisfied based solely on the evidence.

You MUST respond with valid JSON only — no preamble, no markdown fences.

Response schema:
{
  "status": "pass" | "fail" | "partial" | "not_found",
  "finding": "One clear sentence summarising what was found",
  "evidence": ["Direct quote 1 from the passages", "Direct quote 2"],
  "pages": [4, 7]
}

Status definitions:
  pass      — the item is clearly and fully satisfied by the document
  fail      — the item is explicitly contradicted or clearly absent
  partial   — the item is partially addressed but incomplete or ambiguous
  not_found — not enough information in the retrieved passages to determine

Rules:
- Base your answer ONLY on the provided passages
- evidence should be short direct quotes (under 30 words each), maximum 3
- pages should list all page numbers where evidence was found
- If status is not_found, evidence and pages can be empty arrays
- Never invent or assume information not present in the passages"""


def _build_checklist_prompt(item: ChecklistItem, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Page {c.get('page_number', '?')}]: {c['text']}"
        for c in chunks
    )
    return f"""CHECKLIST ITEM: {item.label}
QUESTION: {item.question}

DOCUMENT PASSAGES:
{context}

Based on these passages, is the checklist item "{item.label}" satisfied?
Respond with JSON only."""


# ── Main checklist function ───────────────────────────────────────────────────

async def run_checklist(
    document_id: str,
    items: list[ChecklistItem],
) -> list[ChecklistItemResult]:
    """
    Run a due diligence checklist against a single document.

    For each item:
      1. RAG retrieval using the item's question as the query
      2. Gemini structured analysis → pass/fail/partial/not_found
      3. Return cited evidence with page numbers

    Args:
        document_id: the document to analyse
        items:       list of ChecklistItem to verify

    Returns:
        list of ChecklistItemResult — one per item, in order
    """
    vector_store = get_vector_store()
    client = genai.Client(api_key=settings.google_api_key)
    results: list[ChecklistItemResult] = []

    for item in items:
        logger.info(f"Checking item: '{item.label}'")
        result = await _check_item(
            client=client,
            vector_store=vector_store,
            document_id=document_id,
            item=item,
        )
        results.append(result)

    # Summary log
    status_counts = {s: 0 for s in ["pass", "fail", "partial", "not_found"]}
    for r in results:
        status_counts[r.status] += 1
    logger.info(
        f"Checklist complete for document {document_id}: "
        + ", ".join(f"{k}={v}" for k, v in status_counts.items())
    )

    return results


async def _check_item(
    client: genai.Client,
    vector_store,
    document_id: str,
    item: ChecklistItem,
) -> ChecklistItemResult:
    """
    Check a single checklist item against the document.
    Returns a safe fallback result if retrieval or parsing fails.
    """
    # ── Retrieve relevant chunks ──────────────────────────────────────────────
    try:
        chunks = vector_store.similarity_search(
            query=item.question,
            document_id=document_id,
            k=4,
        )
    except ValueError:
        return _not_found_result(item, reason="Document not found in vector store")

    if not chunks or chunks[0].get("score", 0) < 0.25:
        return _not_found_result(item, reason="No relevant passages found")

    # ── Ask Gemini ────────────────────────────────────────────────────────────
    prompt = _build_checklist_prompt(item, chunks[:3])

    try:
        response = await client.aio.models.generate_content(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_CHECKLIST_SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=512,
            ),
        )

        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)

    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse error for item '{item.id}': {exc}")
        return _not_found_result(item, reason="Could not parse model response")
    except Exception as exc:
        logger.error(f"Gemini error for item '{item.id}': {exc}")
        return _not_found_result(item, reason=f"Model error: {str(exc)[:100]}")

    # ── Validate and normalise parsed response ────────────────────────────────
    valid_statuses = {"pass", "fail", "partial", "not_found"}
    status = parsed.get("status", "not_found")
    if status not in valid_statuses:
        status = "not_found"

    return ChecklistItemResult(
        id=item.id,
        label=item.label,
        status=status,
        finding=parsed.get("finding", "No finding available."),
        evidence=parsed.get("evidence", [])[:3],      # cap at 3 quotes
        pages=parsed.get("pages", []),
        required=item.required,
    )


def _not_found_result(item: ChecklistItem, reason: str = "") -> ChecklistItemResult:
    """Safe fallback when retrieval or parsing fails."""
    return ChecklistItemResult(
        id=item.id,
        label=item.label,
        status="not_found",
        finding=reason or "Could not determine from available content.",
        evidence=[],
        pages=[],
        required=item.required,
    )


def get_template(template_name: str) -> list[ChecklistItem]:
    """Return a built-in checklist template by name."""
    if template_name not in CHECKLIST_TEMPLATES:
        raise ValueError(
            f"Unknown template '{template_name}'. "
            f"Available: {', '.join(CHECKLIST_TEMPLATES.keys())}"
        )
    return CHECKLIST_TEMPLATES[template_name]