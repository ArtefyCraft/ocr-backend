#!/usr/bin/env python3
"""
extract_telecom_router.py
--------------------------
Deterministic telecom bill router for SIA Woodenlays OCR pipeline.

Detects provider from PDF text, routes to the correct extractor,
and returns a normalized output schema that is consistent across
all supported providers (Tele2, Bite).

Supported providers:
  - tele2  → tools/extract_tele2_pdf.py
  - bite   → tools/extract_bite_pdf.py
  - unknown → returns extraction_status=NEEDS_REVIEW

Detection logic (deterministic, no LLM):
  1. Text contains 'Tele2' + DFC_ metadata  → tele2 (confidence: high)
  2. Text contains 'Tele2' (no DFC_)        → tele2 (confidence: medium)
  3. Text contains 'BITE Latvija'           → bite  (confidence: high)
  4. Neither found                          → unknown (confidence: low)

Normalized output schema:
  provider, seller_name, invoice_number, invoice_date,
  billing_period_from, billing_period_to,
  total_payable, previous_balance_or_overdue,
  current_period_amount, booking_amount, late_fee,
  line_items, warnings,
  extraction_status, confidence, source_file

extraction_status values:
  OK           — all fields extracted, validation PASS, no warnings
  NEEDS_REVIEW — validation JĀPĀRBAUDA, or extraction warnings present
  FAILED       — validation FAIL, or provider unknown, or extraction crashed

Usage:
  from tools.extract_telecom_router import route_telecom
  result = route_telecom("samples/telecom/tele2_pdf/9.pdf")

  or directly:
  python tools/extract_telecom_router.py samples/telecom/tele2_pdf/9.pdf
"""

import re
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError(
        "pdfplumber is required. Install with: pip install pdfplumber"
    )

# Provider extractors — imported lazily inside route_telecom to avoid
# circular imports and to allow the module to load even if one extractor fails.


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _peek_text(pdf_path: str, max_pages: int = 2) -> str:
    """
    Extract text from the first N pages only — used for fast provider detection.
    Avoids reading the full PDF twice when the provider can be identified from
    the first page.
    """
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def detect_provider(text: str) -> tuple[str, str]:
    """
    Identify the telecom provider from PDF text.

    Returns:
      (provider, confidence)
      provider:   'tele2' | 'bite' | 'unknown'
      confidence: 'high' | 'medium' | 'low'
    """
    has_tele2 = "Tele2" in text
    has_dfc = "DFC_" in text          # Tele2 digital PDFs embed DFC_ metadata
    has_bite = "BITE Latvija" in text

    if has_tele2 and has_bite:
        # Both labels found — unexpected; flag for review
        return "unknown", "low"

    if has_tele2:
        confidence = "high" if has_dfc else "medium"
        return "tele2", confidence

    if has_bite:
        return "bite", "high"

    return "unknown", "low"


# ---------------------------------------------------------------------------
# Billing period normalisation
# ---------------------------------------------------------------------------

def _split_billing_period(period: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Split 'DD.MM.YYYY - DD.MM.YYYY' into (YYYY-MM-DD, YYYY-MM-DD).
    Returns (None, None) if period is missing or cannot be parsed.
    """
    if not period:
        return None, None
    m = re.match(
        r'(\d{2}\.\d{2}\.\d{4})\s*[-\u2013]\s*(\d{2}\.\d{2}\.\d{4})',
        period.strip()
    )
    if not m:
        return None, None
    try:
        from_date = datetime.strptime(m.group(1), "%d.%m.%Y").strftime("%Y-%m-%d")
        to_date = datetime.strptime(m.group(2), "%d.%m.%Y").strftime("%Y-%m-%d")
        return from_date, to_date
    except ValueError:
        return None, None


# ---------------------------------------------------------------------------
# Status / confidence derivation
# ---------------------------------------------------------------------------

def _derive_status(
    validation_status: str,
    extraction_warnings: list,
    provider_confidence: str,
) -> tuple[str, str]:
    """
    Map validation outcome + provider confidence → (extraction_status, confidence).

    extraction_status:
      OK           — PASS, no warnings, high or medium provider confidence
      NEEDS_REVIEW — JĀPĀRBAUDA, or extraction warnings present
      FAILED       — validation FAIL

    confidence:
      high   — provider identified with high confidence, PASS, no warnings
      medium — provider identified, minor warnings, or medium provider confidence
      low    — validation FAIL or provider unknown
    """
    if validation_status == "FAIL":
        return "FAILED", "low"

    if validation_status == "JĀPĀRBAUDA" or extraction_warnings:
        return "NEEDS_REVIEW", "medium"

    # PASS — no warnings
    confidence = "high" if provider_confidence == "high" else "medium"
    return "OK", confidence


# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------

def _normalise(raw: dict, provider: str, provider_confidence: str) -> dict:
    """
    Map provider-specific extractor output to the shared normalized schema.
    Runs validation and sets extraction_status and confidence.
    """
    from tools.validate_telecom_amounts import validate_telecom

    validation = validate_telecom(raw)

    billing_from, billing_to = _split_billing_period(raw.get("billing_period"))

    extraction_warnings = raw.get("warnings", [])
    extraction_status, confidence = _derive_status(
        validation["overall_status"],
        extraction_warnings,
        provider_confidence,
    )

    return {
        # Identity
        "provider": provider,
        "source_file": raw.get("source_file"),
        # Invoice metadata
        "seller_name": raw.get("seller_name"),
        "invoice_number": raw.get("invoice_number"),
        "invoice_date": raw.get("invoice_date"),           # YYYY-MM-DD
        "billing_period_from": billing_from,               # YYYY-MM-DD
        "billing_period_to": billing_to,                   # YYYY-MM-DD
        # Amounts — critical accounting fields
        "total_net": raw.get("total_net"),
        "total_vat": raw.get("total_vat"),
        "total_gross": raw.get("total_gross"),
        "total_payable": raw.get("total_payable"),
        "previous_balance_or_overdue": raw.get("previous_balance_or_overdue"),
        "current_period_amount": raw.get("current_period_amount"),
        "booking_amount": raw.get("booking_amount"),       # always = current_period_amount
        "late_fee": raw.get("late_fee", 0.0),
        # Line items
        "line_items": raw.get("line_items", []),
        # Status
        "extraction_status": extraction_status,
        "confidence": confidence,
        "warnings": extraction_warnings,
        # Internals — useful for debugging and audit trail
        "_validation": validation,
        "_field_sources": raw.get("field_sources", {}),
        "_extraction_method": raw.get("extraction_method"),
        "_llm_fallback_used": raw.get("llm_fallback_used", False),
    }


def _make_unknown_result(pdf_path: str, reason: str) -> dict:
    """Return a NEEDS_REVIEW result when provider cannot be identified."""
    return {
        "provider": "unknown",
        "source_file": Path(pdf_path).name,
        "seller_name": None,
        "invoice_number": None,
        "invoice_date": None,
        "billing_period_from": None,
        "billing_period_to": None,
        "total_payable": None,
        "previous_balance_or_overdue": None,
        "current_period_amount": None,
        "booking_amount": None,
        "late_fee": 0.0,
        "line_items": [],
        "extraction_status": "NEEDS_REVIEW",
        "confidence": "low",
        "warnings": [f"Provider not identified: {reason}"],
        "_validation": None,
        "_field_sources": {},
        "_extraction_method": "failed",
        "_llm_fallback_used": False,
    }


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def route_telecom(pdf_path: str) -> dict:
    """
    Route a telecom PDF to the correct extractor and return normalized output.

    Steps:
      1. Peek at first 2 pages to detect provider
      2. Call provider-specific extractor (reads full PDF)
      3. Normalize output to shared schema
      4. Run shared validator
      5. Set extraction_status and confidence
      6. Return normalized dict
    """
    pdf_path = str(pdf_path)

    # 1. Detect provider
    try:
        peek = _peek_text(pdf_path)
    except Exception as exc:
        return _make_unknown_result(pdf_path, f"PDF read failed: {exc}")

    provider, provider_confidence = detect_provider(peek)

    if provider == "unknown":
        return _make_unknown_result(
            pdf_path,
            f"Neither 'Tele2' nor 'BITE Latvija' found in first 2 pages"
        )

    # 2. Call provider extractor
    try:
        if provider == "tele2":
            from tools.extract_tele2_pdf import extract_tele2
            raw = extract_tele2(pdf_path)
        elif provider == "bite":
            from tools.extract_bite_pdf import extract_bite
            raw = extract_bite(pdf_path)
        else:
            return _make_unknown_result(pdf_path, f"Unhandled provider: {provider}")
    except Exception as exc:
        return _make_unknown_result(pdf_path, f"Extractor crashed: {exc}")

    # 3-5. Normalize, validate, set status
    return _normalise(raw, provider, provider_confidence)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_telecom_router.py <path_to_pdf>")
        sys.exit(1)

    result = route_telecom(sys.argv[1])
    # Remove internal debug keys for clean CLI output
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    print(json.dumps(clean, indent=2, ensure_ascii=False))
