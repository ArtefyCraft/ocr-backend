#!/usr/bin/env python3
"""
ocr_bridge_digital_invoice.py
------------------------------
n8n-compatible bridge for deterministic digital B2B invoice extraction.

Mirrors the contract of ocr_bridge_telecom.py but targets digital invoice PDFs
(family: digital_pdf in samples/invoices/digital_pdf/).

Calling conventions:
  # As a Python library (used by ocr_bridge_service.py):
  from tools.ocr_bridge_digital_invoice import process_invoice_bridge
  result = process_invoice_bridge({"source_file": "...", ...})

Input schema (JSON):
  source_file   str  required  Absolute or relative path to the PDF file
  file_name     str  optional  Original filename (used in output metadata)
  mime_type     str  optional  Expected "application/pdf"; non-PDF returns FAILED
  object_name   str  optional  Storage object path
  person_name   str  optional  Company or person the invoice belongs to
  source_type   str  optional  Expected "digital_invoice"
  metadata      obj  optional  Pass-through metadata dict

Output schema (JSON):
  status            str   OK | NEEDS_REVIEW | FAILED
  invoice_type      str   invoice | advance_invoice | delivery_invoice | invoice_factura
  seller_name       str?
  seller_vat        str?
  invoice_number    str?
  invoice_date      str?  YYYY-MM-DD
  due_date          str?  YYYY-MM-DD
  currency          str   EUR
  total_gross       num?  Total payable incl. VAT
  total_net         num?  Total excl. VAT
  total_vat         num?  VAT amount
  vat_rate          int?  0 or 21 (None for non-VAT payers)
  reverse_charge    bool  True when VAT 0% + reverse charge clause
  non_vat_payer     bool  True when seller is not VAT registered
  booking_amount    num?  = total_gross
  warnings          list
  confidence        str   high | medium | low
  triangle_status   str   OK | NEEDS_REVIEW | INCOMPLETE
  source_file       str
  source_type       str
  person_name       str?
  object_name       str?
  review_reason     str?  set when status = NEEDS_REVIEW
  error_message     str?  set when status = FAILED
"""

import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

SUPPORTED_MIME = {"application/pdf", "application/x-pdf"}


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def _base_output(payload: dict) -> dict:
    return {
        "status": None,
        "invoice_type": None,
        "seller_name": None,
        "seller_vat": None,
        "invoice_number": None,
        "invoice_date": None,
        "due_date": None,
        "currency": "EUR",
        "total_gross": None,
        "total_net": None,
        "total_vat": None,
        "vat_rate": None,
        "reverse_charge": False,
        "non_vat_payer": False,
        "booking_amount": None,
        "warnings": [],
        "confidence": "low",
        "triangle_status": "INCOMPLETE",
        "source_file": payload.get("file_name") or payload.get("source_file", ""),
        "source_type": payload.get("source_type", ""),
        "person_name": payload.get("person_name"),
        "object_name": payload.get("object_name"),
        "review_reason": None,
        "error_message": None,
    }


def _failed(payload: dict, error: str) -> dict:
    out = _base_output(payload)
    out["status"] = "FAILED"
    out["error_message"] = error
    return out


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_input(payload: dict) -> Optional[str]:
    source_file = payload.get("source_file")
    if not source_file:
        return "source_file is required"

    mime = (payload.get("mime_type") or "").lower().strip()
    if mime and mime not in SUPPORTED_MIME:
        return f"Unsupported mime_type '{mime}' — expected application/pdf"

    if not Path(source_file).exists():
        return f"File not found: {source_file}"

    ext = Path(source_file).suffix.lower()
    if ext not in (".pdf",):
        return f"Unsupported file extension '{ext}' — expected .pdf"

    return None


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

def process_invoice_bridge(payload: dict) -> dict:
    """
    Extract structured invoice data from a digital PDF invoice.

    Always returns a valid dict (never raises). Status: OK / NEEDS_REVIEW / FAILED.
    """
    # --- Input validation ---
    err = _validate_input(payload)
    if err:
        return _failed(payload, err)

    source_file = payload["source_file"]

    # --- Extraction ---
    try:
        from tools.extract_digital_invoice_pdf import extract_digital_invoice
        extracted = extract_digital_invoice(source_file)
    except Exception as exc:
        return _failed(payload, f"Extraction error: {exc}")

    # --- Build output ---
    out = _base_output(payload)

    # Copy extractor fields
    for field in [
        "invoice_type", "seller_name", "seller_vat",
        "invoice_number", "invoice_date", "due_date", "currency",
        "total_gross", "total_net", "total_vat", "vat_rate",
        "reverse_charge", "non_vat_payer", "booking_amount",
        "warnings", "confidence", "triangle_status",
    ]:
        if field in extracted:
            out[field] = extracted[field]

    # --- Determine status ---
    if extracted.get("extraction_method") == "failed":
        out["status"] = "FAILED"
        out["error_message"] = (
            extracted["warnings"][0] if extracted.get("warnings")
            else "PDF extraction failed"
        )
        return out

    # Missing critical field → FAILED
    missing = [f for f in ("invoice_number", "invoice_date", "total_gross")
               if not extracted.get(f)]
    if missing:
        out["status"] = "FAILED"
        out["error_message"] = f"Critical fields missing: {', '.join(missing)}"
        return out

    # Triangle mismatch → NEEDS_REVIEW
    if extracted.get("triangle_status") == "NEEDS_REVIEW":
        out["status"] = "NEEDS_REVIEW"
        reasons = [w for w in extracted.get("warnings", []) if "Triangle" in w or "mismatch" in w.lower()]
        out["review_reason"] = reasons[0] if reasons else "Amount triangle mismatch"
        return out

    out["status"] = "OK"
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        payload = json.loads(Path(sys.argv[1]).read_text())
    else:
        payload = json.loads(sys.stdin.read())

    result = process_invoice_bridge(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
