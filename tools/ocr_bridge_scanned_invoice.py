#!/usr/bin/env python3
"""
ocr_bridge_scanned_invoice.py
------------------------------
n8n-compatible bridge for OCR-based scanned invoice extraction.

Mirrors the contract of ocr_bridge_digital_invoice.py but uses the
OCR-first pipeline (Tesseract) for image-only PDFs.

Output schema (JSON):
  status            str   OK | NEEDS_REVIEW | FAILED
  document_family   str   scanned_pdf
  router_decision   str   routed_to_scanned_invoice | invalid_input | extraction_failed
  invoice_type      str   invoice | advance_invoice | delivery_invoice | invoice_factura
  seller_name       str?
  seller_vat        str?
  invoice_number    str?
  invoice_date      str?  YYYY-MM-DD
  due_date          str?  YYYY-MM-DD
  currency          str   EUR
  total_gross       num?
  total_net         num?
  total_vat         num?
  vat_rate          int?  0 or 21
  reverse_charge    bool
  booking_amount    num?  = total_gross
  line_items        list  []  (reserved)
  warnings          list
  confidence        str   high | medium | low
  ocr_engine        str?
  source_file       str
  source_type       str
  person_name       str?
  object_name       str?
  review_reason     str?
  error_message     str?
"""

import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

SUPPORTED_MIME = {"application/pdf", "application/x-pdf"}

DECISION_ROUTED  = "routed_to_scanned_invoice"
DECISION_INVALID = "invalid_input"
DECISION_CRASHED = "extraction_failed"


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def _base_output(payload: dict) -> dict:
    return {
        "status":          None,
        "document_family": "scanned_pdf",
        "router_decision": None,
        "invoice_type":    None,
        "seller_name":     None,
        "seller_vat":      None,
        "invoice_number":  None,
        "invoice_date":    None,
        "due_date":        None,
        "currency":        "EUR",
        "total_gross":     None,
        "total_net":       None,
        "total_vat":       None,
        "vat_rate":        None,
        "reverse_charge":  False,
        "non_vat_payer":   False,
        "booking_amount":  None,
        "line_items":      [],
        "warnings":        [],
        "confidence":      "low",
        "triangle_status": "INCOMPLETE",
        "ocr_engine":      None,
        "source_file":     payload.get("file_name") or payload.get("source_file", ""),
        "source_type":     payload.get("source_type", ""),
        "person_name":     payload.get("person_name"),
        "object_name":     payload.get("object_name"),
        "review_reason":   None,
        "error_message":   None,
    }


def _failed(payload: dict, decision: str, error: str) -> dict:
    out = _base_output(payload)
    out["status"]          = "FAILED"
    out["router_decision"] = decision
    out["error_message"]   = error
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

def process_scanned_invoice_bridge(payload: dict) -> dict:
    """
    Extract structured invoice data from a scanned PDF.
    Always returns a valid dict (never raises).
    """
    err = _validate_input(payload)
    if err:
        return _failed(payload, DECISION_INVALID, err)

    source_file = payload["source_file"]

    try:
        from tools.extract_scanned_invoice_pdf import extract_scanned_invoice
        extracted = extract_scanned_invoice(source_file)
    except Exception as exc:
        return _failed(payload, DECISION_CRASHED, f"Extraction error: {exc}")

    out = _base_output(payload)
    out["router_decision"] = DECISION_ROUTED

    for field in [
        "invoice_type", "seller_name", "seller_vat",
        "invoice_number", "invoice_date", "due_date", "currency",
        "total_gross", "total_net", "total_vat", "vat_rate",
        "reverse_charge", "non_vat_payer", "booking_amount",
        "warnings", "confidence", "triangle_status", "ocr_engine",
    ]:
        if field in extracted:
            out[field] = extracted[field]

    # Extraction failed entirely
    if extracted.get("extraction_method") == "failed":
        out["status"]        = "FAILED"
        out["error_message"] = (
            extracted["warnings"][0] if extracted.get("warnings")
            else "OCR extraction failed"
        )
        return out

    # Missing critical field
    missing = [f for f in ("invoice_number", "invoice_date", "total_gross")
               if not extracted.get(f)]
    if missing:
        out["status"]        = "FAILED"
        out["error_message"] = f"Critical fields missing after OCR: {', '.join(missing)}"
        return out

    # Triangle mismatch → human review
    if extracted.get("triangle_status") == "NEEDS_REVIEW":
        out["status"]        = "NEEDS_REVIEW"
        reasons = [w for w in extracted.get("warnings", [])
                   if "mismatch" in w.lower() or "triangle" in w.lower()]
        out["review_reason"] = reasons[0] if reasons else "Amount triangle mismatch"
        return out

    out["status"] = "OK"
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        payload = json.loads(Path(sys.argv[1]).read_text())
    else:
        payload = json.loads(sys.stdin.read())
    result = process_scanned_invoice_bridge(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
