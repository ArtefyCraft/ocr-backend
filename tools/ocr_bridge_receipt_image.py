#!/usr/bin/env python3
"""
ocr_bridge_receipt_image.py
----------------------------
n8n-compatible bridge for OCR-based receipt image extraction.

Output schema (JSON):
  status            str   OK | NEEDS_REVIEW | FAILED
  document_family   str   receipt_image
  router_decision   str   routed_to_receipt_image | invalid_input | extraction_failed
  seller_name       str?
  receipt_number    str?
  receipt_date      str?  YYYY-MM-DD
  currency          str   EUR
  total_gross       num?
  total_net         num?
  total_vat         num?
  payment_method    str?  cash | card | bank_transfer
  discounts         list  [{description, amount}]
  line_items        list  []  (reserved)
  warnings          list
  confidence        str   high | medium | low
  triangle_status   str   OK | NEEDS_REVIEW | INCOMPLETE
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

SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

DECISION_ROUTED  = "routed_to_receipt_image"
DECISION_INVALID = "invalid_input"
DECISION_CRASHED = "extraction_failed"


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def _base_output(payload: dict) -> dict:
    return {
        "status":          None,
        "document_family": "receipt_image",
        "router_decision": None,
        "seller_name":     None,
        "receipt_number":  None,
        "receipt_date":    None,
        "currency":        "EUR",
        "total_gross":     None,
        "total_net":       None,
        "total_vat":       None,
        "payment_method":  None,
        "discounts":       [],
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
    if not Path(source_file).exists():
        return f"File not found: {source_file}"
    ext = Path(source_file).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"Unsupported file extension '{ext}' — expected image file"
    mime = (payload.get("mime_type") or "").lower().strip()
    if mime and not (mime.startswith("image/") or mime in ("application/octet-stream",)):
        return f"Unsupported mime_type '{mime}' — expected image/*"
    return None


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

def process_receipt_bridge(payload: dict) -> dict:
    """
    Extract structured receipt data from an image file.
    Always returns a valid dict (never raises).
    """
    err = _validate_input(payload)
    if err:
        return _failed(payload, DECISION_INVALID, err)

    source_file = payload["source_file"]

    try:
        from tools.extract_receipt_image import extract_receipt
        extracted = extract_receipt(source_file)
    except Exception as exc:
        return _failed(payload, DECISION_CRASHED, f"Extraction error: {exc}")

    out = _base_output(payload)
    out["router_decision"] = DECISION_ROUTED

    for field in [
        "seller_name", "receipt_number", "receipt_date", "item_description",
        "currency", "total_gross", "total_net", "total_vat", "payment_method",
        "discounts", "line_items", "warnings", "confidence",
        "triangle_status", "ocr_engine",
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
    missing = [f for f in ("receipt_date", "total_gross")
               if not extracted.get(f)]
    if len(missing) >= 2:
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

    # Missing one critical field → NEEDS_REVIEW rather than FAILED
    if missing:
        out["status"]        = "NEEDS_REVIEW"
        out["review_reason"] = f"Could not extract: {', '.join(missing)}"
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
    result = process_receipt_bridge(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
