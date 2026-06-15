#!/usr/bin/env python3
"""
ocr_bridge_telecom.py
---------------------
n8n-compatible bridge for deterministic telecom bill extraction.

Wraps tools/extract_telecom_router.py with:
  - strict input/output contract matching n8n's Execute Command node convention
  - input validation (required fields, MIME type, file existence)
  - enriched output schema (status, router_decision, review_reason, error_message)
  - clean error handling — never crashes, always returns valid JSON

Calling conventions:
  # From a file path (recommended for n8n Execute Command node):
  python tools/ocr_bridge_telecom.py /tmp/n8n_payload.json

  # From stdin (for piping / local tests):
  echo '{"source_file": "..."}' | python tools/ocr_bridge_telecom.py

  # As a Python library:
  from tools.ocr_bridge_telecom import process_telecom_bridge
  result = process_telecom_bridge({"source_file": "...", ...})

Input schema (JSON):
  source_file   str  required  Absolute or relative path to the PDF file
  file_name     str  optional  Original filename (used in output metadata)
  mime_type     str  optional  Expected "application/pdf"; non-PDF returns FAILED
  object_name   str  optional  Storage object path (e.g. "telecom/tele2_pdf/9.pdf")
  person_name   str  optional  Company or person the invoice belongs to
  source_type   str  optional  Expected "telecom_bill"; mismatches generate a warning
  metadata      obj  optional  Pass-through metadata dict

Output schema (JSON):
  status                   str   OK | NEEDS_REVIEW | FAILED
  provider                 str   tele2 | bite | unknown
  router_decision          str   routed_to_tele2 | routed_to_bite |
                                 provider_unknown | invalid_input | extraction_failed
  seller_name              str?
  invoice_number           str?
  invoice_date             str?  YYYY-MM-DD
  billing_period_from      str?  YYYY-MM-DD
  billing_period_to        str?  YYYY-MM-DD
  total_payable            num?
  previous_balance_or_overdue num?
  current_period_amount    num?
  booking_amount           num?
  late_fee                 num
  line_items               list
  warnings                 list
  confidence               str   high | medium | low
  source_file              str
  source_type              str
  person_name              str?
  object_name              str?
  review_reason            str?  set when status = NEEDS_REVIEW
  error_message            str?  set when status = FAILED
"""

import sys
import json
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Router decision labels
# ---------------------------------------------------------------------------

DECISION_TELE2    = "routed_to_tele2"
DECISION_BITE     = "routed_to_bite"
DECISION_UNKNOWN  = "provider_unknown"
DECISION_INVALID  = "invalid_input"
DECISION_CRASHED  = "extraction_failed"

SUPPORTED_MIME = {"application/pdf", "application/x-pdf"}


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def _base_output(payload: dict) -> dict:
    """Skeleton output populated with pass-through metadata."""
    return {
        "status": None,
        "provider": "unknown",
        "router_decision": None,
        "seller_name": None,
        "invoice_number": None,
        "invoice_date": None,
        "billing_period_from": None,
        "billing_period_to": None,
        "total_net": None,
        "total_vat": None,
        "total_gross": None,
        "total_payable": None,
        "previous_balance_or_overdue": None,
        "current_period_amount": None,
        "booking_amount": None,
        "late_fee": 0.0,
        "line_items": [],
        "warnings": [],
        "confidence": "low",
        "source_file": payload.get("file_name") or payload.get("source_file", ""),
        "source_type": payload.get("source_type", ""),
        "person_name": payload.get("person_name"),
        "object_name": payload.get("object_name"),
        "review_reason": None,
        "error_message": None,
    }


def _failed(payload: dict, decision: str, error: str) -> dict:
    out = _base_output(payload)
    out["status"] = "FAILED"
    out["router_decision"] = decision
    out["error_message"] = error
    return out


def _needs_review(payload: dict, decision: str, reason: str,
                  partial: Optional[dict] = None) -> dict:
    out = _base_output(payload)
    out["status"] = "NEEDS_REVIEW"
    out["router_decision"] = decision
    out["review_reason"] = reason
    if partial:
        _merge_router_output(out, partial)
    return out


def _merge_router_output(out: dict, router_result: dict) -> None:
    """Copy router fields into the bridge output dict in-place."""
    fields = [
        "provider", "seller_name", "invoice_number", "invoice_date",
        "billing_period_from", "billing_period_to",
        "total_net", "total_vat", "total_gross",
        "total_payable", "previous_balance_or_overdue",
        "current_period_amount", "booking_amount", "late_fee",
        "line_items", "warnings", "confidence",
    ]
    for f in fields:
        if f in router_result:
            out[f] = router_result[f]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_input(payload: dict) -> Optional[str]:
    """
    Return an error message string if the payload is invalid, else None.
    """
    source_file = payload.get("source_file")
    if not source_file:
        return "source_file is required"

    # MIME check — only block if explicitly wrong; absent MIME is OK
    mime = (payload.get("mime_type") or "").lower().strip()
    if mime and mime not in SUPPORTED_MIME:
        return f"Unsupported mime_type '{mime}' — expected application/pdf"

    # File existence
    if not Path(source_file).exists():
        return f"File not found: {source_file}"

    # Extension sanity check
    ext = Path(source_file).suffix.lower()
    if ext not in (".pdf",):
        return f"Unsupported file extension '{ext}' — expected .pdf"

    return None


# ---------------------------------------------------------------------------
# Router decision label
# ---------------------------------------------------------------------------

def _router_decision(provider: str) -> str:
    if provider == "tele2":
        return DECISION_TELE2
    if provider == "bite":
        return DECISION_BITE
    return DECISION_UNKNOWN


# ---------------------------------------------------------------------------
# Review reason builder
# ---------------------------------------------------------------------------

def _build_review_reason(router_result: dict) -> Optional[str]:
    """Compose a human-readable review reason from router output."""
    reasons = []
    confidence = router_result.get("confidence", "low")
    ext_status = router_result.get("extraction_status", "")
    warnings = router_result.get("warnings", [])
    val = router_result.get("_validation")

    if ext_status == "FAILED":
        reasons.append("Validation FAIL")
    if ext_status == "NEEDS_REVIEW":
        reasons.append("Validation result: JĀPĀRBAUDA")
    if confidence == "medium":
        reasons.append("Provider detected with medium confidence")
    if confidence == "low":
        reasons.append("Provider not identified")
    if warnings:
        reasons.append(f"Extraction warnings ({len(warnings)}): {'; '.join(warnings[:3])}")
    if val and val.get("overall_status") == "FAIL":
        failed_checks = [c["message"] for c in val.get("checks", []) if c["status"] == "FAIL"]
        reasons.append("Validation failures: " + "; ".join(failed_checks[:2]))

    return " | ".join(reasons) if reasons else None


# ---------------------------------------------------------------------------
# Core bridge function (library entry point)
# ---------------------------------------------------------------------------

def process_telecom_bridge(payload: dict) -> dict:
    """
    Process a telecom PDF extraction request.

    Args:
        payload: dict matching the bridge input schema

    Returns:
        dict matching the bridge output schema — never raises an exception
    """
    # 1. Validate input
    input_error = _validate_input(payload)
    if input_error:
        return _failed(payload, DECISION_INVALID, input_error)

    source_file = payload["source_file"]

    # 2. Route through telecom router
    try:
        from tools.extract_telecom_router import route_telecom
        router_result = route_telecom(source_file)
    except Exception as exc:
        return _failed(payload, DECISION_CRASHED, f"Router exception: {exc}")

    # 3. Map router output to bridge output
    provider = router_result.get("provider", "unknown")
    ext_status = router_result.get("extraction_status", "FAILED")
    confidence = router_result.get("confidence", "low")
    decision = _router_decision(provider)

    out = _base_output(payload)
    out["router_decision"] = decision
    _merge_router_output(out, router_result)

    # 4. Set source_file to the actual filename (not the path)
    out["source_file"] = Path(source_file).name

    # 5. Source type warning
    source_type = payload.get("source_type", "")
    if source_type and source_type != "telecom_bill":
        out["warnings"].append(
            f"source_type '{source_type}' — expected 'telecom_bill'; processing anyway"
        )

    # 6. Map extraction_status → bridge status
    if provider == "unknown":
        out["status"] = "NEEDS_REVIEW"
        out["review_reason"] = _build_review_reason(router_result) or \
            "Provider not identified — manual classification required"
    elif ext_status == "FAILED":
        out["status"] = "FAILED"
        val = router_result.get("_validation")
        failed_checks = []
        if val:
            failed_checks = [c["message"] for c in val.get("checks", []) if c["status"] == "FAIL"]
        out["error_message"] = "Validation FAIL: " + "; ".join(failed_checks) \
            if failed_checks else "Extraction or validation failed"
    elif ext_status == "NEEDS_REVIEW":
        out["status"] = "NEEDS_REVIEW"
        out["review_reason"] = _build_review_reason(router_result)
    else:
        out["status"] = "OK"

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """
    Read JSON payload from file path (argv[1]) or stdin.
    Write JSON result to stdout.
    Exit code: 0 always — errors are encoded in the output JSON.
    """
    # Read input
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
        try:
            with open(input_path, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            result = {
                "status": "FAILED",
                "router_decision": DECISION_INVALID,
                "error_message": f"Cannot read input file '{input_path}': {exc}",
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(0)
    else:
        try:
            payload = json.load(sys.stdin)
        except Exception as exc:
            result = {
                "status": "FAILED",
                "router_decision": DECISION_INVALID,
                "error_message": f"Cannot parse stdin JSON: {exc}",
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(0)

    result = process_telecom_bridge(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
