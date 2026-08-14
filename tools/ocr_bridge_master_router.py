#!/usr/bin/env python3
"""
ocr_bridge_master_router.py
----------------------------
n8n-compatible master router bridge.

Accepts a generic document (any supported family), classifies it using
extract_master_router.classify_document(), then dispatches to the correct
family bridge and wraps the result in a top-level master output contract.

Supported families:
  telecom_pdf          → ocr_bridge_telecom.process_telecom_bridge()
  digital_invoice_pdf  → ocr_bridge_digital_invoice.process_invoice_bridge()
  scanned_invoice_pdf  → ocr_bridge_scanned_invoice.process_scanned_invoice_bridge()
  receipt_image        → ocr_bridge_receipt_image.process_receipt_bridge()

Master output contract:
  status            str   OK | NEEDS_REVIEW | FAILED
  document_family   str   telecom_pdf | digital_invoice_pdf | scanned_invoice_pdf | receipt_image | unknown
  router_decision   str   routed_to_<family> | classification_failed | invalid_input | extraction_failed
  classification    dict  {family, confidence, reason, telecom_score, invoice_score}
  extracted_data    dict  family-specific bridge output (or {} on classification failure)
  source_file       str
  source_type       str
  person_name       str?
  object_name       str?
  error_message     str?
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# _CLUSTER_A_v1 — VERBATIM_CLASSIFIER routing behind env flag.
#
# Values: off (default) | shadow | enforce
#   off      → scanner still runs inside classify_document + result attached
#              to classification.verbatim_prescan for observability. No routing
#              change.
#   shadow   → same as off but sets a `verbatim_classifier_mode` marker so
#              downstream logs / metrics can segment.
#   enforce  → verbatim hits with a family that has a DEDICATED bridge
#              (currently ONLY the pre-existing families) override the
#              heuristic classifier. Hits for NEW families with no bridge yet
#              (credit_note / advance_invoice / delivery_note / receipt_z /
#              receipt_x) set status=NEEDS_REVIEW instead of silently misrouting.
_VERBATIM_MODES = {"off", "shadow", "enforce"}
_ENFORCE_UNIMPLEMENTED_ROUTES = {
    "credit_note",
    "advance_invoice",
    "delivery_note",
    "receipt_z",
    "receipt_x",
}


def _resolve_verbatim_mode() -> str:
    raw = (os.environ.get("VERBATIM_CLASSIFIER", "") or "").strip().lower()
    if raw in _VERBATIM_MODES:
        return raw
    # Unknown values (typo like "SHADOWW", "on", "true") fall back to off
    # with a stderr warning so a misconfigured flag surfaces on first request.
    if raw:
        print(
            f"[cluster-a] WARN: unknown VERBATIM_CLASSIFIER value {raw!r} "
            f"— falling back to 'off'. Expected: off | shadow | enforce.",
            file=sys.stderr,
        )
    return "off"

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS   = {".pdf"}
ALL_EXTENSIONS   = IMAGE_EXTENSIONS | PDF_EXTENSIONS

DECISION_ROUTED             = "routed_to_{family}"
DECISION_INVALID            = "invalid_input"
DECISION_CLASSIFICATION_FAIL = "classification_failed"
DECISION_EXTRACTION_FAIL    = "extraction_failed"


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def _base_output(payload: dict, classification: dict) -> dict:
    return {
        "status":          None,
        "document_family": classification.get("family", "unknown"),
        "router_decision": None,
        "classification":  {
            "family":        classification.get("family"),
            "confidence":    classification.get("confidence"),
            "reason":        classification.get("reason"),
            "telecom_score": classification.get("telecom_score", 0),
            "invoice_score": classification.get("invoice_score", 0),
            "page_count":    classification.get("page_count", 0),
        },
        "page_count":      classification.get("page_count", 0),
        "extracted_data":  {},
        "source_file":     payload.get("source_file") or payload.get("file_name", ""),
        "source_type":     payload.get("source_type", ""),
        "person_name":     payload.get("person_name"),
        "object_name":     payload.get("object_name"),
        "error_message":   None,
    }


def _failed(payload: dict, decision: str, error: str,
            classification: Optional[dict] = None) -> dict:
    cls = classification or {"family": "unknown", "confidence": "low",
                              "reason": error, "telecom_score": 0, "invoice_score": 0}
    out = _base_output(payload, cls)
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
    if ext not in ALL_EXTENSIONS:
        return (
            f"Unsupported file extension '{ext}' — "
            f"expected one of: {', '.join(sorted(ALL_EXTENSIONS))}"
        )
    return None


# ---------------------------------------------------------------------------
# Family dispatch
# ---------------------------------------------------------------------------

_FAMILY_BRIDGES = {
    "telecom_pdf":         "tools.ocr_bridge_telecom.process_telecom_bridge",
    "digital_invoice_pdf": "tools.ocr_bridge_digital_invoice.process_invoice_bridge",
    "scanned_invoice_pdf": "tools.ocr_bridge_scanned_invoice.process_scanned_invoice_bridge",
    "receipt_image":       "tools.ocr_bridge_receipt_image.process_receipt_bridge",
}


def _dispatch(family: str, payload: dict) -> dict:
    """Call the correct bridge for the given family. Returns bridge output dict."""
    bridge_path = _FAMILY_BRIDGES.get(family)
    if not bridge_path:
        raise ValueError(f"No bridge registered for family '{family}'")

    module_path, func_name = bridge_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    bridge_fn = getattr(module, func_name)
    return bridge_fn(payload)


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

def process_master_bridge(payload: dict) -> dict:
    """
    Classify and extract structured data from any supported document.
    Always returns a valid dict (never raises).
    """
    # Input validation
    err = _validate_input(payload)
    if err:
        return _failed(payload, DECISION_INVALID, err)

    source_file = payload["source_file"]

    # Classify
    try:
        from tools.extract_master_router import classify_document
        classification = classify_document(source_file, payload.get("source_type", ""))
    except Exception as exc:
        return _failed(payload, DECISION_CLASSIFICATION_FAIL,
                       f"Classification error: {exc}")

    # _CLUSTER_A_v1 — verbatim classifier prescan integration.
    # classify_document attaches verbatim_prescan={verbatim_hit, route_hint, ...}
    # to its result. Here we optionally override family routing per env flag.
    verbatim_mode = _resolve_verbatim_mode()
    prescan = classification.get("verbatim_prescan") or {}
    classification["verbatim_classifier_mode"] = verbatim_mode
    family = classification.get("family", "unknown")

    if verbatim_mode == "enforce" and prescan.get("verbatim_hit"):
        route_hint = prescan.get("route_hint")
        # Only override when the target family HAS a dedicated bridge —
        # otherwise mark NEEDS_REVIEW so the operator handles it manually
        # (no silent misroute into wrong extractor).
        if route_hint in _ENFORCE_UNIMPLEMENTED_ROUTES:
            out = _base_output(payload, classification)
            out["status"] = "NEEDS_REVIEW"
            out["router_decision"] = f"verbatim_route_unimplemented_{route_hint}"
            out["error_message"] = (
                f"Verbatim scanner detected '{prescan.get('matched_token')}' "
                f"→ route {route_hint!r} has no dedicated bridge yet — "
                f"NEEDS_REVIEW (Cluster A ships shadow-only; per-family "
                f"bridges land in a later chunk)."
            )
            return out
        # route_hint is None or matches a family we already handle — no-op.

    # Unknown family → cannot route
    if family == "unknown":
        out = _base_output(payload, classification)
        out["status"]          = "FAILED"
        out["router_decision"] = DECISION_CLASSIFICATION_FAIL
        out["error_message"]   = (
            f"Cannot classify document: {classification.get('reason', 'unknown reason')}"
        )
        return out

    # Dispatch to family bridge
    try:
        extracted = _dispatch(family, payload)
    except Exception as exc:
        out = _base_output(payload, classification)
        out["status"]          = "FAILED"
        out["router_decision"] = DECISION_EXTRACTION_FAIL
        out["error_message"]   = f"Bridge dispatch error: {exc}"
        return out

    # Wrap result
    out = _base_output(payload, classification)
    out["router_decision"] = DECISION_ROUTED.format(family=family)
    out["extracted_data"]  = extracted

    # Inherit top-level status from extracted result
    out["status"] = extracted.get("status", "FAILED")

    # Apply normalizer: financial inference + new output contract fields
    try:
        from tools.ocr_normalizer import build_output_contract
        out = build_output_contract(out)
    except Exception as exc:
        # Normalizer failure is non-fatal — log and continue with raw output
        out.setdefault("error_message", f"Normalizer error: {exc}")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        payload = json.loads(Path(sys.argv[1]).read_text())
    else:
        payload = json.loads(sys.stdin.read())
    result = process_master_bridge(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
