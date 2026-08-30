#!/usr/bin/env python3
"""
ocr_bridge_service.py
----------------------
FastAPI HTTP service wrapping process_telecom_bridge().

Replaces the Execute Command pattern for n8n Cloud compatibility.
n8n Cloud cannot run shell commands, so the bridge runs as an external
HTTP service and n8n calls it via HTTP Request node.

Endpoints:
  GET  /           — service info
  GET  /health     — health check
  POST /process    — extract telecom bill data from PDF

Running locally:
  python tools/ocr_bridge_service.py
  # or:
  uvicorn tools.ocr_bridge_service:app --host 0.0.0.0 --port 8000

Expose publicly for n8n Cloud:
  ngrok http 8000
  # Set OCR_BRIDGE_URL = https://xxxx.ngrok-free.app in n8n env vars
"""

import base64
import os
import sys
import tempfile
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from tools.ocr_bridge_telecom import process_telecom_bridge
from tools.ocr_bridge_digital_invoice import process_invoice_bridge
from tools.ocr_bridge_scanned_invoice import process_scanned_invoice_bridge
from tools.ocr_bridge_receipt_image import process_receipt_bridge
from tools.ocr_bridge_master_router import process_master_bridge

# ---------------------------------------------------------------------------
# Unified schema normalizer (Section 12 of architecture plan)
# ---------------------------------------------------------------------------

def _normalize_to_unified_schema(master_result: dict) -> dict:
    """
    Transform family-specific master router output into the unified
    Section 12 schema that matches the n8n pipeline's extracted_data contract.
    """
    ext = master_result.get("extracted_data", {})
    family = master_result.get("document_family", "unknown")
    provider = ext.get("provider", "")

    # --- Build unified items array ---
    items = []
    is_tele2 = (provider == "tele2")
    is_bite = (provider == "bite")

    if is_tele2 or is_bite:
        if is_bite:
            doc_total_net = ext.get("total_net")
            doc_total_vat = ext.get("total_vat")
            doc_total_gross = ext.get("total_gross")
            bite_items = []
            for li in ext.get("line_items", []):
                conn_id = li.get("connection_id") or li.get("phone_number") or ""
                sub = li.get("subtotal_eur", 0) or 0
                if sub == 0:
                    continue
                bite_items.append((conn_id, round(sub, 2)))
            sub_sum = round(sum(s for _, s in bite_items), 2)
            # Allocate doc-level net/vat ONLY when subtotals align with doc gross
            can_allocate = (
                doc_total_net is not None
                and doc_total_vat is not None
                and doc_total_gross is not None
                and sub_sum > 0
                and abs(sub_sum - doc_total_gross) <= 0.05
            )
            running_net = 0.0
            running_vat = 0.0
            for idx, (conn_id, sub) in enumerate(bite_items):
                if can_allocate:
                    is_last = (idx == len(bite_items) - 1)
                    if is_last:
                        net = round(doc_total_net - running_net, 2)
                        vat = round(doc_total_vat - running_vat, 2)
                    else:
                        weight = sub / sub_sum
                        net = round(doc_total_net * weight, 2)
                        vat = round(doc_total_vat * weight, 2)
                        running_net = round(running_net + net, 2)
                        running_vat = round(running_vat + vat, 2)
                    gross = round(net + vat, 2)
                else:
                    # Fallback: subtotals don't match (VAT-exempt items present)
                    net = sub
                    gross = sub
                    vat = 0
                desc = f"Mobilo sakaru pakalpojumi \u2014 {conn_id}" if conn_id else "Telefonsakaru pakalpojumi"
                items.append({
                    "description": desc,
                    "total_gross": gross,
                    "net_amount": net,
                    "vat_amount": vat,
                })
        else:
            # Tele2 amounts are GROSS
            for li in ext.get("line_items", []):
                conn_id = li.get("connection_id") or li.get("phone_number") or ""
                sub = li.get("subtotal_eur", 0) or 0
                if sub == 0:
                    continue
                gross = round(sub, 2)
                net = round(sub / 1.21, 2)
                vat = round(gross - net, 2)
                desc = f"Mobilo sakaru pakalpojumi \u2014 {conn_id}" if conn_id else "Telefonsakaru pakalpojumi"
                items.append({
                    "description": desc,
                    "total_gross": gross,
                    "net_amount": net,
                    "vat_amount": vat,
                })
    else:
        # Invoice families: line_items not yet extracted by pdfplumber
        for li in ext.get("line_items", []):
            items.append({
                "description": li.get("description") or li.get("name") or "",
                "total_gross": li.get("total_gross") or li.get("amount"),
                "net_amount": li.get("net_amount") or li.get("net"),
                "vat_amount": li.get("vat_amount") or li.get("vat"),
            })

    # --- Determine totals ---
    total_gross = ext.get("total_gross") or ext.get("current_period_amount") or ext.get("total_payable")
    total_net = ext.get("total_net")
    total_vat = ext.get("total_vat")
    booking = ext.get("booking_amount") or total_gross

    # --- Build unified extracted_data ---
    unified_ext = {
        "invoice_type": ext.get("invoice_type", "invoice"),
        "seller_name": ext.get("seller_name"),
        "invoice_number": ext.get("invoice_number"),
        "invoice_date": ext.get("invoice_date"),
        "currency": ext.get("currency", "EUR"),
        "total_gross": total_gross,
        "total_net": total_net,
        "total_vat": total_vat,
        "vat_rate": ext.get("vat_rate"),
        "total_discount": ext.get("total_discount", 0) or 0,
        "booking_amount": booking,
        "reverse_charge": ext.get("reverse_charge", False),
        "non_vat_payer": ext.get("non_vat_payer", False),
        "items": items,
        "device_installments": ext.get("device_installments"),
        "late_fee": ext.get("late_fee"),
        "settlement_charges": ext.get("settlement_charges"),
        "_connections_include_device_late": is_tele2,
        "_amounts_are_gross": is_tele2,
        "warnings": ext.get("warnings", []),
        "review_reason": ext.get("review_reason"),
        "confidence": ext.get("confidence", "medium"),
        "triangle_status": ext.get("triangle_status", "NEEDS_REVIEW"),
        "extraction_method": "pdfplumber-deterministic",
    }

    # --- Build top-level response ---
    return {
        "status": master_result.get("status", "FAILED"),
        "document_family": family,
        "source_file": master_result.get("source_file", ""),
        "error_message": master_result.get("error_message"),
        "extracted_data": unified_ext,
    }

# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("PDF_SERVER_API_KEY", "dev-key-local")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Telecom OCR Bridge",
    description=(
        "Deterministic telecom bill extractor. "
        "Accepts PDF via source_file path (local) or base64 content (cloud). "
        "Returns normalized JSON conforming to the telecom bridge output contract."
    ),
    version="1.0.0",
)

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.url.path in ("/health", "/"):
        return await call_next(request)
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or missing API key"}
        )
    return await call_next(request)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    # One of these two must be provided:
    source_file: Optional[str] = None          # Absolute path — for local/dev use
    file_content_base64: Optional[str] = None  # Base64-encoded PDF — for cloud use

    # Pass-through metadata (all optional)
    file_name: Optional[str] = None
    mime_type: Optional[str] = None
    object_name: Optional[str] = None
    person_name: Optional[str] = None
    source_type: Optional[str] = None
    metadata: Optional[dict] = None

    model_config = {"extra": "allow"}  # Forward unknown fields to bridge


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["info"])
def root():
    return {
        "service": "OCR Bridge",
        "version": "2.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /extract": "Unified extraction — auto-classify and extract (Section 12 schema)",
            "POST /process": "Extract telecom bill data from PDF",
            "POST /process-invoice": "Extract digital B2B invoice data from PDF",
            "POST /process-scanned-invoice": "Extract scanned invoice data from PDF via OCR",
            "POST /process-receipt": "Extract receipt data from image via OCR",
            "POST /process-master": "Auto-classify and extract from any supported document",
            "POST /preprocess-receipt": "Crop receipt-paper band + tile under Vision pixel budget (_RECEIPT_TILING_v1)",
        },
        "usage": {
            "local": "POST any endpoint with source_file",
            "cloud": "Same endpoints with file_content_base64 (base64-encoded file)",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    return HealthResponse(
        status="ok",
        service="Telecom OCR Bridge",
        version="1.0.0",
    )


@app.post("/process", tags=["extraction"])
def process(req: ProcessRequest):
    """
    Extract telecom bill data from a PDF file.

    Accepts either:
    - source_file: absolute path to a PDF on the service host (local/dev)
    - file_content_base64: base64-encoded PDF content (cloud deployment)

    Returns the normalized telecom bridge output contract.
    """

    tmp_path: Optional[Path] = None

    try:
        if req.file_content_base64:
            # Decode base64 PDF → write to temp file → process
            try:
                pdf_bytes = base64.b64decode(req.file_content_base64)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"file_content_base64 is not valid base64: {exc}",
                )

            suffix = f"_{req.file_name}" if req.file_name else ".pdf"
            fd, tmp_str = tempfile.mkstemp(
                prefix="ocr_bridge_",
                suffix=suffix,
                dir=tempfile.gettempdir(),
            )
            os.close(fd)
            tmp_path = Path(tmp_str)
            tmp_path.write_bytes(pdf_bytes)

            source_file = str(tmp_path)

        elif req.source_file:
            source_file = req.source_file

        else:
            raise HTTPException(
                status_code=400,
                detail="Either source_file or file_content_base64 must be provided.",
            )

        # Build payload for the existing bridge (same contract as before)
        payload = {
            "source_file": source_file,
            "file_name":   req.file_name,
            "mime_type":   req.mime_type,
            "object_name": req.object_name,
            "person_name": req.person_name,
            "source_type": req.source_type,
            "metadata":    req.metadata or {},
        }

        result = process_telecom_bridge(payload)
        return result

    finally:
        # Always delete the temp file if we created one
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/process-invoice", tags=["extraction"])
def process_invoice(req: ProcessRequest):
    """
    Extract digital B2B invoice data from a PDF file.

    Accepts either:
    - source_file: absolute path to a PDF on the service host (local/dev)
    - file_content_base64: base64-encoded PDF content (cloud deployment)

    Returns the normalized digital invoice bridge output contract.
    """
    tmp_path: Optional[Path] = None

    try:
        if req.file_content_base64:
            try:
                pdf_bytes = base64.b64decode(req.file_content_base64)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"file_content_base64 is not valid base64: {exc}",
                )

            suffix = f"_{req.file_name}" if req.file_name else ".pdf"
            fd, tmp_str = tempfile.mkstemp(
                prefix="ocr_invoice_",
                suffix=suffix,
                dir=tempfile.gettempdir(),
            )
            os.close(fd)
            tmp_path = Path(tmp_str)
            tmp_path.write_bytes(pdf_bytes)
            source_file = str(tmp_path)

        elif req.source_file:
            source_file = req.source_file

        else:
            raise HTTPException(
                status_code=400,
                detail="Either source_file or file_content_base64 must be provided.",
            )

        payload = {
            "source_file": source_file,
            "file_name":   req.file_name,
            "mime_type":   req.mime_type,
            "object_name": req.object_name,
            "person_name": req.person_name,
            "source_type": req.source_type,
            "metadata":    req.metadata or {},
        }

        return process_invoice_bridge(payload)

    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/process-receipt", tags=["extraction"])
def process_receipt(req: ProcessRequest):
    """
    Extract receipt data from an image file via OCR (Tesseract).

    Accepts either:
    - source_file: absolute path to an image on the service host (local/dev)
    - file_content_base64: base64-encoded image content (cloud deployment)

    Returns the normalized receipt bridge output contract.
    """
    tmp_path: Optional[Path] = None

    try:
        if req.file_content_base64:
            try:
                img_bytes = base64.b64decode(req.file_content_base64)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"file_content_base64 is not valid base64: {exc}",
                )

            suffix = f"_{req.file_name}" if req.file_name else ".jpg"
            fd, tmp_str = tempfile.mkstemp(
                prefix="ocr_receipt_",
                suffix=suffix,
                dir=tempfile.gettempdir(),
            )
            os.close(fd)
            tmp_path = Path(tmp_str)
            tmp_path.write_bytes(img_bytes)
            source_file = str(tmp_path)

        elif req.source_file:
            source_file = req.source_file

        else:
            raise HTTPException(
                status_code=400,
                detail="Either source_file or file_content_base64 must be provided.",
            )

        payload = {
            "source_file": source_file,
            "file_name":   req.file_name,
            "mime_type":   req.mime_type,
            "object_name": req.object_name,
            "person_name": req.person_name,
            "source_type": req.source_type,
            "metadata":    req.metadata or {},
        }

        return process_receipt_bridge(payload)

    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/process-scanned-invoice", tags=["extraction"])
def process_scanned_invoice(req: ProcessRequest):
    """
    Extract scanned invoice data from a PDF file via OCR (Tesseract).

    Accepts either:
    - source_file: absolute path to a PDF on the service host (local/dev)
    - file_content_base64: base64-encoded PDF content (cloud deployment)

    Returns the normalized scanned invoice bridge output contract.
    """
    tmp_path: Optional[Path] = None

    try:
        if req.file_content_base64:
            try:
                pdf_bytes = base64.b64decode(req.file_content_base64)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"file_content_base64 is not valid base64: {exc}",
                )

            suffix = f"_{req.file_name}" if req.file_name else ".pdf"
            fd, tmp_str = tempfile.mkstemp(
                prefix="ocr_scan_",
                suffix=suffix,
                dir=tempfile.gettempdir(),
            )
            os.close(fd)
            tmp_path = Path(tmp_str)
            tmp_path.write_bytes(pdf_bytes)
            source_file = str(tmp_path)

        elif req.source_file:
            source_file = req.source_file

        else:
            raise HTTPException(
                status_code=400,
                detail="Either source_file or file_content_base64 must be provided.",
            )

        payload = {
            "source_file": source_file,
            "file_name":   req.file_name,
            "mime_type":   req.mime_type,
            "object_name": req.object_name,
            "person_name": req.person_name,
            "source_type": req.source_type,
            "metadata":    req.metadata or {},
        }

        return process_scanned_invoice_bridge(payload)

    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/process-master", tags=["extraction"])
def process_master(req: ProcessRequest):
    """
    Auto-classify a document and extract structured data.

    Classifies the file into one of: telecom_pdf, digital_invoice_pdf,
    scanned_invoice_pdf, receipt_image — then dispatches to the correct
    family bridge. Accepts any supported extension (.pdf, .jpeg, .jpg,
    .png, .bmp, .tiff, .tif, .webp).

    Accepts either:
    - source_file: absolute path to a file on the service host (local/dev)
    - file_content_base64: base64-encoded content + file_name with extension (cloud)

    Returns the master router output contract (classification + extracted_data).
    """
    tmp_path: Optional[Path] = None

    try:
        if req.file_content_base64:
            try:
                file_bytes = base64.b64decode(req.file_content_base64)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"file_content_base64 is not valid base64: {exc}",
                )

            suffix = f"_{req.file_name}" if req.file_name else ".bin"
            fd, tmp_str = tempfile.mkstemp(
                prefix="ocr_master_",
                suffix=suffix,
                dir=tempfile.gettempdir(),
            )
            os.close(fd)
            tmp_path = Path(tmp_str)
            tmp_path.write_bytes(file_bytes)
            source_file = str(tmp_path)

        elif req.source_file:
            source_file = req.source_file

        else:
            raise HTTPException(
                status_code=400,
                detail="Either source_file or file_content_base64 must be provided.",
            )

        payload = {
            "source_file": source_file,
            "file_name":   req.file_name,
            "mime_type":   req.mime_type,
            "object_name": req.object_name,
            "person_name": req.person_name,
            "source_type": req.source_type,
            "metadata":    req.metadata or {},
        }

        return process_master_bridge(payload)

    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/extract-text", tags=["extraction"])
def extract_text(req: ProcessRequest):
    """
    Raw text extraction only. No classification, no field extraction.
    Returns full PDF text + tables as clean JSON block.
    For use by MB subworkflow to replace Mistral OCR.
    """
    tmp_path = None
    try:
        if req.file_content_base64:
            file_bytes = base64.b64decode(req.file_content_base64)
            suffix = f"_{req.file_name}" if req.file_name else ".pdf"
            fd, tmp_str = tempfile.mkstemp(prefix="ocr_text_", suffix=suffix)
            os.close(fd)
            tmp_path = Path(tmp_str)
            tmp_path.write_bytes(file_bytes)
            source_file = str(tmp_path)
        elif req.source_file:
            source_file = req.source_file
        else:
            raise HTTPException(status_code=400, detail="source_file or file_content_base64 required")

        pages_text = []
        page_tables = {}  # page_index -> list of (table_index, rows)

        with pdfplumber.open(source_file) as pdf:
            tbl_counter = 0
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_text.append(text)
                tables = page.extract_tables() or []
                for table in tables:
                    page_tables.setdefault(i, []).append((tbl_counter, table))
                    tbl_counter += 1

        # Build Mistral-compatible pages array
        mistral_pages = []
        for i, text in enumerate(pages_text):
            page_tbls = []
            for tbl_idx, rows in page_tables.get(i, []):
                # Convert row arrays to HTML table
                html = "<table>"
                for ri, row in enumerate(rows):
                    tag = "th" if ri == 0 else "td"
                    cells = "".join(f"<{tag}>{cell or ''}</{tag}>" for cell in row)
                    html += f"<tr>{cells}</tr>"
                html += "</table>"
                tbl_id = f"tbl-{tbl_idx}.html"
                page_tbls.append({"id": tbl_id, "content": html})
            mistral_pages.append({
                "index": i,
                "markdown": text,
                "tables": page_tbls,
            })

        # Classify document family (lightweight call on the same temp file)
        document_family = "unknown"
        try:
            from tools.extract_master_router import classify_document
            classification = classify_document(source_file, "")
            document_family = classification.get("family", "unknown")
        except Exception:
            pass

        return {
            "pages": mistral_pages,
            "page_count": len(mistral_pages),
            "document_family": document_family,
            "file_name": req.file_name or ""
        }
    finally:
        if tmp_path:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/extract-raw", tags=["extraction"])
def extract_raw(req: ProcessRequest):
    """
    Pure raw PDF dump. No interpretation, no field extraction, no pattern matching.
    Returns every word and every table cell exactly as pdfplumber reads them.
    """
    tmp_path: Optional[Path] = None
    try:
        if req.file_content_base64:
            try:
                file_bytes = base64.b64decode(req.file_content_base64)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"file_content_base64 is not valid base64: {exc}")
            suffix = f"_{req.file_name}" if req.file_name else ".pdf"
            fd, tmp_str = tempfile.mkstemp(prefix="ocr_raw_", suffix=suffix, dir=tempfile.gettempdir())
            os.close(fd)
            tmp_path = Path(tmp_str)
            tmp_path.write_bytes(file_bytes)
            source_file = str(tmp_path)
        elif req.source_file:
            source_file = req.source_file
        else:
            raise HTTPException(status_code=400, detail="Either source_file or file_content_base64 must be provided.")

        pages_out = []
        with pdfplumber.open(source_file) as pdf:
            for i, page in enumerate(pdf.pages):
                # 1. Full text (backward-compatible)
                text = page.extract_text() or ""

                # 2. Text lines with bounding boxes
                raw_lines = page.extract_text_lines(strip=True, return_chars=False)
                text_lines = [
                    {
                        "text": ln["text"],
                        "x0": round(ln["x0"], 2),
                        "x1": round(ln["x1"], 2),
                        "top": round(ln["top"], 2),
                        "bottom": round(ln["bottom"], 2),
                    }
                    for ln in raw_lines
                ]

                # 3. Words with font metadata
                raw_words = page.extract_words(
                    extra_attrs=["fontname", "size"],
                    return_chars=False,
                )
                words = [
                    {
                        "text": w["text"],
                        "x0": round(w["x0"], 2),
                        "top": round(w["top"], 2),
                        "x1": round(w["x1"], 2),
                        "bottom": round(w["bottom"], 2),
                        "fontname": w.get("fontname", ""),
                        "size": round(w.get("size", 0), 1),
                    }
                    for w in raw_words
                ]

                # 4. Tables with bounding boxes + cell text rows
                found_tables = page.find_tables() or []
                raw_table_data = page.extract_tables() or []
                tables_out = []
                for ti, tbl in enumerate(found_tables):
                    tbl_dict = {"bbox": [round(c, 2) for c in tbl.bbox]}
                    if ti < len(raw_table_data):
                        tbl_dict["rows"] = raw_table_data[ti]
                    tables_out.append(tbl_dict)

                pages_out.append({
                    "page": i + 1,
                    "width": round(page.width, 2),
                    "height": round(page.height, 2),
                    "text": text,
                    "text_lines": text_lines,
                    "words": words,
                    "tables": tables_out,
                })

        return {
            "file_name": req.file_name or "",
            "page_count": len(pages_out),
            "pages": pages_out,
        }
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/preprocess-receipt", tags=["preprocessing"])
def preprocess_receipt(req: ProcessRequest):
    """
    _RECEIPT_TILING_v1 — crop the receipt-paper band and tile tall crops so
    every tile stays under the Anthropic Vision ~1.15MP server-side downscale
    budget (nothing gets downscaled -> ~1.2-1.8x effective text size).

    Contract: NEVER fails the caller. On any internal error (or
    RECEIPT_PREPROCESS=off, or PDF input) returns the original base64 as a
    single tile with fallback=true so the bridge behaves exactly as today.
    """
    if not req.file_content_base64:
        raise HTTPException(status_code=400, detail="file_content_base64 required")
    from tools.receipt_preprocess import preprocess_receipt_image
    return preprocess_receipt_image(req.file_content_base64, req.mime_type)


@app.post("/extract", tags=["extraction"])
def extract(req: ProcessRequest):
    """
    Unified extraction endpoint. Auto-classifies document and returns
    the Section 12 unified schema (same structure for all families).

    Accepts either:
    - source_file: absolute path to a file on the service host (local/dev)
    - file_content_base64: base64-encoded content + file_name with extension (cloud)

    Returns the unified extracted_data contract matching the n8n pipeline schema.
    """
    tmp_path: Optional[Path] = None

    try:
        if req.file_content_base64:
            try:
                file_bytes = base64.b64decode(req.file_content_base64)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"file_content_base64 is not valid base64: {exc}",
                )

            suffix = f"_{req.file_name}" if req.file_name else ".bin"
            fd, tmp_str = tempfile.mkstemp(
                prefix="ocr_extract_",
                suffix=suffix,
                dir=tempfile.gettempdir(),
            )
            os.close(fd)
            tmp_path = Path(tmp_str)
            tmp_path.write_bytes(file_bytes)
            source_file = str(tmp_path)

        elif req.source_file:
            source_file = req.source_file

        else:
            raise HTTPException(
                status_code=400,
                detail="Either source_file or file_content_base64 must be provided.",
            )

        payload = {
            "source_file": source_file,
            "file_name":   req.file_name,
            "mime_type":   req.mime_type,
            "object_name": req.object_name,
            "person_name": req.person_name,
            "source_type": req.source_type,
            "metadata":    req.metadata or {},
        }

        raw_result = process_master_bridge(payload)
        return _normalize_to_unified_schema(raw_result)

    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("OCR_BRIDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("OCR_BRIDGE_PORT", "8000")))

    print(f"Starting Telecom OCR Bridge on {host}:{port}")
    print(f"Workspace: {WORKSPACE}")
    print("Press Ctrl+C to stop.")

    uvicorn.run(
        "tools.ocr_bridge_service:app",
        host=host,
        port=port,
        reload=False,
    )
