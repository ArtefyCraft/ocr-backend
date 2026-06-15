#!/usr/bin/env python3
"""
extract_scanned_invoice_pdf.py
-------------------------------
OCR-based extractor for scanned (image-only) invoice PDFs.

Strategy:
  1. Try pdfplumber text extraction (handles hybrid PDFs that have a sparse text layer).
  2. If no text found → render each page as an image → Tesseract OCR (lav+eng).
  3. Clean OCR artefacts (pipe chars, doubled spaces, stray punctuation).
  4. Apply the same regex field patterns used for digital invoices — OCR text is
     structurally similar to pdfplumber output with added noise.
  5. When a total_gross amount is missing or garbled, derive it from
     total_net + total_vat (triangle) if both are present.
  6. Return confidence=low if OCR quality indicators are poor.

Tesseract setup:
  Binary  : C:/Program Files/Tesseract-OCR/tesseract.exe
  Tessdata: ./tessdata/   (lav.traineddata + eng.traineddata copied here)
  Language: lav+eng  (Latvian primary, English for company names / Google / Canva)

Usage:
  from tools.extract_scanned_invoice_pdf import extract_scanned_invoice
  result = extract_scanned_invoice("samples/invoices/scanned_pdf/laskana_26_024_scan.pdf")

  or directly:
  python tools/extract_scanned_invoice_pdf.py samples/invoices/scanned_pdf/laskana_26_024_scan.pdf
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is required: pip install pdfplumber")

try:
    import pytesseract
    from PIL import Image
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

# ---------------------------------------------------------------------------
# Tesseract configuration
# ---------------------------------------------------------------------------

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR  = str(Path(__file__).parent.parent / "tessdata")
OCR_LANG      = "lav+eng"
OCR_CONFIG    = "--oem 3 --psm 6"   # LSTM + assume uniform block of text

if _HAS_TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

# Re-use field parsers from digital invoice extractor
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.extract_digital_invoice_pdf import (
    _parse_invoice_number,
    _parse_invoice_date,
    _parse_due_date,
    _parse_seller_name,
    _parse_seller_vat,
    _detect_invoice_type,
    _detect_reverse_charge,
    _detect_non_vat_payer,
    _detect_vat_rate,
    _detect_currency,
    _parse_total_net,
    _parse_total_vat,
    _parse_amount,
)


# ---------------------------------------------------------------------------
# OCR text extraction
# ---------------------------------------------------------------------------

def _has_text_layer(pdf_path: str, min_chars: int = 50) -> bool:
    """Return True if pdfplumber finds substantial text in the PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = sum(
                len(p.extract_text() or "") for p in pdf.pages
            )
        return total >= min_chars
    except Exception:
        return False


def _extract_via_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber (digital / hybrid PDFs)."""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def _extract_via_ocr(pdf_path: str, dpi: int = 200) -> tuple[str, str]:
    """
    Render each page as an image and OCR it.
    Returns (text, ocr_engine_used).
    """
    if not _HAS_TESSERACT:
        return "", "tesseract_unavailable"

    parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pi = page.to_image(resolution=dpi)
                pil_img = pi.original.convert("RGB")
                text = pytesseract.image_to_string(
                    pil_img, lang=OCR_LANG, config=OCR_CONFIG
                )
                parts.append(text)
        return "\n".join(parts), "tesseract_lav_eng"
    except Exception as exc:
        return "", f"ocr_error:{exc}"


def _ocr_image_file(img_path: str) -> tuple[str, str]:
    """OCR a standalone image file (JPEG, PNG, etc.)."""
    if not _HAS_TESSERACT:
        return "", "tesseract_unavailable"
    try:
        img = Image.open(img_path).convert("RGB")
        text = pytesseract.image_to_string(img, lang=OCR_LANG, config=OCR_CONFIG)
        return text, "tesseract_lav_eng"
    except Exception as exc:
        return "", f"ocr_error:{exc}"


# ---------------------------------------------------------------------------
# OCR text cleaning
# ---------------------------------------------------------------------------

def _clean_ocr_text(text: str) -> str:
    """
    Normalise common OCR artefacts so existing field-parser regexes match.

    Artefacts seen in practice:
      - Pipe characters | from table borders
      - Multiple consecutive spaces
      - Stray leading/trailing spaces on lines
      - Non-breaking spaces (0xa0)
      - Lone dashes used as cell separators
    """
    # Remove pipe chars (table border artefact)
    text = text.replace("|", " ")
    # Non-breaking space → regular space
    text = text.replace("\xa0", " ")
    # Collapse multiple spaces (but keep newlines)
    text = re.sub(r" {2,}", " ", text)
    # Trim trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Remove lines that are only dashes/equals/asterisks (separator lines)
    text = re.sub(r"^[-=*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text


# ---------------------------------------------------------------------------
# OCR-specific total_gross parser
# ---------------------------------------------------------------------------

def _parse_total_gross_ocr(text: str) -> tuple[Optional[float], str]:
    """
    Extended total_gross parser for OCR text. Handles:
      - Standard patterns (same as digital invoice extractor)
      - Table-cell artefacts where decimal separator is dropped
        e.g. '49343' instead of '493,43'
    Falls back to triangle derivation when direct extraction fails.
    """
    # --- Standard patterns (reused from digital invoice extractor) ---
    patterns = [
        (r"Pavisam apmaksai:\s*EUR\s*([\d.,]+)",           "pavisam_apmaksai"),
        (r"Kop.j.\s*summa\s*\(EUR\)\s*([\d.,\s]+)",        "kopeja_summa"),
        (r"Summa\s+apmaksai\s*\(EUR\)\s*([\d.,]+)",        "summa_apmaksai_eur"),
        (r"KOP.\s*EUR\s*([\d.,]+)",                         "kopa_eur"),
        (r"Kop.\s*apmaksai[:\s]*[€]?\s*([\d.,]+)",         "kopa_apmaksai"),
        (r"Summa\s+apmaksai\s+([\d.,]+)",                  "summa_apmaksai"),
        (r"Kopsumma\s+(?!bez\s+PVN)([\d.,]+)",             "kopsumma"),
        (r"Total\s+in\s+EUR\s*[€$]?\s*([\d.]+)",           "total_in_eur"),
        (r"^Total\s+[\d.]+\s+[\d.]+\s+EUR\s+([\d.]+)",    "canva_total"),
    ]
    for pattern, method in patterns:
        flags = re.MULTILINE if method == "canva_total" else 0
        matches = list(re.finditer(pattern, text, flags))
        if matches:
            raw = matches[-1].group(1).strip()
            val = _parse_amount(raw)
            if val is not None and val > 0:
                return val, method

    # --- OCR artefact: amount digits without decimal separator ---
    # 'Kopējā summa (EUR) \ufffd \ufffd 49343' → garbled table cell → 493.43
    # Use [^\d]* to skip any non-digit junk (including \ufffd replacement chars)
    m = re.search(r"Kop.j.\s*summa\s*\(EUR\)[^\d]*(\d{4,6})\s*$", text, re.MULTILINE)
    if m:
        raw_digits = m.group(1)
        # Last two digits are cents; insert decimal separator
        val = float(raw_digits[:-2] + "." + raw_digits[-2:])
        if 0 < val < 100000:
            return round(val, 2), "kopeja_summa_ocr_digits"

    # --- Triangle derivation ---
    # Caller already tries this after _parse_total_gross_ocr returns None,
    # but expose it here too for completeness (avoids double call in edge cases).

    return None, "not_found"


# ---------------------------------------------------------------------------
# OCR quality indicator
# ---------------------------------------------------------------------------

def _ocr_confidence(text: str) -> str:
    """
    Rough quality estimate based on ratio of garbled chars.
    Returns 'high', 'medium', or 'low'.
    """
    if not text:
        return "low"
    # Count replacement chars and obvious OCR noise
    noise_chars = text.count("\ufffd") + text.count("?") + text.count("~")
    total_chars = len(text.replace("\n", "").replace(" ", "")) or 1
    noise_ratio = noise_chars / total_chars
    if noise_ratio < 0.05:
        return "high"
    if noise_ratio < 0.15:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_scanned_invoice(pdf_path: str) -> dict:
    """
    Extract structured fields from a scanned (image-only) invoice PDF.

    Auto-detects whether OCR is needed:
      - If pdfplumber finds substantial text → use it (hybrid/digital PDF)
      - Otherwise → render pages → Tesseract OCR
    """
    path = Path(pdf_path)

    result = {
        "source_file":    path.name,
        "document_family": "scanned_pdf",
        "invoice_type":   None,
        "invoice_number": None,
        "invoice_date":   None,
        "due_date":       None,
        "seller_name":    None,
        "seller_vat":     None,
        "currency":       "EUR",
        "total_gross":    None,
        "total_net":      None,
        "total_vat":      None,
        "vat_rate":       None,
        "reverse_charge": False,
        "non_vat_payer":  False,
        "booking_amount": None,
        "ocr_engine":     None,
        "extraction_method": "ocr",
        "llm_fallback_used": False,
        "warnings":       [],
        "field_sources":  {},
    }

    warnings: list = result["warnings"]
    src: dict      = result["field_sources"]

    # --- Determine extraction path ---
    try:
        if _has_text_layer(str(path)):
            raw_text = _extract_via_pdfplumber(str(path))
            result["ocr_engine"]         = "pdfplumber"
            result["extraction_method"]  = "pdfplumber_fallback"
        else:
            raw_text, engine = _extract_via_ocr(str(path))
            result["ocr_engine"] = engine
            if not raw_text:
                result["warnings"].append(f"OCR returned empty text: {engine}")
                result["extraction_method"] = "failed"
    except Exception as exc:
        result["warnings"].append(f"Text extraction failed: {exc}")
        result["extraction_method"] = "failed"
        return result

    raw_text = _clean_ocr_text(raw_text)

    # --- VAT flags ---
    result["reverse_charge"] = _detect_reverse_charge(raw_text)
    result["non_vat_payer"]  = _detect_non_vat_payer(raw_text)
    result["vat_rate"]       = _detect_vat_rate(raw_text)
    result["currency"]       = _detect_currency(raw_text)
    result["invoice_type"]   = _detect_invoice_type(raw_text)

    # --- Core fields ---
    result["invoice_number"], src["invoice_number"] = _parse_invoice_number(raw_text)
    result["invoice_date"],   src["invoice_date"]   = _parse_invoice_date(raw_text)
    result["due_date"],       src["due_date"]        = _parse_due_date(raw_text)
    result["seller_name"],    src["seller_name"]     = _parse_seller_name(raw_text)
    result["seller_vat"],     src["seller_vat"]      = _parse_seller_vat(raw_text)

    # --- Amounts ---
    result["total_gross"], src["total_gross"] = _parse_total_gross_ocr(raw_text)
    result["total_net"],   src["total_net"]   = _parse_total_net(raw_text)
    result["total_vat"],   src["total_vat"]   = _parse_total_vat(raw_text)

    # Non-VAT payer: net == gross
    if result["non_vat_payer"] and result["total_net"] is None and result["total_gross"] is not None:
        result["total_net"] = result["total_gross"]
        src["total_net"] = "derived:non_vat_payer"

    # Triangle derivation: fill total_gross if missing but net+vat present
    if result["total_gross"] is None:
        if result["total_net"] is not None and result["total_vat"] is not None:
            result["total_gross"] = round(result["total_net"] + result["total_vat"], 2)
            src["total_gross"] = "derived:net_plus_vat"
            warnings.append("total_gross derived from net+vat (direct extraction failed)")

    # Triangle validation
    triangle_status = "INCOMPLETE"
    if result["total_gross"] is not None:
        if result["total_net"] is not None and result["total_vat"] is not None:
            computed = round(result["total_net"] + result["total_vat"], 2)
            if abs(computed - result["total_gross"]) > 0.05:
                warnings.append(
                    f"Triangle mismatch: net({result['total_net']}) + "
                    f"vat({result['total_vat']}) = {computed} ≠ {result['total_gross']}"
                )
                triangle_status = "NEEDS_REVIEW"
            else:
                triangle_status = "OK"
        else:
            triangle_status = "INCOMPLETE"   # gross known but net/vat missing
    result["triangle_status"] = triangle_status

    # Booking amount
    if result["total_gross"] is not None:
        result["booking_amount"] = result["total_gross"]
        src["booking_amount"] = "derived:total_gross"

    # --- Confidence ---
    ocr_qual  = _ocr_confidence(raw_text)
    n_missing = sum(1 for f in ("invoice_number", "invoice_date", "total_gross", "seller_name")
                    if not result.get(f))
    if triangle_status == "NEEDS_REVIEW" or n_missing > 1:
        result["confidence"] = "low"
    elif n_missing == 1 or warnings or ocr_qual == "low":
        result["confidence"] = "medium"
    else:
        result["confidence"] = "high" if ocr_qual != "medium" else "medium"

    # --- Warnings for missing required fields ---
    for field in ("invoice_number", "invoice_date", "total_gross", "seller_name"):
        if not result.get(field):
            warnings.append(f"{field}: extraction failed")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_scanned_invoice_pdf.py <path_to_pdf>")
        sys.exit(1)
    result = extract_scanned_invoice(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))
