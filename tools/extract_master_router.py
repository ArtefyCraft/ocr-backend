#!/usr/bin/env python3
"""
extract_master_router.py
-------------------------
Document classification for the master router.

Determines document family from file properties + content:
  - telecom_pdf
  - digital_invoice_pdf
  - scanned_invoice_pdf
  - receipt_image
  - unknown

Strategy:
  1. Physical type: file extension → image vs PDF vs unknown
  2. Images → receipt_image (high confidence)
  3. PDFs:
     a. Extract text with pdfplumber
     b. If text < MIN_TEXT_CHARS → scanned_invoice_pdf (no text layer)
     c. Score telecom signals vs invoice signals in extracted text
     d. Dominant score with gap ≥ MIN_SCORE_GAP → family assigned
     e. Tied/low scores → unknown (NEEDS_REVIEW)
  4. source_type hint: applies a +1 boost to corresponding signal score,
     but content score always takes precedence when decisive.
"""

import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS   = {".pdf"}

MIN_TEXT_CHARS  = 50     # below this → treat PDF as image-only (scanned)
MIN_SCORE       = 2      # minimum score to claim a family
MIN_SCORE_GAP   = 1      # winner must beat runner-up by at least this

# ---------------------------------------------------------------------------
# Telecom signal patterns (weighted)
# ---------------------------------------------------------------------------

_TELECOM_SIGNALS = [
    # Provider names — strong indicator
    (r"\bTele2\b",                                        3),
    (r"\bLMT\b",                                          3),
    (r"\bLattelecom\b",                                   3),
    (r"\bTet\s+SIA\b",                                    3),
    (r"Bite\s+Latvija",                                   2),  # lower: BITE also issues product invoices
    # Document markers
    (r"DFC_DOCTYPENAME\s*:\s*TELE",                       4),
    (r"Periods?\s+\d{2}\.\d{2}\.\d{4}\s*[-–]\s*\d{2}",  3),  # billing period
    # Telecom-specific service terms
    (r"\babonements\b",                                   2),
    (r"\bsarunu\s+laiks\b",                               2),
    (r"\bdatu\s+pakete\b",                                2),
    (r"\bmobil[ao]\s+internets?\b",                       2),
    (r"\bplatjoslas?\b",                                  2),
    (r"\bSMS\s+\d",                                       1),
    (r"\bzvani\b",                                        1),
    (r"\bpakalpojum[ai]\b",                               1),
    # BITE-specific telecom signals
    (r"Balss\s+un\s+interneta",                           4),  # "Balss un interneta pakalpojumi" section header
    (r"Tarifu\s+pl.{1,3}ns",                              3),  # tariff plan column
    (r"manabite\.lv|www\.bite\.lv",                       3),  # BITE self-service portal
    (r"itunes\.com/bill",                                 2),  # Apple charges via telecom
    (r"Klienta\s+Nr\.\s+\d{6,}",                         2),  # telecom customer number (not invoice nr.)
    (r"Biznesa\s+mened.era",                              2),  # "biznesa menedžera" — telecom account mgr
    (r"\b371\d{8}\b",                                     1),  # Latvian phone number in rows
]

# ---------------------------------------------------------------------------
# Invoice signal patterns (weighted)
# ---------------------------------------------------------------------------

_INVOICE_SIGNALS = [
    # Invoice number / type markers
    (r"R.{0,3}[Ii][Nn][Ss]\s+Nr\.?",                     3),  # Rēķins Nr.
    (r"Invoice\s+(?:number|#)",                           3),
    (r"Avansa\s+r.{0,3}ins",                              3),  # advance invoice
    (r"Pavadz.{1,3}me\s+[-–]?\s*R.{0,3}ins",             3),  # delivery-invoice
    (r"R.{0,3}INS[-–]FAKT.{0,3}RA",                      3),  # invoice-factura
    (r"\bPAVADZ.{1,3}ME\b",                               3),  # standalone PAVADZĪME header
    # Seller/buyer structure
    (r"Pieg.d.t.js",                                      2),  # Piegādātājs (seller)
    (r"Izpild.t.js",                                      2),  # Izpildītājs
    (r"PIEG.D.T.JS",                                      2),
    (r"Sa.{1,3}m.js",                                     1),  # Saņēmējs (buyer)
    # Tax / VAT identifiers typical of B2B invoices
    (r"PVN\s+(?:maks.t.ja\s+)?kods?\s+LV\d{11}",         2),
    (r"Re.istr.cijas\s+numurs",                           1),
    # Amount labels specific to B2B invoices
    (r"Kop.j.\s+summa",                                   2),
    (r"Summa\s+apmaksai",                                 2),
    (r"Pavisam\s+apmaksai",                               2),
    (r"KOPA\s+EUR",                                       1),
    (r"Summa\s+bez\s+PVN",                                2),  # net amount label on invoices
    (r"Bankas\s+p.rskait.jums",                           2),  # bank transfer — invoices only
    # English invoice markers (Google, Canva, etc.)
    (r"Invoice\s+date",                                   2),
    (r"Provided\s+by",                                    1),
    (r"Due\s+date",                                       1),
]

# ---------------------------------------------------------------------------
# _CLUSTER_A_v1 — Verbatim classifier prescan (LV document-family tokens)
#
# Runs AFTER OCR (image path) or PDF text extraction (PDF path) inside
# `classify_document`. Scans the header region for a whitelisted LV token;
# on hit, emits `verbatim_prescan={verbatim_hit, route_hint, matched_token,
# fuzzy_distance, matched_line}` on the classification dict. Downstream
# `process_master_bridge` decides shadow (log-only) vs enforce (override
# routing) based on env `VERBATIM_CLASSIFIER=off|shadow|enforce`.
#
# Design fixes from workflow-1 adversarial critic:
#   - P0-1 chicken-and-egg SOLVED: scanner runs INSIDE classify_document
#     which has already produced OCR text (image L242, PDF L282).
#   - P0-3 z/x collision: kept exact-match-only (dist=0) since 'z-atskaite'
#     vs 'x-atskaite' differ by exactly 1 char.
#   - Cluster A halt Option A (2026-08-11 halt decision): scanner in bridge
#     post-OCR, before family dispatch — this file.
# ---------------------------------------------------------------------------

# Whitelist: [normalized_token, route_hint, standalone_only, allow_fuzzy]
# Order matters — multi-word first, most specific first.
_VERBATIM_WHITELIST: list[tuple[str, str, bool, bool]] = [
    ("savstarpējais ieskaits", "credit_note",     False, True),
    ("ieskaita akts",          "credit_note",     False, True),
    ("kredīta rēķins",         "credit_note",     False, True),
    ("kredītrēķins",           "credit_note",     False, True),
    ("avansa rēķins",          "advance_invoice", False, True),
    ("priekšapmaksa",          "advance_invoice", False, True),
    ("pro forma",              "advance_invoice", False, True),
    ("pro-forma",              "advance_invoice", False, True),
    # z/x — EXACT match only (dist=0) — critic P0-3: fuzzy-1 collision
    ("z-atskaite",             "receipt_z",       False, False),
    ("x-atskaite",             "receipt_x",       False, False),
    # standalone-only guarded — 'pavadzīme' collides with 'rēķins-pavadzīme'
    ("pavadzīme",              "delivery_note",   True,  True),
]

_HEADER_CHAR_LIMIT = 800
_HEADER_LINE_LIMIT = 15


def _lev(a: str, b: str) -> int:
    """Levenshtein distance capped at |len(a)-len(b)| shortcut; O(len(a)*len(b))."""
    if not a or not b:
        return 99
    m, n = len(a), len(b)
    if abs(m - n) > 3:
        return abs(m - n)
    prev_row = list(range(n + 1))
    for i in range(1, m + 1):
        cur_row = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur_row[j] = min(cur_row[j - 1] + 1, prev_row[j] + 1, prev_row[j - 1] + cost)
        prev_row = cur_row
    return prev_row[n]


def _normalize_line(s: str) -> str:
    # Lowercase, collapse whitespace, strip zero-width, keep LV diacritics.
    return re.sub(r"\s+", " ", re.sub(r"[​-‍﻿]", "", s.lower())).strip()


def _header_lines(text: str) -> list[str]:
    clipped = text[:_HEADER_CHAR_LIMIT]
    lines = [_normalize_line(ln) for ln in clipped.splitlines()[:_HEADER_LINE_LIMIT]]
    return [ln for ln in lines if ln]


def _scan_line(nline: str, token: str, allow_fuzzy: bool) -> dict:
    """Return {dist, exact} for the best match of `token` inside `nline`."""
    if token in nline:
        return {"dist": 0, "exact": True}
    if not allow_fuzzy:
        return {"dist": 99, "exact": False}
    tl = len(token)
    best = 99
    # Slide a window of length tl±1 across the line, keep best distance
    for w in range(max(3, tl - 1), tl + 2):
        for i in range(0, max(0, len(nline) - w + 1)):
            chunk = nline[i : i + w]
            # cheap 3-gram prefilter
            if tl >= 4 and len(chunk) >= 4:
                if chunk[:3] != token[:3] and chunk[-3:] != token[-3:]:
                    continue
            d = _lev(chunk, token)
            if d < best:
                best = d
            if best == 0:
                return {"dist": 0, "exact": True}
    return {"dist": best, "exact": False}


def _passes_standalone_guard(nline: str, token: str) -> bool:
    """`pavadzīme` must NOT be part of `rēķins-pavadzīme` (hybrid invoice)."""
    if token != "pavadzīme":
        return True
    if re.search(r"\br[ēe]ķins\b", nline):
        return False
    if re.search(r"\brēķins-?pavadzīme\b", nline):
        return False
    if re.search(r"\bkv[īi]ts\b", nline):
        return False
    return True


def verbatim_prescan(text: str) -> dict:
    """
    Scan the header region of `text` for a whitelisted LV document-family
    token. First hit wins (whitelist ordered specific → general).

    Returns:
      {
        "verbatim_hit":    bool,
        "route_hint":      str | None,   # credit_note | advance_invoice | delivery_note | receipt_z | receipt_x
        "matched_token":   str | None,
        "fuzzy_distance":  int | None,   # 0 = exact, 1 = one edit
        "matched_line":    str | None,
        "standalone_only": bool | None,
        "scanner_version": "cluster_a_v1_shadow"
      }

    Pure function; never raises. Empty text = no-op (verbatim_hit=False).
    """
    empty = {
        "verbatim_hit": False,
        "route_hint": None,
        "matched_token": None,
        "fuzzy_distance": None,
        "matched_line": None,
        "standalone_only": None,
        "scanner_version": "cluster_a_v1_shadow",
    }
    if not text:
        return empty
    for nline in _header_lines(text):
        for token, route, standalone_only, allow_fuzzy in _VERBATIM_WHITELIST:
            res = _scan_line(nline, token, allow_fuzzy)
            if res["dist"] <= (1 if allow_fuzzy else 0) and _passes_standalone_guard(nline, token):
                return {
                    "verbatim_hit": True,
                    "route_hint": route,
                    "matched_token": token,
                    "fuzzy_distance": res["dist"],
                    "matched_line": nline,
                    "standalone_only": standalone_only,
                    "scanner_version": "cluster_a_v1_shadow",
                }
    return empty


# ---------------------------------------------------------------------------
# Physical type
# ---------------------------------------------------------------------------

def detect_physical_type(path: str) -> str:
    """Return 'image', 'pdf', or 'unknown' based on file extension."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    return "unknown"


# ---------------------------------------------------------------------------
# PDF text extraction (classification only — lightweight)
# ---------------------------------------------------------------------------

def _extract_pdf_text_for_classification(pdf_path: str, max_pages: int = 3) -> tuple[str, int]:
    """
    Extract up to max_pages of text for classification purposes.
    Returns (text, page_count). Returns ("", 0) on error.
    """
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages[:max_pages]:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts), page_count
    except Exception:
        return "", 0


def _ocr_image_for_classification(image_path: str) -> str:
    """
    Quick Tesseract pass on an image for content-based classification.
    Returns empty string if Tesseract is unavailable or errors.
    """
    try:
        import pytesseract
        from PIL import Image
        import os
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        os.environ.setdefault(
            "TESSDATA_PREFIX",
            str(Path(__file__).parent.parent / "tessdata"),
        )
        img = Image.open(image_path).convert("RGB")
        return pytesseract.image_to_string(img, lang="lav+eng", config="--oem 3 --psm 6")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def _score_signals(text: str, signals: list) -> int:
    """Sum weights of all matching signal patterns in text."""
    score = 0
    for pattern, weight in signals:
        if re.search(pattern, text, re.I):
            score += weight
    return score


# ---------------------------------------------------------------------------
# Source-type hint mapping
# ---------------------------------------------------------------------------

_SOURCE_TYPE_HINTS = {
    "telecom":            "telecom_pdf",
    "telecom_bill":       "telecom_pdf",
    "telecom_pdf":        "telecom_pdf",
    "digital_invoice":    "digital_invoice_pdf",
    "digital_invoice_pdf":"digital_invoice_pdf",
    "invoice":            "digital_invoice_pdf",
    "scanned_invoice":    "scanned_invoice_pdf",
    "scanned_invoice_pdf":"scanned_invoice_pdf",
    "scanned_pdf":        "scanned_invoice_pdf",
    "receipt":            "receipt_image",
    "receipt_image":      "receipt_image",
    "image_receipt":      "receipt_image",
}


def _hint_from_source_type(source_type: str) -> Optional[str]:
    if not source_type:
        return None
    return _SOURCE_TYPE_HINTS.get(source_type.lower().strip())


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_document(
    file_path: str,
    source_type: str = "",
) -> dict:
    """
    Classify a document into one of the supported families.

    Returns:
      {
        "family":     str,   # telecom_pdf | digital_invoice_pdf | scanned_invoice_pdf | receipt_image | unknown
        "confidence": str,   # high | medium | low
        "reason":     str,   # human-readable explanation
        "physical_type": str,# image | pdf | unknown
        "telecom_score": int,
        "invoice_score": int,
        "text_length":   int,
        "verbatim_prescan": dict,  # _CLUSTER_A_v1 shadow scan result
      }
    """
    path = Path(file_path)
    hint = _hint_from_source_type(source_type)

    # _CLUSTER_A_v1 — captured OCR text for verbatim prescan. Populated by
    # image OCR / PDF text-extract branches below; _finalize attaches it to
    # every return so downstream can decide shadow vs enforce override.
    ocr_text_for_prescan: str = ""

    def _finalize(result: dict) -> dict:
        result["verbatim_prescan"] = verbatim_prescan(ocr_text_for_prescan)
        return result

    # ── Physical type ──────────────────────────────────────────────────────
    phys = detect_physical_type(str(path))

    if phys == "unknown":
        return _finalize({
            "family": "unknown", "confidence": "low",
            "reason": f"Unrecognised file extension '{path.suffix}'",
            "physical_type": phys, "telecom_score": 0, "invoice_score": 0, "text_length": 0,
            "page_count": 0,
        })

    # ── Image: content-based classification via quick OCR pass ────────────
    if phys == "image":
        ocr_text     = _ocr_image_for_classification(str(path))
        inv_score    = _score_signals(ocr_text, _INVOICE_SIGNALS) if ocr_text else 0
        tel_score    = _score_signals(ocr_text, _TELECOM_SIGNALS) if ocr_text else 0
        text_len     = len(ocr_text.replace(" ", "").replace("\n", "")) if ocr_text else 0
        # _CLUSTER_A_v1: capture OCR text so _finalize can prescan it.
        ocr_text_for_prescan = ocr_text or ""

        # Invoice/delivery note signals in image → treat as scanned document.
        # Threshold 3: a lone PVN registration number (2 pts) appears on POS
        # receipts too; only dedicated document-type markers (RĒĶINS / PAVADZĪME
        # = 3 pts each) should trigger reclassification.
        if inv_score >= 3:
            conf   = "medium"   # image OCR less reliable than PDF text layer
            reason = (
                f"Image reclassified as scanned_invoice_pdf by content "
                f"(invoice_score={inv_score}, text_len={text_len})"
            )
            if hint and hint != "scanned_invoice_pdf":
                reason += f" (source_type hint '{source_type}' overridden by content)"
            return _finalize({
                "family": "scanned_invoice_pdf", "confidence": conf, "reason": reason,
                "physical_type": phys, "telecom_score": tel_score,
                "invoice_score": inv_score, "text_length": text_len,
                "page_count": 0,
            })

        # Default: receipt_image
        conf = "high" if hint in (None, "receipt_image") else "medium"
        if text_len > 0:
            reason = f"Image → receipt_image (no invoice signals, invoice_score={inv_score})"
        else:
            reason = "Image → receipt_image (OCR returned no text, assumed receipt)"
        if hint and hint != "receipt_image":
            reason += f" (source_type hint '{source_type}' noted but content overrides)"
        return _finalize({
            "family": "receipt_image", "confidence": conf, "reason": reason,
            "physical_type": phys, "telecom_score": tel_score,
            "invoice_score": inv_score, "text_length": text_len,
            "page_count": 0,
        })

    # ── PDF: extract text for classification ──────────────────────────────
    text, page_count = _extract_pdf_text_for_classification(str(path))
    text_len = len(text.replace(" ", "").replace("\n", ""))
    # _CLUSTER_A_v1: capture text so _finalize can prescan it.
    ocr_text_for_prescan = text or ""

    # No text layer → scanned invoice
    if text_len < MIN_TEXT_CHARS:
        conf = "high" if hint in (None, "scanned_invoice_pdf") else "medium"
        reason = f"PDF has no text layer (chars={text_len}) → scanned_invoice_pdf"
        if hint and hint != "scanned_invoice_pdf":
            reason += f" (source_type hint '{source_type}' overridden by content)"
        return _finalize({
            "family": "scanned_invoice_pdf", "confidence": conf, "reason": reason,
            "physical_type": phys, "telecom_score": 0, "invoice_score": 0, "text_length": text_len,
            "page_count": page_count,
        })

    # Score signals
    telecom_score = _score_signals(text, _TELECOM_SIGNALS)
    invoice_score = _score_signals(text, _INVOICE_SIGNALS)

    # Apply source_type hint boost (+1 to the hinted family's score)
    if hint == "telecom_pdf":
        telecom_score += 1
    elif hint in ("digital_invoice_pdf", "scanned_invoice_pdf"):
        invoice_score += 1

    # Determine family
    gap = abs(telecom_score - invoice_score)

    if telecom_score >= MIN_SCORE and telecom_score > invoice_score and gap >= MIN_SCORE_GAP:
        conf = "high" if telecom_score >= 5 else "medium"
        reason = f"Telecom signals dominant (telecom={telecom_score}, invoice={invoice_score})"
        return _finalize({
            "family": "telecom_pdf", "confidence": conf, "reason": reason,
            "physical_type": phys, "telecom_score": telecom_score,
            "invoice_score": invoice_score, "text_length": text_len,
            "page_count": page_count,
        })

    if invoice_score >= MIN_SCORE and invoice_score > telecom_score and gap >= MIN_SCORE_GAP:
        conf = "high" if invoice_score >= 5 else "medium"
        reason = f"Invoice signals dominant (invoice={invoice_score}, telecom={telecom_score})"
        return _finalize({
            "family": "digital_invoice_pdf", "confidence": conf, "reason": reason,
            "physical_type": phys, "telecom_score": telecom_score,
            "invoice_score": invoice_score, "text_length": text_len,
            "page_count": page_count,
        })

    # Ambiguous
    reason = (
        f"Ambiguous — telecom={telecom_score}, invoice={invoice_score}, "
        f"gap={gap} < {MIN_SCORE_GAP} required"
    )
    if telecom_score == 0 and invoice_score == 0:
        reason = f"No family signals found in PDF text (text_len={text_len})"
    return _finalize({
        "family": "unknown", "confidence": "low", "reason": reason,
        "physical_type": phys, "telecom_score": telecom_score,
        "invoice_score": invoice_score, "text_length": text_len,
        "page_count": page_count,
    })
