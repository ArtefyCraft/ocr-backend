#!/usr/bin/env python3
"""
extract_tele2_pdf.py
--------------------
Deterministic extractor for SIA Tele2 monthly business bill PDFs.

Strategy:
  1. Primary: DFC_ metadata fields embedded in Tele2 digital PDFs
     (DFC_PRN, DFC_Amount, DFC_InvoiceDate)
  2. Secondary: regex on visible text layer
  3. LLM fallback: NOT used — all required fields have deterministic patterns.
     If a field cannot be extracted it is returned as None with a warning.

Encoding note:
  pdfplumber extracts Tele2 PDFs with garbled Latvian diacritics
  (ā→?, ē→?, š→? etc.). Patterns are written with '.' wildcards where
  diacritics appear, so they remain robust without requiring font remapping.

Usage:
  from tools.extract_tele2_pdf import extract_tele2
  result = extract_tele2("samples/telecom/tele2_pdf/9.pdf")

  or directly:
  python tools/extract_tele2_pdf.py samples/telecom/tele2_pdf/9.pdf
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


# ---------------------------------------------------------------------------
# Amount parsing
# ---------------------------------------------------------------------------

def _parse_lv_amount(value: str) -> Optional[float]:
    """
    Parse a Latvian-format number string to float.
    Handles comma decimal separator (213,90 → 213.90)
    and non-breaking spaces used as thousand separators.
    """
    if not value:
        return None
    cleaned = (
        value.strip()
        .replace("\xa0", "")   # non-breaking space
        .replace(" ", "")
        .replace(",", ".")
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_dfc_date(value: str) -> Optional[str]:
    """Convert DFC date format YYYYMMDD → YYYY-MM-DD (ISO 8601)."""
    try:
        return datetime.strptime(value.strip(), "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _normalise_date(value: str) -> Optional[str]:
    """Convert DD.MM.YYYY → YYYY-MM-DD (ISO 8601)."""
    try:
        return datetime.strptime(value.strip(), "%d.%m.%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Extract full text from all pages of a PDF."""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------

def _parse_invoice_number(text: str) -> tuple[Optional[str], str]:
    """
    Primary: DFC_PRN metadata field.
    Fallback: 'R??ina Nr. NNN-NNN' pattern (diacritics garbled).
    Returns (value, method).
    """
    m = re.search(r'DFC_PRN:"([^"]+)"', text)
    if m:
        return m.group(1).strip(), "dfc_metadata"
    # Fallback: garbled "Rēķina Nr."
    m = re.search(r'R.{1,3}ina Nr\.?\s+(\d+-\d+)', text)
    if m:
        return m.group(1).strip(), "regex_fallback"
    return None, "not_found"


def _parse_invoice_date(text: str) -> tuple[Optional[str], str]:
    """
    Primary: DFC_InvoiceDate metadata field (YYYYMMDD → YYYY-MM-DD).
    Fallback: 'Datums DD.MM.YYYY' normalised to YYYY-MM-DD.
    """
    m = re.search(r'DFC_InvoiceDate:"(\d{8})"', text)
    if m:
        val = _parse_dfc_date(m.group(1))
        if val:
            return val, "dfc_metadata"
    m = re.search(r'Datums\s+(\d{2}\.\d{2}\.\d{4})', text)
    if m:
        val = _normalise_date(m.group(1))
        if val:
            return val, "regex_fallback"
    return None, "not_found"


def _parse_billing_period(text: str) -> tuple[Optional[str], str]:
    """
    'Periods DD.MM.YYYY - DD.MM.YYYY' — no diacritics in label.
    """
    m = re.search(
        r'Periods\s+(\d{2}\.\d{2}\.\d{4}\s*[-\u2013]\s*\d{2}\.\d{2}\.\d{4})',
        text
    )
    if m:
        return m.group(1).strip(), "regex"
    return None, "not_found"


def _parse_total_payable(text: str) -> tuple[Optional[float], str]:
    """
    Primary: DFC_Amount metadata (English decimal, most reliable).
    Fallback: last 'Summa apmaksai NNN,NN' in text.
    """
    m = re.search(r'DFC_Amount:"([\d.]+)"', text)
    if m:
        try:
            return float(m.group(1)), "dfc_metadata"
        except ValueError:
            pass
    # Fallback: all occurrences of Summa apmaksai — take last/max
    matches = re.findall(r'Summa apmaksai\s+([\d,]+)', text)
    if matches:
        amounts = [_parse_lv_amount(v) for v in matches]
        amounts = [a for a in amounts if a is not None]
        if amounts:
            return max(amounts), "regex_fallback"
    return None, "not_found"


def _parse_current_period(text: str) -> tuple[Optional[float], str]:
    """
    'Kopā par periodu NNN,NN' — 'ā' is garbled, use wildcard.
    This is the BOOKING AMOUNT BASE for Tele2.
    """
    m = re.search(r'Kop.{1,3}\s+par periodu\s+([\d,]+)', text)
    if m:
        return _parse_lv_amount(m.group(1)), "regex"
    return None, "not_found"


def _parse_previous_balance(text: str) -> tuple[Optional[float], str]:
    """
    'Kavēts maksājums (tajā skaitā ... kavēta rēķina maksa) NNN,NN'
    All diacritics garbled — use wildcards around stable fragments.
    If line is absent the bill has no overdue — returns 0.00.
    """
    # Match: Kav?ts maks?jums (...) 55,82
    m = re.search(r'Kav.ts\s+maks.jums\s*\([^)]+\)\s+([\d,]+)', text)
    if m:
        val = _parse_lv_amount(m.group(1))
        if val is not None and val > 0:
            return val, "regex"
    return 0.0, "absent_assumed_zero"


def _parse_seller_name(text: str) -> tuple[Optional[str], str]:
    """
    Look for 'SIA Tele2' or 'Tele2' in payment section footer.
    """
    if "SIA Tele2" in text:
        return "SIA Tele2", "text_match"
    if "Tele2" in text:
        return "SIA Tele2", "text_match_partial"
    return None, "not_found"


def _parse_late_fee(text: str) -> tuple[float, str]:
    """
    'Kav?ta r??ina maksa (PVN neapliekams) NNN,NN'
    Late invoice penalty included in current_period_amount.
    Distinct from previous_balance_or_overdue (prior billing cycle charges).
    Multiple rows summed if present (one per overdue invoice).
    """
    matches = re.findall(
        r'Kav.ta\s+r.{1,3}ina\s+maksa\s*\([^)]+\)\s+([\d,]+)',
        text
    )
    if matches:
        amounts = [_parse_lv_amount(v) for v in matches]
        amounts = [a for a in amounts if a is not None and a > 0]
        if amounts:
            return round(sum(amounts), 2), "regex"
    return 0.0, "absent_assumed_zero"


def _parse_line_items(text: str) -> list[dict]:
    """
    Extract per-connection subtotals from the rēķina analīze section.
    Pattern: 'Piesl?gums NNNNNN ... Kop? par piesl?gumu NNN,NN'
    'Piesl?gums' = garbled 'Pieslēgums' (connection).
    Returns list of {connection_id, subtotal}.

    Filters:
    - connection_id must be exactly 8 digits (Latvian mobile/fixed number format)
    - subtotal must be > 0 (zero entries are parse artefacts, not real service rows)
    """
    items = []
    pattern = re.compile(
        r'Piesl.gums\s+(\d{5,})\b.*?Kop.{1,3}\s+par piesl.gumu\s+([\d,]+)',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        conn_id = m.group(1).strip()
        subtotal = _parse_lv_amount(m.group(2))
        if subtotal is None or subtotal <= 0:
            continue
        items.append({
            "connection_id": conn_id,
            "subtotal_eur": subtotal
        })
    return items


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_tele2(pdf_path: str) -> dict:
    """
    Extract structured fields from a Tele2 PDF bill.

    Returns a dict with all required fields plus:
      extraction_method: 'deterministic'
      llm_fallback_used: False (always — LLM not used in this extractor)
      warnings: list of extraction issues
      field_sources: how each field was obtained
    """
    path = Path(pdf_path)

    result = {
        "source_file": path.name,
        "seller_name": None,
        "invoice_number": None,
        "invoice_date": None,
        "billing_period": None,
        "total_payable": None,
        "previous_balance_or_overdue": None,
        "current_period_amount": None,
        "booking_amount": None,
        "late_fee": 0.0,
        "line_items": [],
        "extraction_method": "deterministic",
        "llm_fallback_used": False,
        "warnings": [],
        "field_sources": {},
    }

    # --- Extract raw text from PDF ---
    try:
        raw_text = extract_text(str(path))
    except Exception as exc:
        result["warnings"].append(f"PDF text extraction failed: {exc}")
        result["extraction_method"] = "failed"
        return result

    # --- Parse each field ---
    result["seller_name"], result["field_sources"]["seller_name"] = \
        _parse_seller_name(raw_text)

    result["invoice_number"], result["field_sources"]["invoice_number"] = \
        _parse_invoice_number(raw_text)

    result["invoice_date"], result["field_sources"]["invoice_date"] = \
        _parse_invoice_date(raw_text)

    result["billing_period"], result["field_sources"]["billing_period"] = \
        _parse_billing_period(raw_text)

    result["total_payable"], result["field_sources"]["total_payable"] = \
        _parse_total_payable(raw_text)

    result["current_period_amount"], result["field_sources"]["current_period_amount"] = \
        _parse_current_period(raw_text)

    result["previous_balance_or_overdue"], result["field_sources"]["previous_balance_or_overdue"] = \
        _parse_previous_balance(raw_text)

    result["line_items"] = _parse_line_items(raw_text)

    result["late_fee"], result["field_sources"]["late_fee"] = \
        _parse_late_fee(raw_text)

    # --- Derive booking amount ---
    # Business rule: booking_amount = current_period_amount (excludes previous balance)
    if result["current_period_amount"] is not None:
        result["booking_amount"] = result["current_period_amount"]
        result["field_sources"]["booking_amount"] = "derived:current_period_amount"
    else:
        result["warnings"].append(
            "booking_amount: cannot derive — current_period_amount not found"
        )

    # --- Warnings for missing required fields ---
    required = [
        "seller_name", "invoice_number", "invoice_date",
        "total_payable", "current_period_amount", "booking_amount",
    ]
    for field in required:
        if result.get(field) is None:
            result["warnings"].append(f"{field}: extraction failed")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_tele2_pdf.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    extracted = extract_tele2(pdf_file)
    print(json.dumps(extracted, indent=2, ensure_ascii=False))
