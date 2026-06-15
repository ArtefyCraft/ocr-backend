#!/usr/bin/env python3
"""
extract_bite_pdf.py
-------------------
Deterministic extractor for SIA Bite Latvija monthly business bill PDFs.

Strategy:
  1. Primary: regex on visible text layer (Bite does NOT embed DFC_ metadata)
  2. LLM fallback: NOT used — all required fields have deterministic patterns.
     If a field cannot be extracted it is returned as None with a warning.

Encoding note:
  pdfplumber extracts Bite PDFs with garbled Latvian diacritics
  (ā→\ufffd, ē→\ufffd, š→\ufffd etc.). Patterns use '.' wildcards where
  diacritics appear.  Bite amounts use English decimal '.' (not Latvian ',').

Key structural labels (diacritics stripped):
  KOP? APMAKSAI: NNN.NN EUR           → total payable (uppercase, page 1)
  R??ina Nr. NNNNNNNNNN                → invoice number
  R??ina izrakst??anas datums DD.MM.YYYY  → invoice date
  Kop? NNN.NN \\nApmaksas summa …      → current period amount (page 2 summary)
  Apmaksas summa par iepriek?jo periodu → previous balance
  Kav?ts maks?jums: … NNN.NN           → overdue (confirms previous balance)
  P?RSKATS PAR MOBILO SAKARU …, tel. nr. 371XXXXXXXX … Kopsumma: NNN.NN
                                       → per-number subtotals (pages 3-7)

Usage:
  from tools.extract_bite_pdf import extract_bite
  result = extract_bite("samples/telecom/bite_pdf/5.pdf")

  or directly:
  python tools/extract_bite_pdf.py samples/telecom/bite_pdf/5.pdf
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
# Amount / date parsing
# ---------------------------------------------------------------------------

def _parse_amount(value: str) -> Optional[float]:
    """
    Parse a number string to float.
    Handles both Latvian comma (213,90) and English period (213.90) decimal.
    Also strips non-breaking spaces used as thousand separators.
    """
    if not value:
        return None
    cleaned = (
        value.strip()
        .replace("\xa0", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalise_date(value: str) -> Optional[str]:
    """Convert DD.MM.YYYY → YYYY-MM-DD (ISO 8601). Strips trailing period if present."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip().rstrip("."), "%d.%m.%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Extract full text from all pages of a PDF, joined with newlines."""
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

def _parse_seller_name(text: str) -> tuple[Optional[str], str]:
    """
    Look for 'BITE Latvija' in text — normalise to 'SIA Bite Latvija'.
    """
    if "BITE Latvija" in text:
        return "SIA Bite Latvija", "text_match"
    if "BITE" in text:
        return "SIA Bite Latvija", "text_match_partial"
    return None, "not_found"


def _parse_invoice_number(text: str) -> tuple[Optional[str], str]:
    """
    'R??ina Nr. NNNNNNNNNN' (page 1) or 'R??ins Nr. NNNNNNNNNN' (page 2+).
    Diacritics in 'Rēķina'/'Rēķins' garbled — use wildcards.
    """
    # Match both genitive (R??ina) and nominative (R??ins) forms
    m = re.search(r'R.{1,6}in[as]\s+Nr\.\s+(\d+)', text)
    if m:
        return m.group(1).strip(), "regex"
    return None, "not_found"


def _parse_invoice_date(text: str) -> tuple[Optional[str], str]:
    """
    'R??ina izrakst??anas datums DD.MM.YYYY.' — date ends with a period.
    Normalised to YYYY-MM-DD (ISO 8601).
    """
    m = re.search(
        r'R.{1,6}ina izrakst.{1,14}anas datums\s+(\d{2}\.\d{2}\.\d{4})\.?',
        text
    )
    if m:
        val = _normalise_date(m.group(1))
        if val:
            return val, "regex"
    return None, "not_found"


def _parse_billing_period(text: str) -> tuple[Optional[str], str]:
    """
    'DD.MM.YYYY. - DD.MM.YYYY.' — trailing periods on each date.
    Appears in per-number breakdown pages (pages 3+).
    Returns 'DD.MM.YYYY - DD.MM.YYYY' (trailing periods removed).
    """
    m = re.search(
        r'(\d{2}\.\d{2}\.\d{4})\.\s*[-\u2013]\s*(\d{2}\.\d{2}\.\d{4})\.',
        text
    )
    if m:
        return f"{m.group(1)} - {m.group(2)}", "regex"
    return None, "not_found"


def _parse_total_net(text: str) -> tuple[Optional[float], str]:
    """First 'Ar PVN apliekamā summa: NNN.NN' line — page 1 KOPSAVILKUMS section."""
    m = re.search(r'Ar\s+PVN\s+apliekam.\s+summa:\s+([\d.]+)', text)
    if m:
        val = _parse_amount(m.group(1))
        if val is not None:
            return val, "regex"
    return None, "not_found"


def _parse_total_vat(text: str) -> tuple[Optional[float], str]:
    """First 'PVN NN% NNN.NN' line — page 1 KOPSAVILKUMS section."""
    m = re.search(r'PVN\s+\d+%\s+([\d.]+)', text)
    if m:
        val = _parse_amount(m.group(1))
        if val is not None:
            return val, "regex"
    return None, "not_found"


def _parse_current_period_gross(text: str) -> tuple[Optional[float], str]:
    """
    Current period gross — total before adding prior balance.
    Distinct from KOPĀ APMAKSAI which may include prior debt.

    Strategy:
    1. 'Kopā par periodu ... NNN.NN' (b_i_t_e.pdf style — last decimal on line)
    2. 'Kopā NNN.NN' between KOPSAVILKUMS and 'Apmaksas summa par iepriek'
    """
    # Pattern 1: 'Kopā par periodu ... NNN.NN' — anchored to end of line
    m = re.search(r'^Kop.\s+par\s+periodu[^\n]*?([\d]+\.[\d]+)\s*$', text, re.MULTILINE)
    if m:
        val = _parse_amount(m.group(1))
        if val is not None:
            return val, "regex_kopa_par_periodu"

    # Pattern 2: 'Kopā NNN.NN' inside KOPSAVILKUMS section (before previous balance line)
    kopsav = text.find('KOPSAVILKUMS')
    if kopsav >= 0:
        rest = text[kopsav:]
        end_marker = re.search(r'Apmaksas\s+summa\s+par\s+iepriek', rest)
        end = end_marker.start() if end_marker else len(rest)
        section = rest[:end]
        m2 = re.search(r'^Kop.\s+([\d]+\.[\d]+)\s*$', section, re.MULTILINE)
        if m2:
            val = _parse_amount(m2.group(1))
            if val is not None:
                return val, "regex_kopa_summary"
    return None, "not_found"


def _parse_total_payable(text: str) -> tuple[Optional[float], str]:
    """
    'KOP? APMAKSAI: NNN.NN EUR' (uppercase, page 1 and page 2 summary).
    Diacritics in 'KOPĀ' garbled. Takes the maximum if found multiple times
    (both occurrences should be equal; max guards against partial reads).
    """
    matches = re.findall(r'KOP.{1,3}\s+APMAKSAI:\s+([\d.]+)', text)
    if matches:
        amounts = [_parse_amount(v) for v in matches]
        amounts = [a for a in amounts if a is not None]
        if amounts:
            return max(amounts), "regex"
    return None, "not_found"


def _parse_current_period(text: str) -> tuple[Optional[float], str]:
    """
    Page 2 summary structure:
      Kop? NNN.NN
      Apmaksas summa par iepriek?jo periodu NNN.NN

    The 'Kop?' line immediately before 'Apmaksas summa' is the current-period
    total (excluding previous balance).  This is the BOOKING AMOUNT BASE.

    Fallback: if the above anchor is absent (no previous balance on the bill),
    the current period equals total payable.
    """
    # Primary: anchored to next line being 'Apmaksas summa'
    m = re.search(
        r'Kop.{1,3}\s+([\d.]+)\s*\nApmaksas summa',
        text
    )
    if m:
        val = _parse_amount(m.group(1))
        if val is not None and val > 0:
            return val, "regex"

    # Fallback: no previous balance section found → current = total
    total, _ = _parse_total_payable(text)
    if total is not None:
        return total, "derived:total_payable_no_previous_balance"

    return None, "not_found"


def _parse_previous_balance(text: str) -> tuple[Optional[float], str]:
    """
    'Apmaksas summa par iepriek?jo periodu NNN.NN' — previous period amount.
    If line absent, returns 0.00 (bill has no overdue balance).

    Secondary check: 'Kav?ts maks?jums: … NNN.NN' confirms the overdue amount.

    Settled-balance check: if a 'Samaks?ts l?dz DD.MM.YYYY NNN.NN' line within
    the next 3 lines shows a paid amount equal (±0.02) to the previous period
    amount, the balance is fully settled — return 0.0. This handles Bite bills
    that print the previous-period line for reference after the customer has
    already paid it.
    """
    # Primary: explicit previous period line
    m = re.search(
        r'Apmaksas summa par iepriek.{1,6}jo periodu\s+([\d.]+)',
        text
    )
    if m:
        val = _parse_amount(m.group(1))
        if val is not None and val > 0:
            # Settled-balance check: look for 'Samaks?ts l?dz ... AMOUNT' within
            # the next 3 lines. If paid amount matches val (±0.02), balance is settled.
            tail = text[m.end():]
            next_lines = tail.split('\n')[:4]  # current line tail + next 3 lines
            paid_m = re.search(
                r'Samaks.ts\s+l.dz[^\n]*?(\d+\.\d{2})(?!\d)(?=\s|$)',
                '\n'.join(next_lines),
                re.MULTILINE,
            )
            if paid_m:
                paid_val = _parse_amount(paid_m.group(1))
                if paid_val is not None and abs(paid_val - val) <= 0.02:
                    return 0.0, "settled_by_samaksats"
            return val, "regex"

    # Secondary: overdue notice line (last number on the line)
    m = re.search(r'Kav.ts\s+maks.jums:.*?([\d.]+)\s*$', text, re.MULTILINE)
    if m:
        val = _parse_amount(m.group(1))
        if val is not None and val > 0:
            return val, "regex_fallback"

    return 0.0, "absent_assumed_zero"


def _parse_line_items(text: str) -> list[dict]:
    """
    Extract per-number subtotals from phone detail pages (pages 3+).

    Pattern: 'P?RSKATS PAR MOBILO SAKARU PAKALPOJUMIEM, tel. nr. 371XXXXXXXX'
    followed (within the same page block) by 'Kopsumma: NNN.NN'.

    Filters:
    - phone number must be exactly 11 digits starting with 371 (Latvian country code)
    - subtotal must be > 0 (zero entries are parse artefacts)

    Note: per-number Kopsumma values include VAT but exclude any late-payment
    penalty (Līgumsods) which is billed separately at the invoice level.
    Their sum may therefore differ from current_period_amount by the penalty amount.
    """
    items = []
    pattern = re.compile(
        r'P.RSKATS PAR MOBILO SAKARU PAKALPOJUMIEM,\s*tel\.\s*nr\.\s*(371\d{8})'
        r'.*?Kopsumma:\s+([\d.]+)',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        phone = m.group(1).strip()
        subtotal = _parse_amount(m.group(2))
        if subtotal is None or subtotal <= 0:
            continue
        items.append({
            "phone_number": phone,
            "subtotal_eur": subtotal,
        })
    return items


def _parse_late_fee(text: str) -> tuple[float, str]:
    """
    'L?gumsods par r??ina samaksas termi?a nokav?jumu (ar PVN neapliek) NNN.NN'
    Contractual late-payment penalty included in current_period_amount.
    Distinct from previous_balance_or_overdue (prior billing cycle charges).
    """
    m = re.search(
        r'L.gumsods\s+par\s+r.{1,5}ina\s+samaksas\s+termi.{1,4}a\s+nokav.jumu.*?([\d]+\.[\d]+)',
        text
    )
    if m:
        val = _parse_amount(m.group(1))
        if val is not None and val > 0:
            return val, "regex"
    return 0.0, "absent_assumed_zero"


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_bite(pdf_path: str) -> dict:
    """
    Extract structured fields from a Bite Latvija PDF bill.

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
        "total_net": None,
        "total_vat": None,
        "total_gross": None,
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

    result["total_net"], result["field_sources"]["total_net"] = \
        _parse_total_net(raw_text)

    result["total_vat"], result["field_sources"]["total_vat"] = \
        _parse_total_vat(raw_text)

    result["total_gross"], result["field_sources"]["total_gross"] = \
        _parse_current_period_gross(raw_text)

    result["total_payable"], result["field_sources"]["total_payable"] = \
        _parse_total_payable(raw_text)

    result["previous_balance_or_overdue"], result["field_sources"]["previous_balance_or_overdue"] = \
        _parse_previous_balance(raw_text)

    result["current_period_amount"], result["field_sources"]["current_period_amount"] = \
        _parse_current_period(raw_text)

    result["line_items"] = _parse_line_items(raw_text)

    result["late_fee"], result["field_sources"]["late_fee"] = \
        _parse_late_fee(raw_text)

    # --- Derive booking amount ---
    # Business rule: booking_amount = current period charges only.
    # Priority order:
    #   1. total_gross (new — from "Kopā par periodu" / KOPSAVILKUMS Kopā line)
    #   2. current_period_amount (legacy "Kopā NN.NN \n Apmaksas summa" anchor)
    #   3. total_payable - previous_balance (when previous balance is known)
    # In all cases, booking_amount must NOT exceed total_gross when total_gross is known,
    # and must NOT include any prior balance.
    booking = None
    booking_source = None
    tg = result["total_gross"]
    cpa = result["current_period_amount"]
    tp = result["total_payable"]
    pb = result["previous_balance_or_overdue"] or 0.0

    if tg is not None and tg > 0:
        booking = tg
        booking_source = "derived:total_gross"
    elif cpa is not None and cpa > 0:
        booking = cpa
        booking_source = "derived:current_period_amount"
    elif tp is not None and tp > 0 and pb > 0:
        booking = round(tp - pb, 2)
        booking_source = "derived:total_payable_minus_previous_balance"
    elif tp is not None and tp > 0:
        booking = tp
        booking_source = "derived:total_payable_no_previous_balance"

    # Safety: never let booking exceed total_gross when known
    if booking is not None and tg is not None and booking > tg + 0.05:
        booking = tg
        booking_source = "capped:total_gross"

    if booking is not None:
        result["booking_amount"] = booking
        result["field_sources"]["booking_amount"] = booking_source
    else:
        result["warnings"].append(
            "booking_amount: cannot derive — no current period or total found"
        )

    # --- Consistency cross-check ---
    # If total, current, and previous are all present, verify triangle internally.
    # Mismatch here is a warning (not a failure — the validator does the authoritative check).
    tp = result["total_payable"]
    cp = result["current_period_amount"]
    pb = result["previous_balance_or_overdue"] or 0.0
    if tp is not None and cp is not None:
        diff = abs(tp - (cp + pb))
        if diff > 0.05:
            result["warnings"].append(
                f"Internal triangle mismatch: total={tp:.2f}, "
                f"current={cp:.2f} + previous={pb:.2f} = {cp + pb:.2f} "
                f"[diff={diff:.2f}]"
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
        print("Usage: python extract_bite_pdf.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    extracted = extract_bite(pdf_file)
    print(json.dumps(extracted, indent=2, ensure_ascii=False))
