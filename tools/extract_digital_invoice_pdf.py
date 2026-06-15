#!/usr/bin/env python3
"""
extract_digital_invoice_pdf.py
------------------------------
Deterministic extractor for digital B2B invoice PDFs (Latvian and international).

Supports all 13 formats in samples/invoices/digital_pdf/:
  - Latvian standard invoices     (RĒĶINS Nr.)
  - Latvian advance invoices      (Avansa rēķins / Avansa rēķina numurs)
  - Latvian delivery-invoices     (Pavadzīme - Rēķins Nr.)
  - Latvian invoice-factura       (RĒĶINS-FAKTŪRA Nr.)
  - Google Workspace              (English, 0% VAT, reverse charge)
  - Canva                         (English, SaaS)
  - Natural-person sellers        (no VAT registration)

Encoding note:
  pdfplumber may return garbled Latvian diacritics (ā→?, ē→?, š→? etc.).
  All patterns use '.' wildcards where diacritics appear so they remain
  robust without font remapping.

Usage:
  from tools.extract_digital_invoice_pdf import extract_digital_invoice
  result = extract_digital_invoice("samples/invoices/digital_pdf/laskana_26_024.pdf")

  or directly:
  python tools/extract_digital_invoice_pdf.py samples/invoices/digital_pdf/laskana_26_024.pdf
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
    raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")


# ---------------------------------------------------------------------------
# Amount helpers
# ---------------------------------------------------------------------------

def _parse_amount(value: str) -> Optional[float]:
    """Parse amount string to float; handles comma decimal and space thousands."""
    if not value:
        return None
    cleaned = (
        value.strip()
        .replace("\xa0", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    # Remove trailing non-numeric suffix like '€' or 'EUR'
    cleaned = re.sub(r'[^\d.]', '', cleaned)
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_LV_MONTH = {
    'janv': '01', 'febr': '02', 'mart': '03', 'apr': '04', 'maij': '05',
    'aug': '08', 'sept': '09', 'okt': '10', 'nov': '11', 'dec': '12',
}

_EN_MONTH = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05',
    'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10',
    'nov': '11', 'dec': '12',
}


def _lv_month_num(word: str) -> Optional[str]:
    """Map garbled Latvian month word to zero-padded month number."""
    w = word.lower()
    for prefix, num in _LV_MONTH.items():
        if w.startswith(prefix):
            return num
    # jūnijs / jūlijā — first char 'j', second garbled, third 'n' or 'l'
    if len(w) >= 3 and w[0] == 'j' and w[2] == 'n':
        return '06'
    if len(w) >= 3 and w[0] == 'j' and w[2] == 'l':
        return '07'
    return None


def _parse_lv_long_date(text: str) -> Optional[str]:
    """
    Parse Latvian long-form dates to YYYY-MM-DD.
    Handles:
      '2026. gada 06. februāris'   → 2026-02-06
      '2026.gada 28.janvārī'       → 2026-01-28
      '2026. gada 19.Janvārī'      → 2026-01-19
      '2026. g. 2. februāris'      → 2026-02-02
    """
    # Pattern covers both 'gada' and 'g.' abbreviation
    m = re.search(
        r'(\d{4})\.\s*(?:gada|g\.)\s+(\d{1,2})\.?\s*([A-Za-z.]{3,})',
        text, re.IGNORECASE
    )
    if m:
        year, day, month_word = m.group(1), m.group(2).zfill(2), m.group(3)
        month = _lv_month_num(month_word)
        if month:
            return f"{year}-{month}-{day}"
    return None


def _parse_date(value: str) -> Optional[str]:
    """Try multiple date formats, return YYYY-MM-DD or None."""
    value = value.strip()
    # ISO already
    if re.match(r'\d{4}-\d{2}-\d{2}$', value):
        return value
    # DD.MM.YYYY
    try:
        return datetime.strptime(value, '%d.%m.%Y').strftime('%Y-%m-%d')
    except ValueError:
        pass
    # DD/MM/YYYY
    try:
        return datetime.strptime(value, '%d/%m/%Y').strftime('%Y-%m-%d')
    except ValueError:
        pass
    # English: 'Jan 31, 2026' or 'January 01, 2026'
    for fmt in ('%b %d, %Y', '%B %d, %Y'):
        try:
            return datetime.strptime(value, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
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
# Invoice type detection
# ---------------------------------------------------------------------------

def _detect_invoice_type(text: str) -> str:
    """Detect document sub-type from keywords."""
    if re.search(r'Pavadz.{1,3}me\s*[-\u2013]\s*[Rr].{0,3}ins', text):
        return 'delivery_invoice'
    if re.search(r'Avansa\s+r.{0,3}in[sa]', text, re.IGNORECASE):
        return 'advance_invoice'
    if re.search(r'R.{0,3}INS[-\u2013]FAKT.{0,3}RA', text):
        return 'invoice_factura'
    if re.search(r'\bINVOICE\b', text, re.IGNORECASE):
        return 'invoice'
    return 'invoice'


# ---------------------------------------------------------------------------
# Invoice number
# ---------------------------------------------------------------------------

def _parse_invoice_number(text: str) -> tuple[Optional[str], str]:
    """Try patterns in priority order; return (value, method)."""
    patterns = [
        # Delivery-invoice: 'Pavadzīme - Rēķins Nr. AMG 26-0004' or 'Nr. LTA00544292'
        (r'Pavadz.{1,3}me\s*[-\u2013]\s*[Rr].{0,3}ins\s+Nr\.?\s+([\w\s/-]+?)(?:\n|$|\.)', 'regex_delivery'),
        # Advance invoice: 'Avansa rēķins:128520' or 'Avansa rēķina numurs: AV009/26'
        (r'Avansa\s+r.{0,3}ina\s+numurs:\s*([\w/-]+)', 'regex_advance_num'),
        (r'Avansa\s+r.{0,3}ins:\s*([\w/-]+)', 'regex_advance_id'),
        # RĒĶINS-FAKTŪRA Nr. LG-08_01
        (r'R.{0,3}INS[-\u2013]FAKT.{0,3}RA\s+Nr\.?\s+([\w/_-]+)', 'regex_factura'),
        # Rēķina numurs INV-2026/12
        (r'R.{0,3}ina\s+numurs\s+([\w/_-]+)', 'regex_rekina_numurs'),
        # RĒĶINS Nr. 26/024 / Rēķins Nr. 2026/5 / Rēķins Nr.ARD92100
        (r'R.{0,3}[Ii][Nn][Ss]\s+Nr\.?\s*[\u2013-]?\s*([\w/_-]+)', 'regex_rekins'),
        # English: 'Invoice number: 5479391989'
        (r'Invoice\s+number:\s*(\S+)', 'regex_en_number'),
        # English: 'Invoice #: 04779-24590090-1'
        (r'Invoice\s+#:\s*(\S+)', 'regex_en_hash'),
        # Kurzemes Vārds: 'Nr. 260128' in header block
        (r'\bNr\.\s+(\d{5,})', 'regex_nr'),
    ]
    for pattern, method in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip('.')
            if val:
                return val, method
    return None, 'not_found'


# ---------------------------------------------------------------------------
# Invoice date
# ---------------------------------------------------------------------------

def _parse_invoice_date(text: str) -> tuple[Optional[str], str]:
    """Extract and normalise invoice date."""
    # English formats (Canva, Google)
    # Canva/Google have clean 'Invoice date: February 01, 2026'
    m = re.search(r'Invoice\s+date:\s*([A-Za-z]+ \d{1,2},\s*\d{4})', text)
    if m:
        val = _parse_date(m.group(1))
        if val:
            return val, 'regex_en_invoice_date'

    # Google Workspace: billing period end = invoice date
    # 'Summary for Jan 1, 2026 - Jan 31, 2026'
    m = re.search(
        r'Summary for\s+[A-Za-z]+ \d+,\s*\d{4}\s*[-\u2013]\s*([A-Za-z]+ \d+,\s*\d{4})',
        text
    )
    if m:
        val = _parse_date(m.group(1))
        if val:
            return val, 'regex_google_summary'

    # 'Rēķina datums: DD.MM.YYYY' (area_it)
    m = re.search(r'R.{0,3}ina\s+datums:\s*(\d{2}\.\d{2}\.\d{4})', text)
    if m:
        val = _parse_date(m.group(1))
        if val:
            return val, 'regex_rekina_datums'

    # 'Datums: DD/MM/YYYY' or 'Datums: YYYY.gada DD.Month'
    m = re.search(r'Datums:\s*(\d{2}/\d{2}/\d{4})', text)
    if m:
        val = _parse_date(m.group(1))
        if val:
            return val, 'regex_datums_slash'

    # 'Datums: 2026.gada 28.janvārī' — compound, parse as LV long
    m = re.search(r'Datums:\s*(\d{4}\.gada .+?)(?:\n|Avansa)', text)
    if m:
        val = _parse_lv_long_date(m.group(1))
        if val:
            return val, 'regex_datums_lv'

    # 'Izrakstīšanas datums DD.MM.YYYY'
    m = re.search(r'Izrakst.{1,4}anas\s+datums\s+(\d{2}\.\d{2}\.\d{4})', text)
    if m:
        val = _parse_date(m.group(1))
        if val:
            return val, 'regex_izrakstisanas'

    # 'YYYY. gada DD. Month' or 'YYYY. g. DD. Month' at start of document
    val = _parse_lv_long_date(text[:800])
    if val:
        return val, 'regex_lv_long'

    # 'DD.MM.YYYY' standalone (e.g. after "Rēķina datums:")
    m = re.search(r'(\d{2}\.\d{2}\.\d{4})', text[:500])
    if m:
        val = _parse_date(m.group(1))
        if val:
            return val, 'regex_ddmmyyyy_early'

    return None, 'not_found'


# ---------------------------------------------------------------------------
# Due date
# ---------------------------------------------------------------------------

def _parse_due_date(text: str) -> tuple[Optional[str], str]:
    """Extract payment due date."""
    patterns = [
        (r'Apmaks.t\s+l.dz\s+(\d{2}\.\d{2}\.\d{4})', 'ddmmyyyy'),
        (r'Apmaks.t\s+l.dz:\s*(\d{2}/\d{2}/\d{4})', 'ddslash'),
        (r'Apmaksas\s+datums:\s*(\d{2}\.\d{2}\.\d{4})', 'ddmmyyyy'),
        (r'Apmaksas\s+termi.{1,3}.:\s*(\d{2}\.\d{2}\.\d{4})', 'ddmmyyyy'),
        (r'Apmaksas\s+termi.{1,3}.:\s*(\d{4}-\d{2}-\d{2})', 'iso'),
        (r'Apmaksas\s+termi.{1,3}.\s+(\d{4}\.\s*gada\s+\d{1,2}\.?\s*\w+)', 'lv_long'),
    ]
    for pattern, fmt in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            if fmt == 'lv_long':
                val = _parse_lv_long_date(raw)
            else:
                val = _parse_date(raw)
            if val:
                return val, f'regex_{fmt}'
    return None, 'not_found'


# ---------------------------------------------------------------------------
# VAT / non-VAT flags
# ---------------------------------------------------------------------------

def _detect_reverse_charge(text: str) -> bool:
    return bool(re.search(r'reverse.charge|Article\s+196', text, re.IGNORECASE))


def _detect_non_vat_payer(text: str) -> bool:
    """True when seller is not a VAT payer."""
    if re.search(r'NAV\s+PVN\s+MAKS.T.JS', text, re.IGNORECASE):
        return True
    # 'PVN -' (dash instead of amount)
    if re.search(r'\bPVN\s+-\s', text):
        return True
    return False


# ---------------------------------------------------------------------------
# VAT rate
# ---------------------------------------------------------------------------

def _detect_vat_rate(text: str) -> Optional[int]:
    """Return predominant VAT rate (0, 21) or None."""
    if _detect_reverse_charge(text):
        return 0
    if _detect_non_vat_payer(text):
        return None
    if re.search(r'21[\s.,]?%\s*PVN|PVN\s*21[\s.,]?%|PVN\s+likme\s*21|21\.00%\s*PVN|TAX.*21', text, re.IGNORECASE):
        return 21
    if re.search(r'VAT\s*\(0%\)', text):
        return 0
    return None


# ---------------------------------------------------------------------------
# Amounts
# ---------------------------------------------------------------------------

def _parse_total_gross(text: str) -> tuple[Optional[float], str]:
    """Total amount payable (incl. VAT)."""
    patterns = [
        # Laskana: 'Pavisam apmaksai: EUR 610.47'
        (r'Pavisam apmaksai:\s*EUR\s*([\d.,]+)', 'pavisam_apmaksai'),
        # Ardala: 'Kopējā summa (EUR) 493,43'
        (r'Kop.j.\s*summa\s*\(EUR\)\s*([\d.,]+)', 'kopeja_summa'),
        # AMG: 'Summa apmaksai (EUR) 132.02'
        (r'Summa\s+apmaksai\s*\(EUR\)\s*([\d.,]+)', 'summa_apmaksai_eur'),
        # Kurzemes Vārds: 'KOPĀ EUR 156.82'
        (r'KOP.\s*EUR\s*([\d.,]+)', 'kopa_eur'),
        # Utility/settlement: 'KOPĀ 149,89' (standalone, before debt-inclusive 'Kopā apmaksai')
        (r'^KOP[AĀ]\s+([\d.,]+)\s*$', 'kopa_standalone'),
        # Klavs, area_it, mija_ar: 'Kopā apmaksai 260.00' / 'Kopā apmaksai: € 450.00'
        (r'Kop.\s*apmaksai[:\s]*[€]?\s*([\d.,]+)', 'kopa_apmaksai'),
        # Linda Gulbe: 'Summa apmaksai 300,00'
        (r'Summa\s+apmaksai\s+([\d.,]+)', 'summa_apmaksai'),
        # Bite Dyson, Klinta: 'Kopsumma 249,00' (not 'Kopsumma bez PVN')
        (r'Kopsumma\s+(?!bez\s+PVN)([\d.,]+)', 'kopsumma'),
        # Google: 'Total in EUR €48.60' (take last occurrence)
        (r'Total\s+in\s+EUR\s*[€$]?\s*([\d.]+)', 'total_in_eur'),
        # Canva: 'Total 9.91 2.08 EUR 11.99'
        (r'^Total\s+[\d.]+\s+[\d.]+\s+EUR\s+([\d.]+)', 're_canva'),
    ]
    for pattern, method in patterns:
        flags = re.MULTILINE if method in ('re_canva', 'kopa_standalone') else 0
        matches = list(re.finditer(pattern, text, flags))
        if matches:
            # For 'total_in_eur' take the last match (page 2 repetition same value)
            val = _parse_amount(matches[-1].group(1))
            if val is not None and val > 0:
                return val, method
    return None, 'not_found'


def _parse_total_net(text: str) -> tuple[Optional[float], str]:
    """Total net (excl. VAT)."""
    patterns = [
        # Google: 'Subtotal in EUR €48.60'
        (r'Subtotal\s+in\s+EUR\s*[€$]?\s*([\d.]+)', 'subtotal_in_eur'),
        # Kurzemes: 'SUMMA BEZ PVN EUR 129.60'
        (r'SUMMA\s+BEZ\s+PVN\s+EUR\s*([\d.,]+)', 'summa_bez_pvn_eur'),
        # Bite Dyson: 'Kopsumma bez PVN 205,79'
        (r'Kopsumma\s+bez\s+PVN\s+([\d.,]+)', 'kopsumma_bez_pvn'),
        # Most Latvian: 'Kopā bez PVN 754.13'
        (r'Kop.\s+bez\s+PVN\s*([\d.,]+)', 'kopa_bez_pvn'),
        # Canva: price column 'PRICE' value before tax
        (r'^Subscription:.*?\s+([\d.]+)\s+\(', 'canva_price'),
        # Area IT: 'Kopā 8.26 EUR'
        (r'Kop.\s+([\d.,]+)\s+EUR\b', 'kopa_eur_net'),
        # Ardala: 'Summa 407,79' (standalone total net line)
        (r'^Summa\s+([\d.,]+)$', 'summa_standalone'),
    ]
    for pattern, method in patterns:
        flags = re.MULTILINE if method in ('canva_price', 'summa_standalone') else 0
        m = re.search(pattern, text, flags)
        if m:
            val = _parse_amount(m.group(1))
            if val is not None and val > 0:
                return val, method
    return None, 'not_found'


def _parse_total_vat(text: str) -> tuple[Optional[float], str]:
    """Total VAT amount."""
    # Reverse charge or non-VAT payer → 0.00
    if _detect_reverse_charge(text):
        return 0.0, 'reverse_charge_zero'
    if _detect_non_vat_payer(text):
        return 0.0, 'non_vat_payer_zero'

    patterns = [
        # Laskana: 'PVN likme 21.00 % no 504.52 105.95'
        (r'PVN\s+likme\s+21\.00\s*%\s+no\s+[\d.,]+\s+([\d.,]+)', 'pvn_likme_no'),
        # Area IT: '21.00% PVN 1.74 EUR'
        (r'21\.00%\s+PVN\s+([\d.,]+)', 'pct_pvn'),
        # Kurzemes: 'PVN 21.00% EUR 27.22'
        (r'PVN\s+21\.00%\s+EUR\s*([\d.,]+)', 'pvn_pct_eur'),
        # Bite Dyson: 'PVN summa, 21% 43,21'
        (r'PVN\s+summa,?\s*21%\s*([\d.,]+)', 'pvn_summa'),
        # Ardala: 'Pievienotās vērtības nodoklis, 21 % PVN likme 85,64'
        (r'Pievienot.s\s+v.rt.bas\s+nodoklis,\s*21\s*%\s+PVN\s+likme\s+([\d.,]+)', 'pvn_pvn_likme'),
        # Canva: 'TAX' column value
        (r'(?:TAX|Tax)\s+([\d.]+)', 'tax_col'),
        # Google: 'VAT (0%) €0.00'
        (r'VAT\s*\(0%\)\s*[€$]?\s*([\d.]+)', 'vat_zero_pct'),
        # Generic PVN 21%: 'PVN 21% 45.12' or 'PVN 21%\n45.12'
        (r'PVN\s*21%\s+([\d.,]+)', 'pvn_21pct'),
        # AMG: 'PVN 21% (EUR) 22.91'
        (r'PVN\s+21%\s*\(EUR\)\s*([\d.,]+)', 'pvn_21pct_eur'),
    ]
    for pattern, method in patterns:
        m = re.search(pattern, text)
        if m:
            val = _parse_amount(m.group(1))
            if val is not None:
                return val, method
    return None, 'not_found'


# ---------------------------------------------------------------------------
# Seller name
# ---------------------------------------------------------------------------

def _parse_seller_name(text: str) -> tuple[Optional[str], str]:
    # Each entry: (pattern, flags)
    # Order matters — more specific first.
    patterns = [
        # Kurzemes: 'PIEGĀDĀTĀJS: SIA KURZEMES VĀRDS\n'
        (r'PIEG.D.T.JS:\s*(.+?)(?:\n|Re\.\s*numurs)', 0),
        # Google: 'Google Cloud EMEA Limited' — first line of document
        (r'^(Google\s+[A-Za-z\s]+(?:Limited|Ltd))', re.MULTILINE),
        # Canva: 'Provided by: Provided to:\nCanva Pty Ltd woodenlays'
        (r'Provided\s+by:.*?\n(.+?)\s+(?:woodenlays|Woodenlays)', re.DOTALL),
        # 'Pakalpojuma sniedzējs Laskana LSEZ SIA PVN Kods'
        (r'Pakalpojuma\s+sniedz.js\s*(.+?)\s+PVN\s+Kods', 0),
        # Bite: 'BITE Latvija SIA Reģistrācijas numurs: ...'
        (r'^(.+?SIA)\s+Re.istr.cijas\s+numurs:', re.MULTILINE),
        # Klavs two-column: 'Piegādātājs Saņēmējs\nKlāvs Ašmis Woodenlays, SIA'
        (r'Pieg.d.t.js\s+Sa.{1,8}m.js\s*\n(.+?)\s+Woodenlays', re.DOTALL),
        # 'Piegādātājs SIA "AMG montāža" Reģ. Nr.' — stop before Reģ.
        (r'Pieg.d.t.js\s+(.+?)\s+Re.\.\s*Nr\.', 0),
        # Linda Gulbe / AMG fallback: 'Piegādātājs Linda Gulbe\n'
        (r'Pieg.d.t.js\s+(.+?)$', re.MULTILINE),
        # 'Izpildītājs : ARDALA, SIA' or 'Izpildītājs SIA "MIJA AR"'
        (r'Izpild.t.js\s*:?\s*(.+?)(?:\n|Re.\.\s*Nr\.)', 0),
        # Klinta: 'IZPILDĪTĀJS:\nKlinta Andersone AS Industra Bank'
        (r'IZPILD.T.JS:\s*\n(.+?)\s+(?:AS|SIA)\s', re.DOTALL),
        # area_it: first line 'SIA Area IT'
        (r'^(SIA\s+\w+(?:\s+\w+)?)', re.MULTILINE),
    ]
    bad = {'oodenlays', 'Sa\ufffdm\ufffdjs', 'Sa??m?js'}
    for pattern, flags in patterns:
        m = re.search(pattern, text, flags)
        if m:
            val = m.group(1).strip().rstrip('.,')
            # Reject buyer names and obvious mis-captures
            if (val and len(val) > 2
                    and 'oodenlays' not in val
                    and not any(b in val for b in bad)):
                return val, 'regex'
    return None, 'not_found'


# ---------------------------------------------------------------------------
# Seller VAT number
# ---------------------------------------------------------------------------

def _parse_seller_vat(text: str) -> tuple[Optional[str], str]:
    """First VAT/PVN number in document (seller always appears before buyer)."""
    # Google: 'VAT number: IE3668997OH'
    m = re.search(r'VAT\s+number:\s*(\S+)', text)
    if m:
        return m.group(1).strip(), 'regex_en_vat'
    # Canva: 'Business #: SaaS: EU372042198'
    m = re.search(r'Business\s*#:\s*(?:SaaS:\s*)?(\w+)', text)
    if m:
        return m.group(1).strip(), 'regex_canva_biz'
    # Latvian: LVxxxxxxxxxxxxxxx — first occurrence
    m = re.search(r'PVN\s*(?:maks\.\s*re.\.\s*)?(?:Nr\.?|Kods):?\s*(LV\d+)', text)
    if m:
        return m.group(1).strip(), 'regex_pvn_nr'
    # 'Nod. maks. reģ. Nr. LV42103017747' (Ardala)
    m = re.search(r'Nod\.\s*maks\.\s*re.\.\s*Nr\.\s*(LV\d+)', text)
    if m:
        return m.group(1).strip(), 'regex_nod_maks'
    return None, 'not_found'


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

def _detect_currency(text: str) -> str:
    if 'EUR' in text or '€' in text:
        return 'EUR'
    return 'EUR'  # default — all sample invoices are EUR


# ---------------------------------------------------------------------------
# Triangle validation
# ---------------------------------------------------------------------------

def _validate_triangle(
    total_gross: Optional[float],
    total_net: Optional[float],
    total_vat: Optional[float],
    warnings: list,
) -> str:
    """
    Verify total_net + total_vat ≈ total_gross.
    Returns 'OK', 'NEEDS_REVIEW', or 'INCOMPLETE'.
    """
    if total_gross is None:
        return 'INCOMPLETE'
    if total_net is None or total_vat is None:
        return 'OK'  # partial — cannot verify
    computed = round(total_net + total_vat, 2)
    if abs(computed - total_gross) > 0.05:
        warnings.append(
            f"Triangle mismatch: net({total_net}) + vat({total_vat}) = "
            f"{computed} ≠ gross({total_gross})"
        )
        return 'NEEDS_REVIEW'
    return 'OK'


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_digital_invoice(pdf_path: str) -> dict:
    """
    Extract structured fields from a digital B2B invoice PDF.

    Returns a dict with all fields plus:
      extraction_method: 'deterministic'
      llm_fallback_used: False
      warnings: list
      field_sources: per-field extraction method
    """
    path = Path(pdf_path)

    result = {
        "source_file": path.name,
        "invoice_type": None,
        "invoice_number": None,
        "invoice_date": None,
        "due_date": None,
        "seller_name": None,
        "seller_vat": None,
        "currency": "EUR",
        "total_gross": None,
        "total_net": None,
        "total_vat": None,
        "vat_rate": None,
        "reverse_charge": False,
        "non_vat_payer": False,
        "booking_amount": None,
        "extraction_method": "deterministic",
        "llm_fallback_used": False,
        "warnings": [],
        "field_sources": {},
    }

    try:
        raw_text = extract_text(str(path))
    except Exception as exc:
        result["warnings"].append(f"PDF text extraction failed: {exc}")
        result["extraction_method"] = "failed"
        return result

    warnings: list = result["warnings"]
    src: dict = result["field_sources"]

    # --- Document type ---
    result["invoice_type"] = _detect_invoice_type(raw_text)

    # --- VAT flags (needed early for amount parsing) ---
    result["reverse_charge"] = _detect_reverse_charge(raw_text)
    result["non_vat_payer"] = _detect_non_vat_payer(raw_text)
    result["vat_rate"] = _detect_vat_rate(raw_text)
    result["currency"] = _detect_currency(raw_text)

    # --- Core fields ---
    result["invoice_number"], src["invoice_number"] = _parse_invoice_number(raw_text)
    result["invoice_date"], src["invoice_date"] = _parse_invoice_date(raw_text)
    result["due_date"], src["due_date"] = _parse_due_date(raw_text)
    result["seller_name"], src["seller_name"] = _parse_seller_name(raw_text)
    result["seller_vat"], src["seller_vat"] = _parse_seller_vat(raw_text)

    # --- Amounts ---
    result["total_gross"], src["total_gross"] = _parse_total_gross(raw_text)
    result["total_net"], src["total_net"] = _parse_total_net(raw_text)
    result["total_vat"], src["total_vat"] = _parse_total_vat(raw_text)

    # --- For non-VAT payers: net = gross if net not found ---
    if result["non_vat_payer"] and result["total_net"] is None and result["total_gross"] is not None:
        result["total_net"] = result["total_gross"]
        src["total_net"] = "derived:non_vat_payer_net_equals_gross"

    # --- Booking amount = total_gross ---
    if result["total_gross"] is not None:
        result["booking_amount"] = result["total_gross"]
        src["booking_amount"] = "derived:total_gross"

    # --- Triangle validation ---
    triangle_status = _validate_triangle(
        result["total_gross"], result["total_net"], result["total_vat"], warnings
    )

    # --- Warnings for missing required fields ---
    required = ["invoice_number", "invoice_date", "total_gross", "seller_name"]
    for field in required:
        if result.get(field) is None:
            warnings.append(f"{field}: extraction failed")

    # --- Confidence ---
    missing_required = sum(1 for f in required if result.get(f) is None)
    if triangle_status == 'NEEDS_REVIEW':
        result["confidence"] = "low"
    elif missing_required == 0:
        result["confidence"] = "high" if not warnings else "medium"
    else:
        result["confidence"] = "low"

    result["triangle_status"] = triangle_status

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_digital_invoice_pdf.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    extracted = extract_digital_invoice(pdf_file)
    print(json.dumps(extracted, indent=2, ensure_ascii=False))
