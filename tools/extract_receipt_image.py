#!/usr/bin/env python3
"""
extract_receipt_image.py
-------------------------
OCR-based extractor for receipt images (JPEG, PNG, etc.).

Handles Latvian retail receipts from stores like:
  - Tirdzniecibas nams Kursi SIA (building materials)
  - SIA DEPO DIY (large DIY/hardware chain)
  - Fuel station receipts (explicit VAT breakdown)
  - Other POS receipts

Strategy:
  1. Load image → Tesseract OCR (lav+eng, psm 6)
  2. If text yield is poor, retry with psm 4 + preprocessing
  3. Clean OCR artefacts (replacement chars, stray tokens)
  4. Parse: seller_name, receipt_number, receipt_date, total_gross,
            total_net, total_vat, currency, payment_method, discounts
  5. Triangle validation: net + vat ≈ gross; mismatch → NEEDS_REVIEW
  6. Confidence scoring based on OCR noise + missing fields

Tesseract setup:
  Binary  : C:/Program Files/Tesseract-OCR/tesseract.exe
  Tessdata: ./tessdata/   (lav.traineddata + eng.traineddata)
  Language: lav+eng

Usage:
  from tools.extract_receipt_image import extract_receipt
  result = extract_receipt("samples/receipts/image_good/kursi_01.jpeg")

  or directly:
  python tools/extract_receipt_image.py samples/receipts/image_good/kursi_01.jpeg
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Optional

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

# ---------------------------------------------------------------------------
# Tesseract configuration
# ---------------------------------------------------------------------------

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR  = str(Path(__file__).parent.parent / "tessdata")
OCR_LANG      = "lav+eng"
OCR_CONFIG_6  = "--oem 3 --psm 6"   # Uniform text block
OCR_CONFIG_4  = "--oem 3 --psm 4"   # Single column

if _HAS_TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _ocr_image(img: "Image.Image", config: str) -> str:
    """Run Tesseract on a PIL image; return text string."""
    if not _HAS_TESSERACT:
        return ""
    try:
        return pytesseract.image_to_string(img, lang=OCR_LANG, config=config)
    except Exception:
        return ""


def _preprocess(img: "Image.Image") -> "Image.Image":
    """Increase contrast and sharpen to help with low-quality photos."""
    img = img.convert("L")                         # Greyscale
    img = ImageEnhance.Contrast(img).enhance(2.0)  # Boost contrast
    img = img.filter(ImageFilter.SHARPEN)
    # Upscale if small
    w, h = img.size
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _extract_text(image_path: str) -> tuple[str, str]:
    """
    Try OCR with psm 6; if text yield < 80 chars retry with psm 4 + preprocessing.
    Returns (text, method_used).
    """
    if not _HAS_TESSERACT:
        return "", "tesseract_unavailable"
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        return "", f"image_open_error:{exc}"

    text6 = _ocr_image(img, OCR_CONFIG_6)
    if len(text6.replace(" ", "").replace("\n", "")) >= 80:
        return text6, "tesseract_psm6"

    # Retry with preprocessing + psm 4
    try:
        proc = _preprocess(img)
        text4 = _ocr_image(proc, OCR_CONFIG_4)
        if len(text4) > len(text6):
            return text4, "tesseract_psm4_preprocessed"
    except Exception:
        pass

    if text6:
        return text6, "tesseract_psm6_low_yield"
    return "", "ocr_failed"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise OCR artefacts common in receipt photos."""
    text = text.replace("\xa0", " ")
    # Collapse multiple spaces (keep newlines)
    text = re.sub(r" {2,}", " ", text)
    # Trim trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Remove lines that are purely noise (only punctuation/symbols, no alphanumeric)
    text = "\n".join(
        line for line in text.split("\n")
        if re.search(r"[A-Za-z0-9\u0100-\u024F]", line)
    )
    return text


# ---------------------------------------------------------------------------
# Amount parser
# ---------------------------------------------------------------------------

def _parse_amount(raw: str) -> Optional[float]:
    """Parse '179.95', '179,95', '11614' → float. Handles 5-digit artefact."""
    if not raw:
        return None
    s = raw.strip().replace(" ", "")
    # Normalise decimal separator
    if "," in s and "." in s:
        s = s.replace(",", "")   # e.g. "1.179,95" → "1179.95" not ideal; handle below
    elif "," in s:
        s = s.replace(",", ".")
    try:
        val = float(s)
        # OCR artefact: 5-digit integer where last 2 digits are cents
        # e.g. 11614 → 116.14 (period dropped), 17995 → 179.95
        if val >= 1000 and val == int(val) and len(s.replace("-", "")) in (4, 5):
            val = round(val / 100, 2)
        return round(val, 2) if val > 0 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------

# Known sellers — pattern → canonical name
_SELLER_PATTERNS = [
    (r"Tirdzniec.{1,4}bas\s+nams\s+Kur.{1,3}i\s+SIA",  "Tirdzniecibas nams Kursi SIA"),
    (r"SIA\s+DEPO\s+DIY",                                "SIA DEPO DIY"),
    (r"\bDEPO\s+DIY\b",                                  "SIA DEPO DIY"),
    (r"\bNESTE\b",                                        "NESTE"),
    (r"DELVE\s+\d+",                                      "DELVE 2 SIA"),
    (r"Circle\s*K",                                       "Circle K"),
    (r"Statoil",                                          "Statoil"),
    (r"RIMI",                                             "Rimi"),
    (r"MAXIMA",                                           "Maxima"),
]


def _seller_line_quality(line: str) -> bool:
    """Return True if line looks like a legitimate company name, not OCR noise."""
    alnum = sum(1 for c in line if c.isalnum())
    noise = sum(1 for c in line if c in "\ufffd?~*|\\")
    if alnum == 0:
        return False
    if noise / max(alnum, 1) > 0.3:
        return False
    # Reject lines with garbage special chars (em-dash clusters, semicolons, asterisks)
    if re.search(r'[*;\u2014\u2013]', line):
        return False
    # Skip lines where >60% of alphabetic words are ≤2 chars (mirrors Map Row sellerSuspect)
    words = line.split()
    alpha_words = [re.sub(r'[^a-zA-Z\u0101-\u017e]', '', w) for w in words]
    alpha_words = [w for w in alpha_words if w]
    if alpha_words:
        short = sum(1 for w in alpha_words if len(w) <= 2)
        if short / len(alpha_words) > 0.6:
            return False
    return True


def _parse_seller_name(text: str) -> tuple[Optional[str], str]:
    for pattern, name in _SELLER_PATTERNS:
        if re.search(pattern, text, re.I):
            return name, "known_seller"

    # Fall back: first meaningful company-like line
    for line in text.split("\n")[:10]:
        line = line.strip()
        if len(line) > 5 and _seller_line_quality(line):
            if not re.match(r"^[\d+\s,./]+$", line):
                return line[:80], "first_meaningful_line"
    return None, "not_found"


def _parse_receipt_number(text: str) -> tuple[Optional[str], str]:
    patterns = [
        (r"Dokuments?:\s*(\d{5,})",                    "dokuments"),
        (r"PAVADZ.{1,4}ME\s+Nr[.:]?\s*(\S+)",          "pavadzime_nr"),
        (r"Kv.{1,4}ts\s+[Nn][Rr]\.?\s+(\d{4,})",      "kvits_nr"),
        (r"Kvīts\s*:\s*(\d{4,})",                       "kvits"),
        (r"(?:Čas|.as)\.\s*Nr[.:]?\s*(\S+)",           "cas_nr"),
        (r"Kv.{1,3}ts:\s*(\d{4,})",                    "kvits_short"),
    ]
    for pattern, method in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            return m.group(1).strip(), method
    return None, "not_found"


def _valid_date(yr: str, mo: str, dy: str) -> Optional[str]:
    """Return ISO date string if valid, else None."""
    try:
        y, m, d = int(yr), int(mo), int(dy)
        if 2020 <= y <= 2035 and 1 <= m <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{m:02d}-{d:02d}"
    except ValueError:
        pass
    return None


def _parse_receipt_date(text: str) -> tuple[Optional[str], str]:
    # Pattern 1: ISO "Datums: 2026-01-30" (with possible OCR noise on 'D')
    m = re.search(r".{0,2}atums[:\s]+(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        d = _valid_date(m.group(1), m.group(2), m.group(3))
        if d: return d, "datums_iso"

    # Pattern 2: LAIKS/timestamp "LAIKS 2026-02-11 09:42:56"
    m = re.search(r"LAIKS\s+(\d{4})-(\d{2})-(\d{2})", text, re.I)
    if m:
        d = _valid_date(m.group(1), m.group(2), m.group(3))
        if d: return d, "laiks"

    # Pattern 3: "28.01.2026 11:22:14" near Dokuments
    m = re.search(r"Dokuments?:.*?(\d{2})\.(\d{2})\.(\d{4})", text, re.I | re.DOTALL)
    if m:
        d = _valid_date(m.group(3), m.group(2), m.group(1))
        if d: return d, "dokuments_date"

    # Pattern 4: standalone "DD.MM.YYYY HH:MM"
    for m in re.finditer(r"\b(\d{2})\.(\d{2})\.(\d{4})\s+\d{2}:\d{2}", text):
        d = _valid_date(m.group(3), m.group(2), m.group(1))
        if d: return d, "datetime_stamp"

    # Pattern 5: "DATUMS 28.01. 2026" (card slip with optional space)
    m = re.search(r"DATUMS\s+(\d{2})\.(\d{2})\.\s*(\d{4})", text, re.I)
    if m:
        d = _valid_date(m.group(3), m.group(2), m.group(1))
        if d: return d, "datums_card"

    # Pattern 6: Latvian written "2026. gada 28. janvāris"
    MONTHS = {"janv": "01", "febr": "02", "mart": "03", "apr": "04",
              "mai": "05", "j.n": "06", "j.l": "07", "aug": "08",
              "sept": "09", "okt": "10", "nov": "11", "dec": "12"}
    m = re.search(r"(\d{4})\.\s*gada\s+(\d{1,2})\.\s*([a-zA-Z.]{3,6})", text, re.I)
    if m:
        yr, day, mon_raw = m.group(1), m.group(2).zfill(2), m.group(3)[:4].lower()
        for key, num in MONTHS.items():
            if re.match(key, mon_raw):
                d = _valid_date(yr, num, day)
                if d: return d, "latvian_long_date"

    return None, "not_found"


def _parse_total_gross(text: str) -> tuple[Optional[float], str]:
    patterns = [
        # Most Latvian POS receipts ("Kopsumma" or "Kopsuma" — single m OCR variant)
        (r"Kopsumm?a\s+EUR\s+([\d.,]+)",                 "kopsumma_eur"),
        (r"kopsumm?a\s+([\d.,]+)",                        "kopsumma"),
        # Card slip duplicate
        (r"SUMMA[:\s]+([\d.,]+)\s*EUR",                   "summa_eur"),
        # KOPĀ line (with OCR noise tolerance)
        (r"KOP.{1,5}[:\s]+a?\s*([\d]+[.,]\d{2})",        "kopa"),
        # "kops apmaksai / kopā apmaksai" (with 5-digit artefact handled in _parse_amount)
        (r"kop[a-z.]{1,5}\s+ap[nm]aksai\s+([\d.,]+)",    "kopa_apmaksai"),
        # "Apmaksa (Bankas kartes) 116.14"
        (r"Apmaksa\s*\(Bankas\s+kartes\)\s*([\d.,]+)",    "apmaksa_karte"),
        # "Apmaksāts EUR X" / "oamaksāts EUR X" (fuel — OCR noise on first chars)
        (r"\w{1,5}maks\w{1,5}ts\s+EUR\s+([\d.,]+)",       "apmaksats_eur"),
    ]
    for pattern, method in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            val = _parse_amount(m.group(1))
            if val and val > 0:
                return val, method
    return None, "not_found"


def _parse_vat_summary(text: str) -> tuple[Optional[float], Optional[float], str]:
    """
    Extract total_net (bez PVN) and total_vat from VAT summary line.
    Returns (total_net, total_vat, method).

    Common Latvian receipt formats:
      PVN-B 21% 3.63 17.27 20.90     → vat=3.63, net=17.27, gross=20.90
      4=21,00% 95,98 20,16 116,14    → net=95.98, vat=20.16, gross=116.14
      PVN-8 21% — 31.28 148.72 179.95 → vat=31.28, net=148.72, gross=179.95
    """
    # General pattern: finds 3 floats after "PVN" rate marker
    patterns = [
        r"PVN-[A-Z0-9]\s+\d+[.,]?\d*%?\s*[-—]?\s*([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)",
        r"\d+[=]\s*\d+[.,]\d+%\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)",
        r"PVN[- ][A-Z]\s+\d+%\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            nums = []
            for g in [m.group(1), m.group(2), m.group(3)]:
                v = _parse_amount(g)
                if v is not None:
                    nums.append(v)
            if len(nums) == 3:
                a, b, c = sorted(nums)
                # c should be gross (largest), a and b are net and vat
                # Determine which is net: net + vat = gross
                if abs(a + b - c) < 0.10:
                    # a=vat or net, b=the other
                    # net > vat always for 21%: net * 0.21 = vat → vat ≈ 17.4% of gross
                    net = max(a, b)
                    vat = min(a, b)
                    return net, vat, "pvn_summary_line"
    return None, None, "not_found"


_ITEM_SKIP = re.compile(
    r"kopsumm?a|apmaks|pvn[- ]|datums|laiks|dokuments|kvīts|kvits|karte|visa"
    r"|mastercard|skaidra|nauda|kopā|kopa|atlaide|paldies|rēķins|rekins"
    r"|pavadzīme|pavadzime|summa\s+eur|apmaksāts|maksāt|čeks"
    r"|klientai|klients|kasier|kase\s+nr|s/n:|fa:|reģ\.|pvn\s+maks"
    r"|juridisk|adrese|iela\s+\d|nov\.|pag\.|lv-\d{4}"
    r"|\bSIA\b|\bIK\b|\bAS\b"           # company legal forms → buyer/seller line
    r"|\blv\d{8,}\b"                    # VAT/registration number (LV40003242737)
    r"|\d{2}\.\d{2}\.\d{4}|^\d[\d.,\s]*$",
    re.I | re.UNICODE,
)


def _parse_item_description(text: str) -> Optional[str]:
    """
    Extract the first identifiable product/service line from a receipt.
    Skips the seller header area (first 8 lines), totals, dates, and noise.
    """
    lines = text.split("\n")
    for line in lines[8:]:
        line = line.strip()
        if len(line) < 4:
            continue
        if not re.search(r"[A-Za-z\u0100-\u024F]", line):
            continue        # numeric / symbol only
        if _ITEM_SKIP.search(line):
            continue
        if re.match(r"^[\d.,\s]+$", line):
            continue        # pure amount line
        # Skip lines where >50% of words are ≤2 chars — likely OCR noise or
        # a buyer/seller address fragment ("ā am Y", initials, etc.)
        words = line.split()
        if len(words) >= 3:
            short = sum(1 for w in words if len(re.sub(r'[^a-zA-Z\u0101-\u017e]', '', w)) <= 2)
            if short / len(words) > 0.5:
                continue
        return line[:80]
    return None


def _parse_payment_method(text: str) -> Optional[str]:
    if re.search(r"Skaidra\s+nauda|sk\.nauda|Skaidra\s+naud", text, re.I):
        return "cash"
    if re.search(r"Karte|Mastercard|Visa|BEZKONTAKTA|kartes\b", text, re.I):
        return "card"
    if re.search(r"Bankas\s+p.rskait|p.rskait.jums", text, re.I):
        return "bank_transfer"
    return None


def _parse_discounts(text: str) -> list:
    """Extract discount lines; returns list of {description, amount}."""
    discounts = []
    seen = set()

    # "Preces atlaide 12% (Klientam) -0.64"
    for m in re.finditer(
        r"Preces\s+atlaide\s+(\d+%?).*?(-[\d.,]+)", text, re.I
    ):
        raw = m.group(2)
        val = _parse_amount(raw.lstrip("-"))
        if val and val > 0:
            key = (m.group(1), val)
            if key not in seen:
                seen.add(key)
                discounts.append({"description": f"Preces atlaide {m.group(1)}", "amount": -val})

    # "(Klientam) -3.22"
    for m in re.finditer(r"\(Klientam\)\s+(-[\d.,]+)", text):
        val = _parse_amount(m.group(1).lstrip("-"))
        if val and val > 0:
            key = ("klientam", val)
            if key not in seen:
                seen.add(key)
                discounts.append({"description": "Atlaide klientam", "amount": -val})

    # "Cenu samazinājums ar DEPO karti -2.32" (DEPO loyalty card)
    for m in re.finditer(
        r"Cenu\s+samazin.{1,6}jums\s+ar\s+\S+\s+karti\s+(-?[\d.,]+)", text, re.I
    ):
        val = _parse_amount(m.group(1).lstrip("-"))
        if val and val > 0:
            key = ("depo_loyalty", val)
            if key not in seen:
                seen.add(key)
                discounts.append({"description": "Cenu samazinājums ar karti", "amount": -val})

    return discounts


# ---------------------------------------------------------------------------
# OCR confidence
# ---------------------------------------------------------------------------

def _ocr_confidence(text: str) -> str:
    if not text:
        return "low"
    noise = text.count("\ufffd") + text.count("~")
    total = len(text.replace("\n", "").replace(" ", "")) or 1
    ratio = noise / total
    if ratio < 0.05:
        return "high"
    if ratio < 0.15:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_receipt(image_path: str) -> dict:
    """
    Extract structured fields from a receipt image.
    Always returns a valid dict.
    """
    path = Path(image_path)

    result = {
        "source_file":      path.name,
        "document_family":  "receipt_image",
        "seller_name":      None,
        "receipt_number":   None,
        "receipt_date":     None,
        "item_description": None,
        "currency":         "EUR",
        "total_gross":      None,
        "total_net":        None,
        "total_vat":        None,
        "payment_method":   None,
        "discounts":        [],
        "line_items":       [],
        "ocr_engine":       None,
        "extraction_method": "ocr",
        "warnings":         [],
        "field_sources":    {},
    }

    warnings: list = result["warnings"]
    src: dict      = result["field_sources"]

    # --- OCR ---
    raw_text, method = _extract_text(str(path))
    result["ocr_engine"] = "tesseract_lav_eng"
    result["extraction_method"] = method

    if not raw_text or len(raw_text.replace(" ", "").replace("\n", "")) < 20:
        warnings.append(f"OCR returned insufficient text: {method}")
        result["extraction_method"] = "failed"
        return result

    raw_text = _clean_text(raw_text)

    # --- Fields ---
    result["seller_name"],   src["seller_name"]   = _parse_seller_name(raw_text)
    result["receipt_number"],src["receipt_number"] = _parse_receipt_number(raw_text)
    result["receipt_date"],  src["receipt_date"]   = _parse_receipt_date(raw_text)
    result["total_gross"],   src["total_gross"]    = _parse_total_gross(raw_text)
    result["item_description"]                     = _parse_item_description(raw_text)
    result["payment_method"]                       = _parse_payment_method(raw_text)
    result["discounts"]                            = _parse_discounts(raw_text)

    # VAT summary
    total_net, total_vat, vat_method = _parse_vat_summary(raw_text)
    if total_net is not None:
        result["total_net"] = total_net
        src["total_net"]    = vat_method
    if total_vat is not None:
        result["total_vat"] = total_vat
        src["total_vat"]    = vat_method

    # --- Triangle validation ---
    triangle_status = "INCOMPLETE"
    if result["total_gross"] is not None:
        if result["total_net"] is not None and result["total_vat"] is not None:
            computed = round(result["total_net"] + result["total_vat"], 2)
            if abs(computed - result["total_gross"]) > 0.10:
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

    # --- Missing field warnings ---
    for field in ("receipt_date", "total_gross", "seller_name"):
        if not result.get(field):
            warnings.append(f"{field}: extraction failed")

    # --- Confidence ---
    ocr_qual  = _ocr_confidence(raw_text)
    n_missing = sum(1 for f in ("receipt_date", "total_gross", "seller_name")
                    if not result.get(f))
    if triangle_status == "NEEDS_REVIEW" or n_missing > 1:
        result["confidence"] = "low"
    elif n_missing == 1 or warnings or ocr_qual == "low":
        result["confidence"] = "medium"
    else:
        result["confidence"] = "high" if ocr_qual != "medium" else "medium"

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_receipt_image.py <path_to_image>")
        sys.exit(1)
    result = extract_receipt(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))
