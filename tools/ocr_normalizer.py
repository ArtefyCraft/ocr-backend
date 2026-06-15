#!/usr/bin/env python3
"""
ocr_normalizer.py
------------------
Post-bridge normalizer: financial inference + standardized output contract.

Called by ocr_bridge_master_router.py after bridge dispatch.

Responsibilities:
  1. Complete the gross/net/vat triangle from available data.
  2. Build new top-level contract fields alongside the existing ones
     so downstream nodes (Normalize Output, Map Row) can read them.

New contract fields added to master output:
  statuss                       "OK" | "JĀPĀRBAUDA" | "KĻŪDA"
  problemas                     list[str]   — human-readable Latvian problem list
  datums                        str | None  — ISO date
  pardevejs                     str | None  — seller name
  pvn_kopa                      float | None
  precu_pakalpojumu_summa_kopa  float | None  (total gross / booking amount)
  rindas                        list[dict]  — one entry per line item
    Each rinda: {prece_pakalpojums, summa_ar_pvn, summa_bez_pvn, atlaide}
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIANGLE_TOLERANCE = 0.10
# VAT rates tried when gross only is available (most common Latvian rates first)
_VAT_RATES = (0.21, 0.12, 0.05)

# ---------------------------------------------------------------------------
# Numeric helper
# ---------------------------------------------------------------------------


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if f >= 0 else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Seller quality (mirrors Map Row v4 sellerSuspect logic)
# ---------------------------------------------------------------------------


def _seller_suspect(name: Optional[str]) -> bool:
    """Return True if the seller name looks like OCR garbage."""
    if not name:
        return False
    if re.search(r'[*;\u2014\u2013\u22c5]', name):
        return True
    words = name.strip().split()
    if len(words) < 4:
        return False
    alpha_words = [re.sub(r'[^a-zA-Z\u0101-\u017e]', '', w) for w in words]
    alpha_words = [w for w in alpha_words if w]
    if not alpha_words:
        return True
    short = sum(1 for w in alpha_words if len(w) <= 2)
    return short / len(alpha_words) > 0.6


# ---------------------------------------------------------------------------
# Financial inference
# ---------------------------------------------------------------------------


def infer_financials(ext: dict) -> dict:
    """
    Complete the gross/net/vat triangle from available data.

    Returns a new dict (copy of ext) with:
      - possibly filled total_gross, total_net, total_vat
      - updated triangle_status ("OK" | "NEEDS_REVIEW" | "INCOMPLETE")
      - _inferred: list of what was inferred (empty if nothing changed)
    """
    out      = dict(ext)
    inferred: list[str] = []

    gross = _num(out.get("total_gross"))
    net   = _num(out.get("total_net"))
    vat   = _num(out.get("total_vat"))

    if gross is not None and net is not None and vat is not None:
        # All three present — validate triangle
        computed = round(net + vat, 2)
        if abs(computed - gross) <= TRIANGLE_TOLERANCE:
            out["triangle_status"] = "OK"
        else:
            out["triangle_status"] = "NEEDS_REVIEW"
        out["_inferred"] = inferred
        return out

    # net + vat → gross
    if gross is None and net is not None and vat is not None:
        out["total_gross"] = round(net + vat, 2)
        inferred.append("total_gross=net+vat")
        out["triangle_status"] = "OK"

    # gross + vat → net
    elif net is None and gross is not None and vat is not None:
        out["total_net"] = round(gross - vat, 2)
        inferred.append("total_net=gross-vat")
        out["triangle_status"] = "OK"

    # gross + net → vat
    elif vat is None and gross is not None and net is not None:
        out["total_vat"] = round(gross - net, 2)
        inferred.append("total_vat=gross-net")
        out["triangle_status"] = "OK"

    # gross only → try standard VAT rates
    elif gross is not None and net is None and vat is None:
        for rate in _VAT_RATES:
            vat_calc = round(gross * rate / (1 + rate), 2)
            net_calc = round(gross - vat_calc, 2)
            if abs(net_calc + vat_calc - gross) < 0.02:
                out["total_vat"] = vat_calc
                out["total_net"] = net_calc
                inferred.append(
                    f"net+vat inferred from gross@{int(rate * 100)}% (assumed rate)"
                )
                out["triangle_status"] = "INCOMPLETE"  # rate was assumed, not extracted
                break
        else:
            out["triangle_status"] = out.get("triangle_status", "INCOMPLETE")

    else:
        # No usable numbers at all
        out["triangle_status"] = out.get("triangle_status", "INCOMPLETE")

    out["_inferred"] = inferred
    return out


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def _build_rindas(ext: dict, fam: str, gross, net, _vat) -> list:
    """Build the rindas array from extracted_data."""
    items_arr      = ext.get("items", []) or []
    line_items_arr = ext.get("line_items", []) or []

    if items_arr:
        rindas = []
        for item in items_arr:
            prece = (
                item.get("description") or item.get("item") or
                item.get("name") or item.get("service") or item.get("product") or None
            )
            item_gross = _num(next(
                (item.get(k) for k in ["total_gross", "gross_amount", "total", "amount"]
                 if item.get(k) is not None), None
            ))
            item_net = _num(next(
                (item.get(k) for k in ["net_amount", "amount_excl_vat", "net"]
                 if item.get(k) is not None), None
            ))
            rindas.append({
                "prece_pakalpojums": prece,
                "summa_ar_pvn":      item_gross,
                "summa_bez_pvn":     item_net,
                "atlaide":           None,
            })
        return rindas

    if line_items_arr and fam == "telecom_pdf":
        rindas = []
        for li in line_items_arr:
            conn_id = li.get("connection_id", "")
            prece   = f"Savienojums {conn_id}" if conn_id else "Telefonsakaru pakalpojums"
            sub     = _num(li.get("subtotal_eur"))
            rindas.append({
                "prece_pakalpojums": prece,
                "summa_ar_pvn":      sub,
                "summa_bez_pvn":     None,
                "atlaide":           None,
            })
        return rindas

    if line_items_arr:
        rindas = []
        for li in line_items_arr:
            prece = li.get("name") or li.get("description") or li.get("item") or None
            li_gross = _num(next(
                (li.get(k) for k in ["total", "amount", "price", "gross", "total_gross"]
                 if li.get(k) is not None), None
            ))
            li_net = _num(next(
                (li.get(k) for k in ["net", "net_amount", "amount_excl_vat"]
                 if li.get(k) is not None), None
            ))
            rindas.append({
                "prece_pakalpojums": prece,
                "summa_ar_pvn":      li_gross,
                "summa_bez_pvn":     li_net,
                "atlaide":           None,
            })
        return rindas

    # Single-row document
    if fam == "telecom_pdf":
        prece  = "Telefonsakaru pakalpojumi"
        sg     = _num(ext.get("booking_amount")) or gross
    else:
        prece = (
            ext.get("item_description") or ext.get("service_description") or
            ext.get("description") or None
        )
        sg = gross

    td      = _num(ext.get("total_discount"))
    atlaide = abs(td) if td is not None and td != 0 else None

    return [{
        "prece_pakalpojums": prece,
        "summa_ar_pvn":      sg,
        "summa_bez_pvn":     net,
        "atlaide":           atlaide,
    }]


# ---------------------------------------------------------------------------
# Main contract builder
# ---------------------------------------------------------------------------


def build_output_contract(master_out: dict) -> dict:
    """
    Enrich master router output with new standardized contract fields.

    Reads from master_out["extracted_data"] and master_out["status"].
    Adds new keys at the top level; does NOT remove any existing keys.
    Returns master_out (mutated in place).

    statuss logic:
      KĻŪDA      — bridge returned FAILED (system / OCR error, not a business problem)
      JĀPĀRBAUDA — extraction succeeded but datums/pardevejs/prece/sum are missing or suspect
      OK         — datums + pardevejs + at least one prece + sum all present and not suspect
    """
    ext         = master_out.get("extracted_data", {}) or {}
    fam         = master_out.get("document_family", "unknown")
    orig_status = master_out.get("status", "FAILED")

    # --- Financial inference ---
    ext = infer_financials(ext)
    master_out["extracted_data"] = ext

    problems: list[str] = []

    # --- Canonical fields ---
    datums    = ext.get("invoice_date") or ext.get("receipt_date") or None
    pardevejs = ext.get("seller_name") or None

    gross = _num(ext.get("total_gross"))
    net   = _num(ext.get("total_net"))
    vat   = _num(ext.get("total_vat"))

    # Override gross with booking_amount for telecom
    if fam == "telecom_pdf":
        bk = _num(ext.get("booking_amount"))
        if bk is not None:
            gross = bk

    # --- Problem detection ---
    if not datums:
        problems.append("Nav izdevās noteikt datumu")

    if not pardevejs:
        problems.append("Nav izdevās noteikt pārdevēju")
    elif _seller_suspect(pardevejs):
        problems.append("Pārdevēja nosaukums var būt neprecīzs (OCR)")

    if gross is None:
        problems.append("Nav izdevās noteikt kopējo summu")

    ts = ext.get("triangle_status", "INCOMPLETE")
    if ts == "NEEDS_REVIEW":
        problems.append("Summas neatbilst — PVN aprēķins kļūdains")
    elif ts == "INCOMPLETE":
        inferred = ext.get("_inferred", [])
        if any("assumed rate" in s for s in inferred):
            problems.append("PVN likme pieņemta (nav atrasta dokumentā) — pārbaudīt")
        elif net is None and fam != "telecom_pdf":
            problems.append("Summa bez PVN nav noteikta")

    # --- Row building ---
    rindas = _build_rindas(ext, fam, gross, net, vat)

    has_prece = any(r.get("prece_pakalpojums") for r in rindas)
    if not has_prece:
        problems.append("Nav izdevās noteikt preci/pakalpojumu")

    # --- Statuss ---
    if orig_status == "FAILED":
        statuss = "KĻŪDA"
    elif problems:
        statuss = "JĀPĀRBAUDA"
    else:
        statuss = "OK"

    # --- Write new contract fields ---
    master_out["statuss"]                      = statuss
    master_out["problemas"]                    = problems
    master_out["datums"]                       = datums
    master_out["pardevejs"]                    = pardevejs
    master_out["pvn_kopa"]                     = vat
    master_out["precu_pakalpojumu_summa_kopa"] = gross
    master_out["rindas"]                       = rindas

    return master_out
