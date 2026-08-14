#!/usr/bin/env python3
"""
test_verbatim_prescan.py
-------------------------
Unit tests for Cluster A verbatim classifier prescan.

Run: python -m pytest tools/test_verbatim_prescan.py -v
Or:  python tools/test_verbatim_prescan.py
"""
from __future__ import annotations
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.extract_master_router import verbatim_prescan


class TestVerbatimPrescan(unittest.TestCase):
    def test_kreditrekins_exact_match(self):
        r = verbatim_prescan("Kredītrēķins Nr. 12345\nDatums: 08.03.2026")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "credit_note")
        self.assertEqual(r["matched_token"], "kredītrēķins")
        self.assertEqual(r["fuzzy_distance"], 0)

    def test_kredita_rekins_two_word_match(self):
        r = verbatim_prescan("Kredīta rēķins Nr. 001\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "credit_note")

    def test_avansa_rekins_advance_invoice(self):
        r = verbatim_prescan("Avansa rēķins Nr. 42\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "advance_invoice")

    def test_pro_forma_hyphenated_variant(self):
        r = verbatim_prescan("Pro-forma invoice\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "advance_invoice")
        # 'pro forma' (space) matches at fuzzy-1 before 'pro-forma' — same route,
        # semantic pass regardless of which token wins.
        self.assertIn(r["matched_token"], {"pro forma", "pro-forma"})

    def test_pro_forma_spaced_variant(self):
        r = verbatim_prescan("Pro forma invoice\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "advance_invoice")

    def test_priekshapmaksa_advance(self):
        r = verbatim_prescan("Priekšapmaksa 30%\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "advance_invoice")

    def test_z_atskaite_exact_only(self):
        r = verbatim_prescan("Z-atskaite\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "receipt_z")
        self.assertEqual(r["fuzzy_distance"], 0)

    def test_x_atskaite_exact_only_no_zx_collision(self):
        r = verbatim_prescan("X-atskaite\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "receipt_x")
        self.assertEqual(r["fuzzy_distance"], 0)

    def test_ieskaita_akts_credit_note(self):
        r = verbatim_prescan("Ieskaita akts Nr. 5\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "credit_note")

    def test_savstarpējais_ieskaits_credit_note(self):
        r = verbatim_prescan("Savstarpējais ieskaits\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "credit_note")

    def test_pavadzime_standalone_delivery_note(self):
        r = verbatim_prescan("Pavadzīme Nr. 100\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "delivery_note")

    def test_pavadzime_in_hybrid_rejected(self):
        # 'Rēķins-pavadzīme' is a HYBRID (normal invoice) — must NOT fire
        # delivery_note. Standalone guard prevents this.
        r = verbatim_prescan("Rēķins-pavadzīme Nr. 001\n")
        self.assertFalse(r["verbatim_hit"])

    def test_plain_invoice_no_hit(self):
        r = verbatim_prescan("Rēķins Nr. 001\nSIA Piemēra Pārdevējs\n")
        self.assertFalse(r["verbatim_hit"])

    def test_body_reference_not_in_header_ignored(self):
        # Header region is first ~15 lines / 800 chars. A body reference to
        # 'kredītrēķins' shouldn't trigger.
        long_header = "\n".join([f"Some line {i}" for i in range(1, 20)])
        text = long_header + "\nkredītrēķins Nr. 5 (reference in body)\n"
        r = verbatim_prescan(text)
        self.assertFalse(r["verbatim_hit"])

    def test_empty_text_no_op(self):
        r = verbatim_prescan("")
        self.assertFalse(r["verbatim_hit"])
        self.assertIsNone(r["route_hint"])

    def test_fuzzy_distance_1_ocr_dropout(self):
        # OCR drops one char: "kredītrēķns" (missing i between ķ and n) → dist 1
        r = verbatim_prescan("Kredītrēķns Nr. 12345\n")
        self.assertTrue(r["verbatim_hit"])
        self.assertEqual(r["route_hint"], "credit_note")
        self.assertLessEqual(r["fuzzy_distance"], 1)

    def test_z_x_no_fuzzy_prevents_collision(self):
        # 'y-atskaite' is edit-distance-1 from both z-atskaite and x-atskaite;
        # since we set allow_fuzzy=False on both, no false positive.
        r = verbatim_prescan("Y-atskaite\n")
        self.assertFalse(r["verbatim_hit"])


class TestVerbatimModeResolver(unittest.TestCase):
    def test_default_off(self):
        os.environ.pop("VERBATIM_CLASSIFIER", None)
        from tools.ocr_bridge_master_router import _resolve_verbatim_mode
        self.assertEqual(_resolve_verbatim_mode(), "off")

    def test_shadow(self):
        os.environ["VERBATIM_CLASSIFIER"] = "shadow"
        try:
            from tools.ocr_bridge_master_router import _resolve_verbatim_mode
            self.assertEqual(_resolve_verbatim_mode(), "shadow")
        finally:
            os.environ.pop("VERBATIM_CLASSIFIER", None)

    def test_enforce_case_insensitive(self):
        os.environ["VERBATIM_CLASSIFIER"] = "ENFORCE"
        try:
            from tools.ocr_bridge_master_router import _resolve_verbatim_mode
            self.assertEqual(_resolve_verbatim_mode(), "enforce")
        finally:
            os.environ.pop("VERBATIM_CLASSIFIER", None)

    def test_unknown_value_falls_back_to_off(self):
        os.environ["VERBATIM_CLASSIFIER"] = "on"  # typo
        try:
            from tools.ocr_bridge_master_router import _resolve_verbatim_mode
            self.assertEqual(_resolve_verbatim_mode(), "off")
        finally:
            os.environ.pop("VERBATIM_CLASSIFIER", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
