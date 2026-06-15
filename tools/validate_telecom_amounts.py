#!/usr/bin/env python3
"""
validate_telecom_amounts.py
---------------------------
Validates extracted telecom bill amounts for internal consistency.

Rules enforced:
  1. Required fields present
  2. Triangle check: total_payable ≈ current_period + previous_balance (±0.02)
  3. Booking amount equals current_period_amount
  4. All amounts are non-negative (except if credit note — not applicable to Tele2)
  5. Booking amount never equals total_payable when previous_balance > 0

Usage:
  from tools.validate_telecom_amounts import validate_telecom
  report = validate_telecom(extracted_dict)
"""

from typing import Optional


TOLERANCE = 0.02  # EUR rounding tolerance for triangle check


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_required_fields(data: dict) -> list[dict]:
    """All critical fields must be present (not None)."""
    issues = []
    required = {
        "seller_name": "Seller name missing",
        "invoice_number": "Invoice number missing",
        "invoice_date": "Invoice date missing",
        "total_payable": "Total payable missing",
        "current_period_amount": "Current period amount missing — cannot determine booking amount",
        "booking_amount": "Booking amount missing",
    }
    for field, message in required.items():
        if data.get(field) is None:
            issues.append({
                "rule": "required_fields",
                "field": field,
                "status": "FAIL",
                "message": message,
            })
    return issues


def _check_triangle(data: dict) -> list[dict]:
    """
    total_payable = current_period_amount + previous_balance_or_overdue ± tolerance
    This is the core Tele2 accounting check.
    """
    total = data.get("total_payable")
    current = data.get("current_period_amount")
    previous = data.get("previous_balance_or_overdue") or 0.0

    if total is None or current is None:
        return [{
            "rule": "triangle_check",
            "status": "SKIP",
            "message": "Cannot check triangle — total_payable or current_period_amount missing",
        }]

    expected_total = round(current + previous, 2)
    actual_total = round(total, 2)
    diff = abs(actual_total - expected_total)

    if diff <= TOLERANCE:
        return [{
            "rule": "triangle_check",
            "status": "PASS",
            "message": (
                f"{actual_total:.2f} = {current:.2f} (current) "
                f"+ {previous:.2f} (previous) [diff={diff:.2f}]"
            ),
        }]
    else:
        return [{
            "rule": "triangle_check",
            "status": "FAIL",
            "message": (
                f"total_payable {actual_total:.2f} ≠ "
                f"current {current:.2f} + previous {previous:.2f} "
                f"= {expected_total:.2f} [diff={diff:.2f}, tolerance={TOLERANCE}]"
            ),
        }]


def _check_booking_equals_current(data: dict) -> list[dict]:
    """booking_amount must equal current_period_amount."""
    booking = data.get("booking_amount")
    current = data.get("current_period_amount")

    if booking is None or current is None:
        return [{
            "rule": "booking_equals_current",
            "status": "SKIP",
            "message": "Cannot check — booking_amount or current_period_amount missing",
        }]

    if abs(booking - current) <= TOLERANCE:
        return [{
            "rule": "booking_equals_current",
            "status": "PASS",
            "message": f"booking_amount {booking:.2f} == current_period_amount {current:.2f}",
        }]
    else:
        return [{
            "rule": "booking_equals_current",
            "status": "FAIL",
            "message": (
                f"booking_amount {booking:.2f} ≠ current_period_amount {current:.2f} "
                f"— booking amount was incorrectly set"
            ),
        }]


def _check_booking_not_total(data: dict) -> list[dict]:
    """
    When previous_balance > 0, booking_amount must NOT equal total_payable.
    This is the core Tele2 business rule guard.
    """
    booking = data.get("booking_amount")
    total = data.get("total_payable")
    previous = data.get("previous_balance_or_overdue") or 0.0

    if booking is None or total is None:
        return []  # Cannot check, skip silently

    if previous > TOLERANCE:
        if abs(booking - total) <= TOLERANCE:
            return [{
                "rule": "booking_not_total_when_overdue",
                "status": "FAIL",
                "message": (
                    f"CRITICAL: previous_balance={previous:.2f} exists but "
                    f"booking_amount={booking:.2f} equals total_payable={total:.2f}. "
                    f"Previous balance must be excluded from booking."
                ),
            }]
        else:
            return [{
                "rule": "booking_not_total_when_overdue",
                "status": "PASS",
                "message": (
                    f"booking_amount {booking:.2f} correctly excludes "
                    f"previous_balance {previous:.2f} from total {total:.2f}"
                ),
            }]
    return [{
        "rule": "booking_not_total_when_overdue",
        "status": "PASS",
        "message": "No previous balance — not applicable",
    }]


def _check_amounts_positive(data: dict) -> list[dict]:
    """All monetary fields must be >= 0."""
    issues = []
    fields = ["total_payable", "current_period_amount",
              "previous_balance_or_overdue", "booking_amount"]
    for field in fields:
        val = data.get(field)
        if val is not None and val < 0:
            issues.append({
                "rule": "amounts_positive",
                "field": field,
                "status": "FAIL",
                "message": f"{field} = {val:.2f} is negative (unexpected for Tele2 bill)",
            })
    if not issues:
        issues.append({
            "rule": "amounts_positive",
            "status": "PASS",
            "message": "All extracted amounts are non-negative",
        })
    return issues


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_telecom(data: dict) -> dict:
    """
    Run all validation checks on an extracted telecom bill dict.

    Returns:
      {
        "overall_status": "PASS" | "FAIL" | "JĀPĀRBAUDA",
        "checks": [...],
        "summary": "human-readable summary",
        "warnings_from_extraction": [...],
      }
    """
    checks = []
    checks.extend(_check_required_fields(data))
    checks.extend(_check_triangle(data))
    checks.extend(_check_booking_equals_current(data))
    checks.extend(_check_booking_not_total(data))
    checks.extend(_check_amounts_positive(data))

    failed = [c for c in checks if c["status"] == "FAIL"]
    skipped = [c for c in checks if c["status"] == "SKIP"]
    extraction_warnings = data.get("warnings", [])

    if failed:
        overall = "FAIL"
    elif skipped or extraction_warnings:
        overall = "JĀPĀRBAUDA"
    else:
        overall = "PASS"

    summary_parts = [f"Overall: {overall}"]
    summary_parts.append(
        f"{len([c for c in checks if c['status'] == 'PASS'])} passed, "
        f"{len(failed)} failed, "
        f"{len(skipped)} skipped"
    )
    if extraction_warnings:
        summary_parts.append(f"Extraction warnings: {len(extraction_warnings)}")

    return {
        "overall_status": overall,
        "checks": checks,
        "summary": " | ".join(summary_parts),
        "warnings_from_extraction": extraction_warnings,
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python validate_telecom_amounts.py <extracted_json_file>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    report = validate_telecom(data)
    print(json.dumps(report, indent=2, ensure_ascii=False))
