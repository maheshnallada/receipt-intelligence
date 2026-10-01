"""Pure total checks.

Policy: numbers that fail a sum check stay in the payload (they appeared on
the document) but the affected leaves become unverified. Tolerance is 0.05
to cover ordinary currency rounding.
"""

from __future__ import annotations

from app.domain.grounding import SANITY_TOLERANCE
from app.domain.receipt import Receipt, Verification


def line_items_sum(receipt: Receipt) -> float | None:
    amounts = [item.amount.value for item in receipt.line_items if item.amount is not None]
    if not amounts:
        return None
    return sum(amounts)


def sums_match(left: float | None, right: float | None, *, tolerance: float = SANITY_TOLERANCE) -> bool:
    if left is None or right is None:
        return True
    return abs(left - right) <= tolerance


def check_totals(receipt: Receipt, *, tolerance: float = SANITY_TOLERANCE) -> Receipt:
    """Return a new Receipt; mismatched money fields become unverified."""
    verification = dict(receipt.verification)
    items_total = line_items_sum(receipt)
    subtotal = receipt.subtotal.value if receipt.subtotal else None
    tax = receipt.tax.value if receipt.tax else None
    total = receipt.total.value if receipt.total else None

    if items_total is not None and subtotal is not None and not sums_match(items_total, subtotal, tolerance=tolerance):
        verification["subtotal"] = Verification.UNVERIFIED
        for index, item in enumerate(receipt.line_items):
            if item.amount is not None:
                verification[f"line_items.{index}.amount"] = Verification.UNVERIFIED

    if subtotal is not None and total is not None:
        expected = subtotal + (tax if tax is not None else 0.0)
        if not sums_match(expected, total, tolerance=tolerance):
            verification["subtotal"] = Verification.UNVERIFIED
            verification["total"] = Verification.UNVERIFIED
            if tax is not None:
                verification["tax"] = Verification.UNVERIFIED

    return receipt.with_verification(verification)


def describe_failures(receipt: Receipt, *, tolerance: float = SANITY_TOLERANCE) -> list[str]:
    """Plain-language banners for the receipt desk."""
    messages: list[str] = []
    items_total = line_items_sum(receipt)
    subtotal = receipt.subtotal.value if receipt.subtotal else None
    tax = receipt.tax.value if receipt.tax else None
    total = receipt.total.value if receipt.total else None
    if items_total is not None and subtotal is not None and not sums_match(items_total, subtotal, tolerance=tolerance):
        messages.append("Line items do not add up to the subtotal.")
    if subtotal is not None and total is not None:
        expected = subtotal + (tax if tax is not None else 0.0)
        if not sums_match(expected, total, tolerance=tolerance):
            messages.append("Subtotal plus tax does not equal the total.")
    return messages
