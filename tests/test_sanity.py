from app.domain.receipt import LineItem, Money, Receipt, Verification, default_verification
from app.domain.sanity import check_totals, describe_failures


def _receipt(*, items: float, subtotal: float, tax: float, total: float) -> Receipt:
    return Receipt(
        store_name=None,
        date=None,
        line_items=(LineItem(name="Tea", qty=1, unit_price=Money(items), amount=Money(items)),),
        subtotal=Money(subtotal),
        tax=Money(tax),
        total=Money(total),
        verification={
            **default_verification(1),
            "line_items.0.name": Verification.VERIFIED,
            "line_items.0.amount": Verification.VERIFIED,
            "subtotal": Verification.VERIFIED,
            "tax": Verification.VERIFIED,
            "total": Verification.VERIFIED,
        },
    )


def test_matching_sums_stay_verified() -> None:
    receipt = check_totals(_receipt(items=10.0, subtotal=10.0, tax=1.0, total=11.0))
    assert receipt.verification["subtotal"] is Verification.VERIFIED
    assert receipt.verification["total"] is Verification.VERIFIED
    assert receipt.verification["line_items.0.amount"] is Verification.VERIFIED
    assert describe_failures(receipt) == []


def test_mismatched_line_sum_marks_unverified() -> None:
    receipt = check_totals(_receipt(items=10.0, subtotal=12.0, tax=1.0, total=13.0))
    assert receipt.verification["subtotal"] is Verification.UNVERIFIED
    assert receipt.verification["line_items.0.amount"] is Verification.UNVERIFIED
    assert receipt.subtotal is not None
    assert receipt.subtotal.value == 12.0
    assert "Line items do not add up" in describe_failures(receipt)[0]


def test_mismatched_grand_total_marks_unverified() -> None:
    receipt = check_totals(_receipt(items=10.0, subtotal=10.0, tax=1.0, total=20.0))
    assert receipt.verification["total"] is Verification.UNVERIFIED
    assert receipt.verification["tax"] is Verification.UNVERIFIED
    assert receipt.total is not None
    assert receipt.total.value == 20.0
