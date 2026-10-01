"""Receipt aggregate and HTTP schema mapping."""

from __future__ import annotations

import pytest

from app.domain.parse import parse_model_output
from app.domain.receipt import (
    IsoDate,
    LineItem,
    Money,
    Receipt,
    ReceiptInvariantError,
    Verification,
    default_verification,
    empty_receipt,
)


def _valid_receipt() -> Receipt:
    return Receipt(
        store_name="Oak & Ember",
        date=IsoDate("2026-03-15"),
        line_items=(
            LineItem(name="Latte", qty=2, unit_price=Money(4.5), amount=Money(9.0)),
        ),
        subtotal=Money(9.0),
        tax=Money(0.72),
        total=Money(9.72),
        verification=default_verification(1),
    )


def test_receipt_accepts_valid_example() -> None:
    receipt = _valid_receipt()
    assert receipt.store_name == "Oak & Ember"
    assert receipt.date is not None
    assert receipt.date.value == "2026-03-15"
    assert receipt.total is not None
    assert receipt.total.value == 9.72


def test_receipt_rejects_bad_date() -> None:
    with pytest.raises(ReceiptInvariantError):
        Receipt(
            store_name="Oak",
            date=IsoDate("15-03-2026"),
            line_items=(),
            subtotal=None,
            tax=None,
            total=None,
            verification=default_verification(0),
        )


@pytest.mark.parametrize("raw, expected", [
    ("04/12/2018", "2018-12-04"),
    ("4-12-2018", "2018-12-04"),
    ("04.12.2018", "2018-12-04"),
    ("2018-12-04", "2018-12-04"),
    ("12/04/2018", "2018-04-12"),
    ("12/31/2018", None),
    ("31/02/2018", None),
])
def test_model_dates_normalize_day_first_without_guessing(raw, expected) -> None:
    receipt = parse_model_output('{"date": "' + raw + '"}')
    assert (receipt.date.value if receipt.date else None) == expected


def test_receipt_rejects_wrong_money_type() -> None:
    with pytest.raises(ReceiptInvariantError):
        Money("9.72")  # type: ignore[arg-type]


def test_receipt_rejects_missing_verification_leaf() -> None:
    with pytest.raises(ReceiptInvariantError, match="missing verification"):
        Receipt(
            store_name="Oak",
            date=None,
            line_items=(),
            subtotal=None,
            tax=None,
            total=Money(1.0),
            verification={"store_name": Verification.VERIFIED},
        )


def test_from_payload_and_empty() -> None:
    receipt = Receipt.from_payload(
        {
            "store_name": "Oak",
            "date": "2026-03-15",
            "line_items": [{"name": "Tea", "qty": 1, "unit_price": 2, "amount": 2}],
            "subtotal": 2,
            "tax": 0,
            "total": 2,
        }
    )
    assert receipt.verification["store_name"] is Verification.UNVERIFIED
    assert empty_receipt().store_name is None
    assert empty_receipt().verification["total"] is Verification.NOT_FOUND


def test_http_schema_round_trip() -> None:
    from app.interfaces.http.schemas import ReceiptOut

    receipt = _valid_receipt()
    payload = ReceiptOut.from_domain(receipt)
    data = payload.model_dump(by_alias=True)
    assert data["date"] == "2026-03-15"
    assert data["line_items"][0]["unit_price"] == 4.5
    assert data["_verification"]["store_name"] == "unverified"
    rebuilt = payload.to_domain()
    assert rebuilt.store_name == receipt.store_name
    assert rebuilt.total is not None
    assert rebuilt.total.value == receipt.total.value
