import pytest

from app.domain.grounding import date_present, ground_receipt, numbers_match, text_present
from app.domain.receipt import (
    IsoDate,
    LineItem,
    Money,
    Receipt,
    Verification,
    default_verification,
)


def _draft(**overrides: object) -> Receipt:
    data: dict = {
        "store_name": "Oak & Ember",
        "date": IsoDate("2026-03-15"),
        "line_items": (LineItem(name="Latte", qty=2, unit_price=Money(4.5), amount=Money(9.0)),),
        "subtotal": Money(12.0),
        "tax": Money(1.0),
        "total": Money(13.0),
        "verification": default_verification(1),
    }
    data.update(overrides)
    return Receipt(**data)


def test_value_present_is_verified() -> None:
    text = "Oak & Ember  15/03/2026  Latte  2  4.50  9.00  SUBTOTAL 12.00 TAX 1.00 TOTAL 13.00"
    grounded = ground_receipt(_draft(), text)
    assert grounded.store_name == "Oak & Ember"
    assert grounded.verification["store_name"] is Verification.VERIFIED
    assert grounded.verification["date"] is Verification.VERIFIED
    assert grounded.verification["total"] is Verification.VERIFIED
    assert grounded.verification["line_items.0.name"] is Verification.VERIFIED


def test_value_absent_is_nulled() -> None:
    text = "Latte 9.00 TOTAL 13.00"
    grounded = ground_receipt(_draft(), text)
    assert grounded.store_name is None
    assert grounded.verification["store_name"] is Verification.NOT_FOUND
    assert grounded.date is None
    assert grounded.verification["date"] is Verification.NOT_FOUND


def test_line_item_name_is_not_accepted_as_store_name() -> None:
    draft = _draft(
        store_name="TRAD KY TOAST CARTE",
        line_items=(LineItem(name="TRAD KY TOAST CARTE", qty=None, unit_price=None, amount=Money(28.182)),),
    )
    grounded = ground_receipt(draft, "TRAD KY TOAST CARTE 28.182 TOTAL 31.00")

    assert grounded.store_name is None
    assert grounded.verification["store_name"] is Verification.NOT_FOUND


def test_numeric_store_artifact_is_not_accepted() -> None:
    draft = _draft(store_name="0")
    grounded = ground_receipt(draft, "0 COLD OCHA 10.00 TOTAL 10.00")

    assert grounded.store_name is None
    assert grounded.verification["store_name"] is Verification.NOT_FOUND


def test_numeric_formatting_differences() -> None:
    assert numbers_match(1200.5, "Total 1,200.50")
    assert numbers_match(1200.5, "1200.5")
    assert numbers_match(75000, "75,000")
    assert numbers_match(12.0, "TOTAL 12.00")
    assert not numbers_match(12.0, "unrelated amount 112.00")
    assert not numbers_match(12.0, "no money here")
    assert text_present("Oak & Ember", "welcome to OAK & EMBER cafe")


@pytest.mark.parametrize("text", ["10. 00", "70 . 00", "460, 00", "REF X00Y", "ID 120-000", "RATE 0%"])
def test_numeric_fragments_do_not_verify_zero(text) -> None:
    assert not numbers_match(0, text)


@pytest.mark.parametrize("text", ["TOTAL 10. 00", "TOTAL 10 . 00", "TOTAL 10, 00"])
def test_split_decimal_is_one_amount(text) -> None:
    assert numbers_match(10, text)
    assert not numbers_match(0, text)


def test_explicit_zero_is_still_supported() -> None:
    assert numbers_match(0, "TAX 0.00")
    assert numbers_match(0, "TAX 0,00")
    assert numbers_match(0, "TAX 0")
    assert not numbers_match(0, "TAX 0.04")
    assert numbers_match(12, "TOTAL 12.03")


def test_date_grounding_uses_day_first_policy() -> None:
    date = IsoDate("2018-12-04")
    assert date_present(date, "DATE 04/12/2018")
    assert not date_present(date, "DATE 12/04/2018")


def test_empty_document_text_nulls_as_unverified() -> None:
    grounded = ground_receipt(_draft(), "   ")
    assert grounded.store_name is None
    assert grounded.verification["store_name"] is Verification.UNVERIFIED
    assert grounded.total is None
    assert grounded.verification["total"] is Verification.UNVERIFIED


def test_regrounding_preserves_missing_evidence_reason() -> None:
    grounded = ground_receipt(_draft(), "")
    repeated = ground_receipt(grounded, "")
    assert repeated == grounded


def test_regrounding_preserves_unknown_line_item_numbers() -> None:
    grounded = ground_receipt(_draft(), "Oak & Ember Latte 9.00")
    verification = dict(grounded.verification)
    verification["line_items.0.qty"] = Verification.UNVERIFIED
    grounded = grounded.with_verification(verification)
    repeated = ground_receipt(grounded, "Oak & Ember Latte 9.00")
    assert repeated == grounded
