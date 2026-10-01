from app.domain.merge import merge_pages, page_score
from app.domain.receipt import (
    IsoDate,
    LineItem,
    Money,
    Receipt,
    Verification,
    default_verification,
)


def _page(
    *,
    store: str | None,
    date: str | None,
    name: str,
    total: float | None,
    verified: tuple[str, ...],
) -> Receipt:
    verification = default_verification(1)
    for key in verified:
        verification[key] = Verification.VERIFIED
    return Receipt(
        store_name=store,
        date=IsoDate(date) if date else None,
        line_items=(LineItem(name=name, qty=1, unit_price=Money(1.0), amount=Money(1.0)),),
        subtotal=Money(1.0) if total is not None else None,
        tax=Money(0.0) if total is not None else None,
        total=Money(total) if total is not None else None,
        verification=verification,
    )


def test_merge_concatenates_line_items() -> None:
    first = _page(store="Oak", date="2026-01-01", name="Latte", total=3.0, verified=("store_name", "date"))
    second = _page(store="Wrong", date=None, name="Tea", total=10.0, verified=("total", "subtotal", "tax"))
    merged = merge_pages([first, second])
    assert [item.name for item in merged.line_items] == ["Latte", "Tea"]
    assert merged.store_name == "Oak"
    assert merged.date is not None
    assert merged.date.value == "2026-01-01"
    assert merged.total is not None
    assert merged.total.value == 10.0
    assert page_score(second) > page_score(first)
