"""Pure multi-page merge.

Rule (documented here on purpose):

- Concatenate line items in page order.
- Score headers (store_name, date) and money (subtotal, tax, total)
  separately as the count of verified leaves in that group.
- store_name and date come from the best header page; ties prefer the
  earlier page (header is printed at the top).
- subtotal, tax, and total come from the best money page; ties prefer
  the later page (totals are printed at the end).
- Verification for copied scalars follows the winning page. Line-item
  verification is rewritten for the concatenated indices.
"""

from __future__ import annotations

from app.domain.receipt import (
    LineItem,
    Receipt,
    Verification,
    default_verification,
    verification_paths,
)

_HEADER = ("store_name", "date")
_MONEY = ("subtotal", "tax", "total")
_SCORE_FIELDS = _HEADER + _MONEY


def page_score(receipt: Receipt, fields: tuple[str, ...] = _SCORE_FIELDS) -> int:
    return sum(1 for field in fields if receipt.verification.get(field) == Verification.VERIFIED)


def merge_pages(pages: list[Receipt]) -> Receipt:
    if not pages:
        return Receipt(
            store_name=None,
            date=None,
            line_items=(),
            subtotal=None,
            tax=None,
            total=None,
            verification=default_verification(0, for_nulls=True),
        )
    if len(pages) == 1:
        return pages[0]

    items: list[LineItem] = []
    item_verification: dict[str, Verification] = {}
    for page in pages:
        offset = len(items)
        items.extend(page.line_items)
        for index, _item in enumerate(page.line_items):
            for leaf in ("name", "qty", "unit_price", "amount"):
                src = f"line_items.{index}.{leaf}"
                dst = f"line_items.{offset + index}.{leaf}"
                item_verification[dst] = page.verification[src]

    header_winner = _pick(pages, fields=_HEADER, prefer_last=False)
    money_winner = _pick(pages, fields=_MONEY, prefer_last=True)

    verification: dict[str, Verification] = {**item_verification}
    for field in _HEADER:
        verification[field] = header_winner.verification[field]
    for field in _MONEY:
        verification[field] = money_winner.verification[field]

    expected = set(verification_paths(len(items)))
    for path in expected:
        verification.setdefault(path, Verification.NOT_FOUND)

    return Receipt(
        store_name=header_winner.store_name,
        date=header_winner.date,
        line_items=tuple(items),
        subtotal=money_winner.subtotal,
        tax=money_winner.tax,
        total=money_winner.total,
        verification=verification,
    )


def _pick(pages: list[Receipt], *, fields: tuple[str, ...], prefer_last: bool) -> Receipt:
    scored = [(page_score(page, fields), index, page) for index, page in enumerate(pages)]
    if prefer_last:
        return max(scored, key=lambda row: (row[0], row[1]))[2]
    return max(scored, key=lambda row: (row[0], -row[1]))[2]
