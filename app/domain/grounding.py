"""Pure grounding: every non-null leaf must appear in the document text.

Null-out policy (documented next to the rule it implements):

- A predicted value found in normalized document text → keep, verified.
- A predicted value absent from text layer and OCR → null, not_found.
  The raw candidate is not part of the returned Receipt.
- Document text is empty (no text layer and OCR produced nothing) → null,
  unverified. Returning the model value would be a guess.
- Null predictions stay null with not_found.

Numeric match uses the same formatting-insensitive comparison as the tests
(1,200.50 vs 1200.5). Currency symbols and thousands separators are stripped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.domain.receipt import IsoDate, LineItem, Money, Receipt, Verification

_CURRENCY = re.compile(
    r"(?:[$€£¥₹]|rp\.?|idr|usd|eur|gbp|sgd|myr)\s*",
    flags=re.IGNORECASE,
)
_SPACES = re.compile(r"\s+")
SANITY_TOLERANCE = 0.05


def normalize_text(value: str) -> str:
    text = value.casefold()
    text = _CURRENCY.sub("", text)
    text = text.replace("\u00a0", " ")
    text = _SPACES.sub(" ", text).strip()
    return text


def normalize_number_token(value: str) -> str:
    """Strip currency and thousand separators; keep a single decimal point."""
    text = normalize_text(value)
    text = text.replace(" ", "")
    if not text:
        return ""
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        if len(parts[-1]) == 3 and parts[0].lstrip("-").isdigit():
            text = "".join(parts)
        else:
            text = text.replace(",", ".")
    return text


def parse_number(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    token = normalize_number_token(value)
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def numbers_match(predicted: float, document_text: str, *, tolerance: float = SANITY_TOLERANCE) -> bool:
    haystack = normalize_text(document_text)
    haystack = re.sub(r"(?<=\d)\s*([.,])\s*(?=\d)", r"\1", haystack)
    variants = set(_numeric_variants(predicted))
    tokens = re.findall(r"(?<![\w.,+:/%-])[+-]?\d+(?:[.,]\d+)*(?![\w.,+:/%-])(?!\s*%)", haystack)
    for token in tokens:
        if token in variants:
            return True
        parsed = parse_number(token)
        if parsed is not None and (parsed == predicted or predicted != 0 and abs(parsed - predicted) <= tolerance):
            return True
    return False


def text_present(predicted: str, document_text: str) -> bool:
    needle = normalize_text(predicted)
    if not needle:
        return False
    haystack = normalize_text(document_text)
    if needle in haystack:
        return True
    compact_needle = needle.replace(" ", "")
    compact_hay = haystack.replace(" ", "")
    return bool(compact_needle) and compact_needle in compact_hay


def date_present(iso: IsoDate, document_text: str) -> bool:
    raw = iso.value
    year, month, day = raw.split("-")
    month_i, day_i = int(month), int(day)
    candidates = {
        raw,
        f"{day}/{month}/{year}",
        f"{day}-{month}-{year}",
        f"{day}.{month}.{year}",
        f"{day_i}/{month_i}/{year}",
        f"{day_i}-{month_i}-{year}",
        f"{day_i}.{month_i}.{year}",
        f"{day_i} {month_i} {year}",
    }
    haystack = normalize_text(document_text)
    return any(normalize_text(candidate) in haystack for candidate in candidates)


def ground_receipt(receipt: Receipt, document_text: str) -> Receipt:
    """Return a new Receipt with grounded values and verification."""
    text_empty = not normalize_text(document_text)
    line_item_names = {normalize_text(item.name) for item in receipt.line_items}
    store_candidate = receipt.store_name
    normalized_store = normalize_text(store_candidate) if store_candidate else ""
    if normalized_store in line_item_names or normalized_store.replace(" ", "").isdigit():
        store_candidate = None
    store_name, store_status = _ground_text(store_candidate, document_text, text_empty)
    date_value, date_status = _ground_date(receipt.date, document_text, text_empty)
    items: list[LineItem] = []
    verification: dict[str, Verification] = {
        "store_name": store_status,
        "date": date_status,
    }
    for index, item in enumerate(receipt.line_items):
        name, name_status = _ground_text(item.name, document_text, text_empty)
        if name is None:
            # Name is required on a kept line. Drop the row if we cannot ground it.
            continue
        qty, qty_status = _ground_number(item.qty, document_text, text_empty)
        unit, unit_status = _ground_money(item.unit_price, document_text, text_empty)
        amount, amount_status = _ground_money(item.amount, document_text, text_empty)
        new_index = len(items)
        items.append(LineItem(name=name, qty=qty, unit_price=unit, amount=amount))
        verification[f"line_items.{new_index}.name"] = name_status
        verification[f"line_items.{new_index}.qty"] = qty_status
        verification[f"line_items.{new_index}.unit_price"] = unit_status
        verification[f"line_items.{new_index}.amount"] = amount_status
        for field in ("qty", "unit_price", "amount"):
            if getattr(item, field) is None and receipt.verification.get(f"line_items.{index}.{field}") == Verification.UNVERIFIED:
                verification[f"line_items.{new_index}.{field}"] = Verification.UNVERIFIED
    subtotal, sub_status = _ground_money(receipt.subtotal, document_text, text_empty)
    tax, tax_status = _ground_money(receipt.tax, document_text, text_empty)
    total, total_status = _ground_money(receipt.total, document_text, text_empty)
    verification["subtotal"] = sub_status
    verification["tax"] = tax_status
    verification["total"] = total_status
    for field in ("store_name", "date", "subtotal", "tax", "total"):
        if getattr(receipt, field) is None and receipt.verification.get(field) == Verification.UNVERIFIED:
            verification[field] = Verification.UNVERIFIED
    return Receipt(
        store_name=store_name,
        date=date_value,
        line_items=tuple(items),
        subtotal=subtotal,
        tax=tax,
        total=total,
        verification=verification,
    )


def _ground_text(
    value: str | None, document_text: str, text_empty: bool
) -> tuple[str | None, Verification]:
    if value is None:
        return None, Verification.NOT_FOUND
    if text_empty:
        return None, Verification.UNVERIFIED
    if text_present(value, document_text):
        return value, Verification.VERIFIED
    return None, Verification.NOT_FOUND


def _ground_date(
    value: IsoDate | None, document_text: str, text_empty: bool
) -> tuple[IsoDate | None, Verification]:
    if value is None:
        return None, Verification.NOT_FOUND
    if text_empty:
        return None, Verification.UNVERIFIED
    if date_present(value, document_text):
        return value, Verification.VERIFIED
    return None, Verification.NOT_FOUND


def _ground_number(
    value: float | None, document_text: str, text_empty: bool
) -> tuple[float | None, Verification]:
    if value is None:
        return None, Verification.NOT_FOUND
    if text_empty:
        return None, Verification.UNVERIFIED
    if numbers_match(value, document_text):
        return value, Verification.VERIFIED
    return None, Verification.NOT_FOUND


def _ground_money(
    value: Money | None, document_text: str, text_empty: bool
) -> tuple[Money | None, Verification]:
    if value is None:
        return None, Verification.NOT_FOUND
    if text_empty:
        return None, Verification.UNVERIFIED
    if numbers_match(value.value, document_text):
        return value, Verification.VERIFIED
    scaled = value.value * 1000.0
    if abs(value.value) < 1000 and numbers_match(scaled, document_text):
        return Money(scaled), Verification.VERIFIED
    return None, Verification.NOT_FOUND


def _numeric_variants(value: float) -> Iterable[str]:
    as_int = abs(value - round(value)) < 1e-9
    if as_int:
        whole = str(int(round(value)))
        grouped = f"{int(round(value)):,}"
        yield whole
        yield grouped
        yield grouped.replace(",", ".")
    else:
        two = f"{value:.2f}"
        stripped = two.rstrip("0").rstrip(".")
        yield two
        yield stripped
        yield two.replace(".", ",")
        integer, frac = two.split(".")
        yield f"{int(integer):,}.{frac}"
        yield f"{int(integer):,}".replace(",", ".") + f",{frac}"
