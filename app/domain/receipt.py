"""Receipt aggregate, line items, and value objects.

Constructing a Receipt enforces the assignment schema: ISO dates, numeric
money, and a verification entry for every leaf field.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any

_NULL_STRINGS = {"", "null", "none", "unknown", "n/a", "na", "******", "-"}


class Verification(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    NOT_FOUND = "not_found"


class ReceiptInvariantError(ValueError):
    """Raised when a Receipt cannot be constructed."""


@dataclass(frozen=True)
class Money:
    """Numeric money. JSON number, never a string."""

    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ReceiptInvariantError("money must be a JSON number")
        object.__setattr__(self, "value", float(self.value))

    def __float__(self) -> float:
        return self.value


@dataclass(frozen=True)
class IsoDate:
    """Calendar date stored as YYYY-MM-DD."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise ReceiptInvariantError("date must be an ISO string")
        try:
            parsed = datetime.strptime(self.value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ReceiptInvariantError("date must be ISO YYYY-MM-DD") from exc
        object.__setattr__(self, "value", parsed.isoformat())

    def as_date(self) -> date:
        return date.fromisoformat(self.value)


@dataclass(frozen=True)
class LineItem:
    """Entity inside the Receipt aggregate."""

    name: str
    qty: float | None
    unit_price: Money | None
    amount: Money | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ReceiptInvariantError("line item name must be a non-empty string")
        if self.qty is not None:
            if isinstance(self.qty, bool) or not isinstance(self.qty, (int, float)):
                raise ReceiptInvariantError("qty must be a number or null")
            object.__setattr__(self, "qty", float(self.qty))


def verification_paths(line_item_count: int) -> tuple[str, ...]:
    paths = ["store_name", "date"]
    for index in range(line_item_count):
        paths.extend(
            (
                f"line_items.{index}.name",
                f"line_items.{index}.qty",
                f"line_items.{index}.unit_price",
                f"line_items.{index}.amount",
            )
        )
    paths.extend(("subtotal", "tax", "total"))
    return tuple(paths)


def default_verification(line_item_count: int, *, for_nulls: bool = False) -> dict[str, Verification]:
    status = Verification.NOT_FOUND if for_nulls else Verification.UNVERIFIED
    return {path: status for path in verification_paths(line_item_count)}


@dataclass(frozen=True)
class Receipt:
    """Aggregate root for one extracted (and optionally merged) receipt."""

    store_name: str | None
    date: IsoDate | None
    line_items: tuple[LineItem, ...]
    subtotal: Money | None
    tax: Money | None
    total: Money | None
    verification: Mapping[str, Verification] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.store_name is not None and not isinstance(self.store_name, str):
            raise ReceiptInvariantError("store_name must be a string or null")
        if self.store_name is not None and not self.store_name.strip():
            raise ReceiptInvariantError("store_name must be non-empty when set")
        items = tuple(self.line_items)
        object.__setattr__(self, "line_items", items)
        required = set(verification_paths(len(items)))
        present = set(self.verification)
        missing = required - present
        extra = present - required
        if missing:
            raise ReceiptInvariantError(f"missing verification leaves: {sorted(missing)}")
        if extra:
            raise ReceiptInvariantError(f"unknown verification leaves: {sorted(extra)}")
        frozen = {key: Verification(value) for key, value in self.verification.items()}
        object.__setattr__(self, "verification", frozen)

    def with_verification(self, verification: Mapping[str, Verification]) -> Receipt:
        return Receipt(
            store_name=self.store_name,
            date=self.date,
            line_items=self.line_items,
            subtotal=self.subtotal,
            tax=self.tax,
            total=self.total,
            verification=verification,
        )

    def replace(self, **changes: Any) -> Receipt:
        data = {
            "store_name": self.store_name,
            "date": self.date,
            "line_items": self.line_items,
            "subtotal": self.subtotal,
            "tax": self.tax,
            "total": self.total,
            "verification": self.verification,
        }
        data.update(changes)
        return Receipt(**data)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Receipt:
        """Build a Receipt from model JSON. Nulls → not_found, values → unverified."""
        if not isinstance(payload, Mapping):
            raise ReceiptInvariantError("payload must be an object")
        raw_items = payload.get("line_items") or []
        if not isinstance(raw_items, list):
            raise ReceiptInvariantError("line_items must be an array")
        items = tuple(_line_item_from_payload(item) for item in raw_items)
        store = _null_placeholder(payload.get("store_name"))
        if store is not None and not isinstance(store, str):
            raise ReceiptInvariantError("store_name must be a string or null")
        raw_date = _null_placeholder(payload.get("date"))
        date_value = IsoDate(raw_date) if raw_date is not None else None
        receipt = cls(
            store_name=store.strip() if isinstance(store, str) and store.strip() else None,
            date=date_value,
            line_items=items,
            subtotal=_optional_money(payload.get("subtotal"), "subtotal"),
            tax=_optional_money(payload.get("tax"), "tax"),
            total=_optional_money(payload.get("total"), "total"),
            verification=_verification_from_values(
                store_name=store.strip() if isinstance(store, str) and store.strip() else None,
                date_value=date_value,
                items=items,
                subtotal=_optional_money(payload.get("subtotal"), "subtotal"),
                tax=_optional_money(payload.get("tax"), "tax"),
                total=_optional_money(payload.get("total"), "total"),
            ),
        )
        return receipt


def empty_receipt() -> Receipt:
    """Schema-valid all-null receipt used when the model cannot be repaired."""
    return Receipt(
        store_name=None,
        date=None,
        line_items=(),
        subtotal=None,
        tax=None,
        total=None,
        verification=default_verification(0, for_nulls=True),
    )


def _optional_money(value: Any, field_name: str) -> Money | None:
    if value is None:
        return None
    value = _null_placeholder(value)
    if value is None:
        return None
    if isinstance(value, Money):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptInvariantError(f"{field_name} must be a number or null")
    return Money(float(value))


def _null_placeholder(value: Any) -> Any:
    if isinstance(value, str) and value.strip().casefold() in _NULL_STRINGS:
        return None
    return value


def _line_item_from_payload(item: Any) -> LineItem:
    if not isinstance(item, Mapping):
        raise ReceiptInvariantError("line item must be an object")
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ReceiptInvariantError("line item name must be a non-empty string")
    qty = item.get("qty")
    if qty is not None and (isinstance(qty, bool) or not isinstance(qty, (int, float))):
        raise ReceiptInvariantError("qty must be a number or null")
    unit_price = _optional_money(item.get("unit_price"), "unit_price")
    amount = _optional_money(item.get("amount"), "amount")
    if unit_price is None and amount is not None and qty not in (None, 0):
        unit_price = Money(amount.value / float(qty))
    return LineItem(
        name=name.strip(),
        qty=float(qty) if qty is not None else None,
        unit_price=unit_price,
        amount=amount,
    )


def _verification_from_values(
    *,
    store_name: str | None,
    date_value: IsoDate | None,
    items: tuple[LineItem, ...],
    subtotal: Money | None,
    tax: Money | None,
    total: Money | None,
) -> dict[str, Verification]:
    verification: dict[str, Verification] = {
        "store_name": Verification.UNVERIFIED if store_name else Verification.NOT_FOUND,
        "date": Verification.UNVERIFIED if date_value else Verification.NOT_FOUND,
        "subtotal": Verification.UNVERIFIED if subtotal else Verification.NOT_FOUND,
        "tax": Verification.UNVERIFIED if tax else Verification.NOT_FOUND,
        "total": Verification.UNVERIFIED if total else Verification.NOT_FOUND,
    }
    for index, item in enumerate(items):
        verification[f"line_items.{index}.name"] = Verification.UNVERIFIED
        verification[f"line_items.{index}.qty"] = (
            Verification.UNVERIFIED if item.qty is not None else Verification.NOT_FOUND
        )
        verification[f"line_items.{index}.unit_price"] = (
            Verification.UNVERIFIED if item.unit_price is not None else Verification.NOT_FOUND
        )
        verification[f"line_items.{index}.amount"] = (
            Verification.UNVERIFIED if item.amount is not None else Verification.NOT_FOUND
        )
    return verification
