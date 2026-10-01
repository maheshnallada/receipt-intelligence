"""Pydantic validation and normalization for raw VLM receipt JSON."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NULL_STRINGS = {"", "null", "none", "unknown", "n/a", "na", "******", "-"}
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")


def _null_placeholder(value: Any) -> Any:
    if isinstance(value, str) and value.strip().casefold() in _NULL_STRINGS:
        return None
    return value


def _number(value: Any) -> Any:
    value = _null_placeholder(value)
    if value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip().replace("Rp", "").replace("rp", "").replace("@", "").replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        text = "".join(parts) if len(parts[-1]) == 3 else text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return value


class StructuredLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    qty: float | None = None
    unit_price: float | None = None
    amount: float | None = None

    @field_validator("qty", "unit_price", "amount", mode="before")
    @classmethod
    def normalize_number(cls, value: Any) -> Any:
        return _number(value)


class StructuredReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store_name: str | None = None
    date: str | None = None
    line_items: list[StructuredLineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None

    @field_validator("store_name", "date", mode="before")
    @classmethod
    def normalize_text_placeholder(cls, value: Any) -> Any:
        return _null_placeholder(value)

    @field_validator("date")
    @classmethod
    def normalize_date(cls, value: str | None) -> str | None:
        """Accept ISO or day-first receipt dates; never guess month-first order."""
        if value is None:
            return None
        for date_format in _DATE_FORMATS:
            try:
                return datetime.strptime(value.strip(), date_format).date().isoformat()
            except ValueError:
                continue
        return None

    @field_validator("subtotal", "tax", "total", mode="before")
    @classmethod
    def normalize_money(cls, value: Any) -> Any:
        return _number(value)
