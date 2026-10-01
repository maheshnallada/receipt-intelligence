"""Pydantic HTTP contract. Maps to and from the Receipt aggregate."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.receipt import (
    IsoDate,
    LineItem,
    Money,
    Receipt,
    Verification,
)

VerificationStatus = Literal["verified", "unverified", "not_found"]


class LineItemOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    qty: float | None
    unit_price: float | None
    amount: float | None


class ReceiptOut(BaseModel):
    """Assignment JSON. Dates ISO, numbers JSON numbers."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    store_name: str | None
    date: str | None
    line_items: list[LineItemOut]
    subtotal: float | None
    tax: float | None
    total: float | None
    verification: dict[str, VerificationStatus] = Field(alias="_verification")

    @classmethod
    def from_domain(cls, receipt: Receipt) -> ReceiptOut:
        return cls(
            store_name=receipt.store_name,
            date=receipt.date.value if receipt.date else None,
            line_items=[
                LineItemOut(
                    name=item.name,
                    qty=item.qty,
                    unit_price=item.unit_price.value if item.unit_price else None,
                    amount=item.amount.value if item.amount else None,
                )
                for item in receipt.line_items
            ],
            subtotal=receipt.subtotal.value if receipt.subtotal else None,
            tax=receipt.tax.value if receipt.tax else None,
            total=receipt.total.value if receipt.total else None,
            verification={key: value.value for key, value in receipt.verification.items()},
        )

    def to_domain(self) -> Receipt:
        return Receipt(
            store_name=self.store_name,
            date=IsoDate(self.date) if self.date else None,
            line_items=tuple(
                LineItem(
                    name=item.name,
                    qty=item.qty,
                    unit_price=Money(item.unit_price) if item.unit_price is not None else None,
                    amount=Money(item.amount) if item.amount is not None else None,
                )
                for item in self.line_items
            ),
            subtotal=Money(self.subtotal) if self.subtotal is not None else None,
            tax=Money(self.tax) if self.tax is not None else None,
            total=Money(self.total) if self.total is not None else None,
            verification={key: Verification(value) for key, value in self.verification.items()},
        )

    def model_dump_assignment(self) -> dict:
        return self.model_dump(by_alias=True)


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class VersionResponse(BaseModel):
    model_id: str
    adapter_revision: str
    schema_version: str
    backend: str
    prompt_version: str
