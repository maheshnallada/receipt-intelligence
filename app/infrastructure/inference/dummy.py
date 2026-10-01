"""Synthetic CPU test backend: schema-valid fixtures from the page text layer."""

from __future__ import annotations

import json
import re
from typing import Any

from app.domain.document import Page
from app.domain.errors import InferenceTimeout

_ITEM = re.compile(
    r"ITEM:\s*(?P<name>[^|]+)\|\s*QTY:\s*(?P<qty>[^|]+)\|\s*UNIT:\s*(?P<unit>[^|]+)\|\s*AMT:\s*(?P<amt>.+)",
    re.IGNORECASE,
)
_KV = re.compile(r"^(STORE|DATE|SUBTOTAL|TAX|TOTAL):\s*(.+)$", re.IGNORECASE | re.MULTILINE)


class DummyExtractor:
    def __init__(self, *, timeout_s: float = 120.0) -> None:
        self._timeout_s = timeout_s
        self.ready = True

    def extract_page(self, page: Page, *, repair_hint: str | None = None) -> str:
        if "FORCE_TIMEOUT" in (page.text_layer or ""):
            raise InferenceTimeout("dummy inference timeout")
        payload = self._from_text(page.text_layer)
        if repair_hint:
            return json.dumps(payload)
        if "FORCE_BAD_JSON" in (page.text_layer or ""):
            return "{store_name: missing quotes"
        return json.dumps(payload)

    def _from_text(self, text: str) -> dict[str, Any]:
        fields = {match.group(1).upper(): match.group(2).strip() for match in _KV.finditer(text or "")}
        items = []
        for match in _ITEM.finditer(text or ""):
            items.append(
                {
                    "name": match.group("name").strip(),
                    "qty": _number(match.group("qty")),
                    "unit_price": _number(match.group("unit")),
                    "amount": _number(match.group("amt")),
                }
            )
        store = fields.get("STORE")
        date = fields.get("DATE")
        return {
            "store_name": store or None,
            "date": date or None,
            "line_items": items,
            "subtotal": _number(fields.get("SUBTOTAL")),
            "tax": _number(fields.get("TAX")),
            "total": _number(fields.get("TOTAL")),
        }


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    cleaned = raw.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None
