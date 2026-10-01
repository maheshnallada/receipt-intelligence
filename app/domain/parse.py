"""Stdlib-only parse of model text into a Receipt."""

from __future__ import annotations

import json
import re

from app.domain.receipt import Receipt, ReceiptInvariantError
from app.infrastructure.inference.structured import StructuredReceipt

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> str | None:
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return candidate[start : end + 1]


def parse_model_output(text: str) -> Receipt:
    blob = extract_json_object(text)
    if blob is None:
        raise ReceiptInvariantError("model output is not JSON")
    try:
        loaded = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ReceiptInvariantError("model output is not valid JSON") from exc
    if not isinstance(loaded, dict):
        raise ReceiptInvariantError("model output must be a JSON object")
    try:
        structured = StructuredReceipt.model_validate(loaded)
    except Exception as exc:
        raise ReceiptInvariantError("model output does not match the receipt schema") from exc
    return Receipt.from_payload(structured.model_dump())
