"""CORD v2 → receipt schema chat samples. Official split ids are frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

QTY_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def parse_qty(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        values = [parse_money(value) for value in raw]
        values = [value for value in values if value is not None]
        if not values or any(abs(value - values[0]) > 1e-9 for value in values[1:]):
            return None
        return values[0]
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    match = QTY_RE.search(str(raw))
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def parse_money(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        values = [parse_money(value) for value in raw]
        values = [value for value in values if value is not None]
        if not values or any(abs(value - values[0]) > 1e-9 for value in values[1:]):
            return None
        return values[0]
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    text = str(raw).strip().replace("Rp", "").replace("rp", "").replace("@", "").replace(" ", "")
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        text = "".join(parts) if len(parts[-1]) == 3 else text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def as_menu_list(menu: Any) -> list[dict[str, Any]]:
    if menu is None:
        return []
    if isinstance(menu, dict):
        return [menu]
    if isinstance(menu, list):
        return [item for item in menu if isinstance(item, dict)]
    return []


def flatten_menu_items(menu: Any) -> list[dict[str, Any]]:
    """Return parent menu items and nested sub-items in source order."""
    flattened = []
    for item in as_menu_list(menu):
        flattened.append(item)
        flattened.extend(flatten_menu_items(item.get("sub")))
    return flattened


def cord_to_schema(gt_parse: dict[str, Any]) -> dict[str, Any]:
    """Map CORD gt_parse. store_name and date are unlabeled in CORD → null."""
    items = []
    for item in flatten_menu_items(gt_parse.get("menu")):
        name = item.get("nm")
        if not isinstance(name, str) or not name.strip():
            continue
        qty = parse_qty(item.get("cnt"))
        unit_price = parse_money(item.get("unitprice"))
        amount = parse_money(item.get("price"))
        if unit_price is None and amount is not None and qty not in (None, 0):
            unit_price = amount / qty
        items.append({"name": name.strip(), "qty": qty, "unit_price": unit_price, "amount": amount})
    subtotal = gt_parse.get("sub_total") or {}
    total = gt_parse.get("total") or {}
    return {
        "store_name": None,
        "date": None,
        "line_items": items,
        "subtotal": parse_money(subtotal.get("subtotal_price")),
        "tax": parse_money(subtotal.get("tax_price")),
        "total": parse_money(total.get("total_price")),
    }


def ground_truth_of(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["ground_truth"]
    payload = json.loads(raw) if isinstance(raw, str) else raw
    return payload["gt_parse"]


def split_hash(ids: list[str]) -> str:
    return hashlib.sha256(",".join(ids).encode()).hexdigest()[:16]


def freeze_splits(dataset, out: Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    for name in dataset:
        ids = []
        for index, row in enumerate(dataset[name]):
            gt = json.loads(row["ground_truth"]) if isinstance(row["ground_truth"], str) else row["ground_truth"]
            meta = gt.get("meta") or {}
            ids.append(str(meta.get("image_id", f"{name}-{index}")))
        splits[name] = ids
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"splits": splits, "hash": {k: split_hash(v) for k, v in splits.items()}}, indent=2), encoding="utf-8")
    return splits


def to_messages(target: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = (
        "Extract the receipt as JSON with keys store_name, date (YYYY-MM-DD), "
        "line_items (name, qty, unit_price, amount), subtotal, tax, total. "
        "Copy only visible text. Use null when absent. Numbers must be JSON numbers."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target, ensure_ascii=False)}]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/splits.json")
    args = parser.parse_args()
    from datasets import load_dataset

    dataset = load_dataset("naver-clova-ix/cord-v2")
    freeze_splits(dataset, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
