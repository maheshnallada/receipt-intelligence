"""Field F1, exact match, JSON validity, hallucination, latency."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from app.domain.grounding import normalize_text, parse_number


def _norm_name(name: str) -> str:
    return normalize_text(name)


def match_line_items(gold: list[dict], pred: list[dict]) -> list[tuple[dict | None, dict | None]]:
    remaining = list(pred)
    pairs: list[tuple[dict | None, dict | None]] = []
    for item in gold:
        hit = None
        for index, candidate in enumerate(remaining):
            if _norm_name(candidate.get("name") or "") == _norm_name(item.get("name") or ""):
                hit = remaining.pop(index)
                break
        pairs.append((item, hit))
    for extra in remaining:
        pairs.append((None, extra))
    return pairs


def field_scores(golds: list[dict], preds: list[dict], field: str) -> dict[str, float]:
    tp = fp = fn = exact = 0
    for gold, pred in zip(golds, preds, strict=True):
        gv, pv = gold.get(field), pred.get(field)
        if field == "line_items":
            continue
        if gv is None and pv is None:
            exact += 1
            continue
        if gv is None and pv is not None:
            fp += 1
            continue
        if gv is not None and pv is None:
            fn += 1
            continue
        if _values_equal(gv, pv):
            tp += 1
            exact += 1
        else:
            fp += 1
            fn += 1
    return _prf(tp, fp, fn, exact, n=len(golds))


def line_item_scores(golds: list[dict], preds: list[dict]) -> dict[str, Any]:
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "n": 0})
    for gold, pred in zip(golds, preds, strict=True):
        pairs = match_line_items(gold.get("line_items") or [], pred.get("line_items") or [])
        for g_item, p_item in pairs:
            for leaf in ("name", "qty", "unit_price", "amount"):
                bucket = totals[leaf]
                bucket["n"] += 1
                gv = None if g_item is None else g_item.get(leaf)
                pv = None if p_item is None else p_item.get(leaf)
                if gv is None and pv is None:
                    bucket["exact"] += 1
                elif gv is None and pv is not None:
                    bucket["fp"] += 1
                elif gv is not None and pv is None:
                    bucket["fn"] += 1
                elif _values_equal(gv, pv):
                    bucket["tp"] += 1
                    bucket["exact"] += 1
                else:
                    bucket["fp"] += 1
                    bucket["fn"] += 1
    return {
        leaf: _prf(vals["tp"], vals["fp"], vals["fn"], vals["exact"], vals["n"])
        for leaf, vals in totals.items()
    }


def exact_receipt(gold: dict, pred: dict) -> bool:
    for field in ("store_name", "date", "subtotal", "tax", "total"):
        if not _values_equal(gold.get(field), pred.get(field)):
            return False
    gold_items = sorted(gold.get("line_items") or [], key=lambda item: _norm_name(item.get("name") or ""))
    pred_items = sorted(pred.get("line_items") or [], key=lambda item: _norm_name(item.get("name") or ""))
    if len(gold_items) != len(pred_items):
        return False
    return all(
        _values_equal(left.get(key), right.get(key))
        for left, right in zip(gold_items, pred_items, strict=True)
        for key in ("name", "qty", "unit_price", "amount")
    )


def hallucination_rate(preds: list[dict], texts: list[str], *, after_grounding: bool) -> float:
    del after_grounding
    non_null = 0
    missing = 0
    for pred, text in zip(preds, texts, strict=True):
        values: list[Any] = [pred.get("store_name"), pred.get("date"), pred.get("subtotal"), pred.get("tax"), pred.get("total")]
        for item in pred.get("line_items") or []:
            values.extend([item.get("name"), item.get("qty"), item.get("unit_price"), item.get("amount")])
        for value in values:
            if value is None:
                continue
            non_null += 1
            if isinstance(value, str) and normalize_text(value) not in normalize_text(text):
                missing += 1
            elif isinstance(value, (int, float)) and parse_number(str(value)) is not None:
                from app.domain.grounding import numbers_match

                if not numbers_match(float(value), text):
                    missing += 1
    return 0.0 if non_null == 0 else missing / non_null


def latency_percentiles(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {"p50": math.nan, "p95": math.nan}
    ordered = sorted(samples_ms)
    def pct(p: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
        return ordered[index]
    return {"p50": pct(50), "p95": pct(95)}


def _values_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, str) or isinstance(right, str):
        return normalize_text(str(left)) == normalize_text(str(right))
    try:
        return abs(float(left) - float(right)) <= 0.05
    except (TypeError, ValueError):
        return left == right


def _prf(tp: int, fp: int, fn: int, exact: int, n: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact / n if n else 0.0,
    }
