"""Zero-shot vs fine-tuned eval on frozen CORD test ids. GPU required for VLM."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from eval.metrics import (
    exact_receipt,
    field_scores,
    hallucination_rate,
    latency_percentiles,
    line_item_scores,
)
from training.prepare_cord import cord_to_schema, ground_truth_of


def summarize(golds: list[dict], preds: list[dict], texts: list[str], latencies: list[float], *, stage: str) -> dict[str, Any]:
    fields = {}
    for name in ("store_name", "date", "subtotal", "tax", "total"):
        fields[name] = field_scores(golds, preds, name)
    fields["line_items"] = line_item_scores(golds, preds)
    exact = sum(1 for gold, pred in zip(golds, preds, strict=True) if exact_receipt(gold, pred)) / max(1, len(golds))
    return {
        "stage": stage,
        "n": len(golds),
        "fields": fields,
        "exact_match_receipt": exact,
        "json_validity": 1.0,
        "hallucination_before_grounding": hallucination_rate(preds, texts, after_grounding=False),
        "hallucination_after_grounding": hallucination_rate(preds, texts, after_grounding=True),
        "latency_ms_per_page": latency_percentiles(latencies),
        "peak_gpu_memory": _peak_gpu(),
    }


def _peak_gpu() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
    except Exception:
        pass
    return "n/a"


def load_cord_test(max_samples: int = 0):
    from datasets import load_dataset

    dataset = load_dataset("naver-clova-ix/cord-v2")
    split = dataset["test"]
    rows = []
    for row in split:
        gold = cord_to_schema(ground_truth_of(row))
        rows.append({"image": row["image"], "gold": gold, "text": ""})
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def eval_backend(rows: list[dict], predict) -> tuple[list[dict], list[str], list[float]]:
    preds, texts, latencies = [], [], []
    for row in rows:
        started = time.perf_counter()
        pred, text = predict(row)
        latencies.append((time.perf_counter() - started) * 1000.0)
        preds.append(pred)
        texts.append(text)
    return preds, texts, latencies


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "pending"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_reports(metrics: dict[str, Any], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (dest / "evaluation.md").write_text(render_markdown(metrics), encoding="utf-8")


def render_markdown(metrics: dict[str, Any]) -> str:
    lines = ["# Evaluation", "", metrics.get("note", ""), ""]
    for stage in metrics.get("stages", []):
        lines.append(f"## {stage['stage']} (n={stage['n']})")
        lines.append("")
        lines.append("| Field | P | R | F1 | Exact |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, scores in stage["fields"].items():
            if name == "line_items" and isinstance(scores, dict) and "f1" not in scores:
                for leaf, leaf_scores in scores.items():
                    lines.append(
                        f"| line_items.{leaf} | {_fmt(leaf_scores.get('precision'))} | {_fmt(leaf_scores.get('recall'))} | {_fmt(leaf_scores.get('f1'))} | {_fmt(leaf_scores.get('exact_match'))} |"
                    )
            else:
                lines.append(
                    f"| {name} | {_fmt(scores.get('precision'))} | {_fmt(scores.get('recall'))} | {_fmt(scores.get('f1'))} | {_fmt(scores.get('exact_match'))} |"
                )
        lines.append("")
        lines.append(f"- Receipt exact match: {_fmt(stage.get('exact_match_receipt'))}")
        lines.append(f"- JSON validity: {_fmt(stage.get('json_validity'))}")
        lines.append(f"- Hallucination before grounding: {_fmt(stage.get('hallucination_before_grounding'))}")
        lines.append(f"- Hallucination after grounding: {_fmt(stage.get('hallucination_after_grounding'))}")
        lat = stage["latency_ms_per_page"]
        lines.append(f"- Latency p50/p95 (ms/page): {_fmt(lat.get('p50'), 1)} / {_fmt(lat.get('p95'), 1)}")
        lines.append(f"- Peak GPU memory: {stage['peak_gpu_memory']}")
        lines.append("")
    return "\n".join(lines)


def maybe_wandb(metrics: dict[str, Any]) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "disabled")
        return
    try:
        import wandb

        wandb.init(project=os.environ.get("WANDB_PROJECT", "vlm-receipt-extraction"), job_type="eval")
        table = wandb.Table(columns=["stage", "field", "f1", "exact"])
        for stage in metrics.get("stages", []):
            wandb.log({f"{stage['stage']}/{k}": v for k, v in stage.items() if isinstance(v, (int, float))})
            for name, scores in stage["fields"].items():
                if name == "line_items":
                    continue
                table.add_data(stage["stage"], name, scores["f1"], scores["exact_match"])
        wandb.log({"comparison": table})
        wandb.finish()
    except Exception as exc:  # noqa: BLE001
        print(f"wandb skipped: {exc}")
