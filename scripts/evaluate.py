"""Write reports/evaluation.md and reports/metrics.json. Works without a GPU."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from app.application.extract_receipt import ExtractReceipt, receipt_to_dict
from app.config import Settings
from app.infrastructure.cache.memory import MemoryCache
from app.infrastructure.gpu.single_flight import SingleFlightGpuSlot
from app.infrastructure.inference.dummy import DummyExtractor
from app.infrastructure.ocr.tesseract import TesseractTextReader
from app.infrastructure.pdf.renderer import PyMuPdfRenderer
from eval.run_eval import maybe_wandb, render_markdown, summarize, write_reports

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "receipt_ok.pdf": {
        "store_name": "Oak & Ember Cafe",
        "date": "2026-03-15",
        "line_items": [
            {"name": "Latte", "qty": 2.0, "unit_price": 4.5, "amount": 9.0},
            {"name": "Croissant", "qty": 1.0, "unit_price": 3.5, "amount": 3.5},
        ],
        "subtotal": 12.5,
        "tax": 1.0,
        "total": 13.5,
    },
    "receipt_mismatch.pdf": {
        "store_name": "Harbor Books",
        "date": "2026-04-02",
        "line_items": [{"name": "Notebook", "qty": 1.0, "unit_price": 6.0, "amount": 6.0}],
        "subtotal": 10.0,
        "tax": 1.0,
        "total": 11.0,
    },
}


async def dummy_stage() -> dict:
    settings = Settings(model_backend="dummy", redis_url="")
    usecase = ExtractReceipt(
        renderer=PyMuPdfRenderer(),
        extractor=DummyExtractor(),
        ocr=TesseractTextReader(),
        cache=MemoryCache(),
        gpu=SingleFlightGpuSlot(depth=8, timeout_s=5),
        max_bytes=settings.max_bytes,
        max_pages=settings.max_pages,
        long_edge=settings.long_edge,
        ttl_s=settings.ttl_s,
        adapter_revision="dummy",
        schema_version=settings.schema_version,
        prompt_version=settings.prompt_version,
    )
    golds, preds, texts, latencies = [], [], [], []
    for name, gold in SAMPLES.items():
        content = (ROOT / "samples" / name).read_bytes()
        result = await usecase.execute(content=content, content_type="application/pdf", filename=name)
        pred = receipt_to_dict(result.receipt)
        pred.pop("_verification", None)
        golds.append(gold)
        preds.append(pred)
        pages = PyMuPdfRenderer().render(content, settings.long_edge)
        texts.append("\n".join(page.text_layer for page in pages))
        latencies.append(result.inference_ms)
    return summarize(golds, preds, texts, latencies, stage="dummy-samples")


def pending_vlm_stage(label: str) -> dict:
    empty = {"precision": None, "recall": None, "f1": None, "exact_match": None}
    return {
        "stage": label,
        "n": 0,
        "fields": {name: empty for name in ("store_name", "date", "subtotal", "tax", "total", "line_items")},
        "exact_match_receipt": None,
        "json_validity": None,
        "hallucination_before_grounding": None,
        "hallucination_after_grounding": None,
        "latency_ms_per_page": {"p50": None, "p95": None},
        "peak_gpu_memory": "n/a — run on a T4: python scripts/evaluate.py --cord --backend qwen",
    }


def main() -> None:
    os.environ.setdefault("WANDB_MODE", "disabled" if not os.environ.get("WANDB_API_KEY") else os.environ.get("WANDB_MODE", "online"))
    dummy = asyncio.run(dummy_stage())
    metrics = {
        "note": (
            "Dummy-backend numbers are measured on committed sample PDFs. "
            "Zero-shot and fine-tuned CORD rows are pending a T4 run. "
            "Command: MODEL_BACKEND=qwen python scripts/evaluate.py --cord"
        ),
        "stages": [dummy, pending_vlm_stage("zero-shot"), pending_vlm_stage("fine-tuned")],
        "hardware": "CPU dummy path on this machine",
        "cost_assumption": "AWS g4dn.xlarge T4 on-demand ≈ $0.526/hr",
    }
    dest = ROOT / "reports"
    write_reports(metrics, dest)
    maybe_wandb(metrics)
    print(render_markdown(metrics))
    print(f"wrote {dest / 'evaluation.md'} and {dest / 'metrics.json'}")


if __name__ == "__main__":
    main()
