"""Opt-in, private diagnostics for one document using the service pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from app.application.extract_receipt import ExtractReceipt, receipt_to_dict
from app.config import Settings
from app.domain.grounding import ground_receipt
from app.domain.parse import parse_model_output
from app.domain.receipt import ReceiptInvariantError
from app.infrastructure.cache.memory import MemoryCache
from app.infrastructure.gpu.single_flight import SingleFlightGpuSlot
from app.infrastructure.ocr.tesseract import TesseractTextReader
from app.infrastructure.pdf.renderer import PyMuPdfRenderer
from app.main import _build_extractor


async def diagnose_document(settings, content, filename, *, extractor, renderer, ocr) -> dict:
    trace = {
        "filename": filename,
        "settings": {
            "backend": settings.model_backend,
            "model_id": settings.model_id,
            "adapter_revision": settings.adapter_revision,
            "long_edge": settings.long_edge,
            "max_pixels": settings.max_pixels,
        },
        "pages": [],
        "attempts": [],
        "ocr_reads": [],
        "grounding": [],
    }

    def render(document, long_edge):
        pages = renderer.render(document, long_edge)
        trace["pages"] = [
            {"page": page.index + 1, "width": page.width, "height": page.height,
             "text_layer": page.text_layer, "text_source": "pdf" if page.text_layer.strip() else "ocr"}
            for page in pages
        ]
        return pages

    def generate(page, *, repair_hint=None):
        attempt = {"page": page.index + 1, "repair_hint": repair_hint}
        trace["attempts"].append(attempt)
        raw = extractor.extract_page(page, repair_hint=repair_hint)
        attempt["raw_output"] = raw
        return raw

    def parse(raw):
        attempt = trace["attempts"][-1]
        try:
            receipt = parse_model_output(raw)
        except ReceiptInvariantError as exc:
            attempt["parse_error"] = str(exc)
            attempt["validation_detail"] = str(exc.__cause__) if exc.__cause__ else None
            raise
        attempt["parsed"] = receipt_to_dict(receipt)
        return receipt

    def read_ocr(image):
        text = ocr.read(image) or ""
        trace["ocr_reads"].append({"text": text, "characters": len(text.strip())})
        return text

    def ground(receipt, text):
        grounded = ground_receipt(receipt, text)
        trace["grounding"].append({
            "text": text,
            "characters": len(text.strip()),
            "before": receipt_to_dict(receipt),
            "after": receipt_to_dict(grounded),
        })
        return grounded

    usecase = ExtractReceipt(
        renderer=SimpleNamespace(inspect=renderer.inspect, render=render),
        extractor=SimpleNamespace(extract_page=generate),
        ocr=SimpleNamespace(read=read_ocr),
        cache=MemoryCache(maxsize=1, ttl_s=settings.ttl_s),
        gpu=SingleFlightGpuSlot(depth=1, timeout_s=settings.queue_timeout_s),
        max_bytes=settings.max_bytes, max_pages=settings.max_pages,
        long_edge=settings.long_edge, ttl_s=settings.ttl_s,
        adapter_revision=settings.adapter_revision,
        schema_version=settings.schema_version, prompt_version=settings.prompt_version,
        parse=parse, ground=ground,
    )
    try:
        result = await usecase.execute(
            content=content, content_type="application/pdf", filename=filename, force_refresh=True,
        )
        trace["result"] = receipt_to_dict(result.receipt)
        trace["processing_ms"] = result.inference_ms
    except Exception as exc:
        trace["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--backend", choices=("qwen", "dummy"), default="qwen")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    settings = Settings(model_backend=args.backend, redis_url="", logfire_send_to_logfire="false")
    if settings.model_backend == "qwen" and not settings.adapter_path:
        parser.error("Set ADAPTER_PATH to the same adapter snapshot used by the API.")
    content = args.pdf.read_bytes()
    destination = args.out or Path("diagnostics") / f"receipt-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "x", encoding="utf-8", opener=lambda path, flags: os.open(path, flags, 0o600)) as output:
        try:
            extractor = _build_extractor(settings)
            trace = asyncio.run(diagnose_document(
                settings, content, args.pdf.name, extractor=extractor,
                renderer=PyMuPdfRenderer(), ocr=TesseractTextReader(),
            ))
        except Exception as exc:
            trace = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        json.dump(trace, output, indent=2, ensure_ascii=True)
        output.write("\n")
    print(f"Private diagnostic saved: {destination}. Contains receipt text and raw predictions; do not publish.")
    if "error" in trace:
        raise SystemExit(1)


if __name__ == "__main__":
    main()