"""ExtractReceipt: the only orchestration. Imports domain only."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.domain.document import Page, UploadedDocument, create_uploaded_document, validate_upload
from app.domain.grounding import ground_receipt
from app.domain.merge import merge_pages
from app.domain.parse import parse_model_output
from app.domain.ports import ExtractionCache, GpuSlot, PageRenderer, ReceiptExtractor, TextReader
from app.domain.receipt import Receipt, ReceiptInvariantError, Verification, empty_receipt
from app.domain.sanity import check_totals

GroundFn = Callable[[Receipt, str], Receipt]
SanityFn = Callable[[Receipt], Receipt]
ParseFn = Callable[[str], Receipt]


@dataclass(frozen=True)
class ExtractResult:
    receipt: Receipt
    document: UploadedDocument
    cache_hit: bool
    queue_wait_ms: float
    inference_ms: float
    page_count: int
    verified: int
    unverified: int
    not_found: int


class ExtractReceipt:
    def __init__(
        self,
        *,
        renderer: PageRenderer,
        extractor: ReceiptExtractor,
        ocr: TextReader,
        cache: ExtractionCache,
        gpu: GpuSlot,
        max_bytes: int,
        max_pages: int,
        long_edge: int,
        ttl_s: int,
        adapter_revision: str,
        schema_version: str,
        prompt_version: str,
        ground: GroundFn = ground_receipt,
        sanity: SanityFn = check_totals,
        parse: ParseFn = parse_model_output,
    ) -> None:
        self._renderer = renderer
        self._extractor = extractor
        self._ocr = ocr
        self._cache = cache
        self._gpu = gpu
        self._max_bytes = max_bytes
        self._max_pages = max_pages
        self._long_edge = long_edge
        self._ttl_s = ttl_s
        self._adapter_revision = adapter_revision
        self._schema_version = schema_version
        self._prompt_version = prompt_version
        self._ground = ground
        self._sanity = sanity
        self._parse = parse

    def cache_key(self, sha256: str) -> str:
        return f"{sha256}:{self._adapter_revision}:{self._schema_version}:{self._prompt_version}"

    async def execute(
        self,
        *,
        content: bytes,
        content_type: str,
        filename: str,
        force_refresh: bool = False,
    ) -> ExtractResult:
        validate_upload(content, content_type, filename, max_bytes=self._max_bytes)
        inspection = self._renderer.inspect(content)
        document = create_uploaded_document(
            content,
            content_type,
            filename,
            max_bytes=self._max_bytes,
            max_pages=self._max_pages,
            page_count=inspection.page_count,
        )
        key = self.cache_key(document.sha256)
        if not force_refresh:
            cached = self._cache.get(key)
            if cached is not None:
                receipt = _receipt_from_cached(cached)
                return self._result(receipt, document, cache_hit=True, queue_wait_ms=0.0, inference_ms=0.0)

        async with self._gpu.hold() as queue_wait_ms:
            started = time.perf_counter()
            worker = asyncio.create_task(asyncio.to_thread(self._extract_document, document))
            try:
                checked = await asyncio.shield(worker)
            except asyncio.CancelledError:
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                raise
            inference_ms = (time.perf_counter() - started) * 1000.0

        self._cache.set(key, receipt_to_assignment_dict(checked), self._ttl_s)
        return self._result(
            checked,
            document,
            cache_hit=False,
            queue_wait_ms=queue_wait_ms,
            inference_ms=inference_ms,
        )

    def _extract_document(self, document: UploadedDocument) -> Receipt:
        pages = self._renderer.render(document.content, self._long_edge)
        extracted = [self._extract_one(page) for page in pages]
        page_texts = [self._page_text(page) for page in pages]
        grounded_pages = [
            self._ground(receipt, text)
            for receipt, text in zip(extracted, page_texts, strict=True)
        ]
        merged = merge_pages(grounded_pages)
        combined = "\n".join(page_texts)
        grounded = self._ground(merged, combined)
        return self._sanity(grounded)

    def _extract_one(self, page: Page) -> Receipt:
        raw = self._extractor.extract_page(page)
        try:
            return self._parse(raw)
        except ReceiptInvariantError as first_error:
            repaired = self._extractor.extract_page(page, repair_hint=str(first_error))
            try:
                return self._parse(repaired)
            except ReceiptInvariantError:
                return empty_receipt()

    def _page_text(self, page: Page) -> str:
        text = (page.text_layer or "").strip()
        if text:
            return text
        return self._ocr.read(page.image_png) or ""

    def _result(
        self,
        receipt: Receipt,
        document: UploadedDocument,
        *,
        cache_hit: bool,
        queue_wait_ms: float,
        inference_ms: float,
    ) -> ExtractResult:
        counts = {"verified": 0, "unverified": 0, "not_found": 0}
        for status in receipt.verification.values():
            counts[status.value] += 1
        return ExtractResult(
            receipt=receipt,
            document=document,
            cache_hit=cache_hit,
            queue_wait_ms=queue_wait_ms,
            inference_ms=inference_ms,
            page_count=document.page_count,
            verified=counts["verified"],
            unverified=counts["unverified"],
            not_found=counts["not_found"],
        )


def receipt_to_assignment_dict(receipt: Receipt) -> dict:
    return {
        "store_name": receipt.store_name,
        "date": receipt.date.value if receipt.date else None,
        "line_items": [
            {
                "name": item.name,
                "qty": item.qty,
                "unit_price": item.unit_price.value if item.unit_price else None,
                "amount": item.amount.value if item.amount else None,
            }
            for item in receipt.line_items
        ],
        "subtotal": receipt.subtotal.value if receipt.subtotal else None,
        "tax": receipt.tax.value if receipt.tax else None,
        "total": receipt.total.value if receipt.total else None,
        "_verification": {key: value.value for key, value in receipt.verification.items()},
    }


def _receipt_from_cached(payload: dict) -> Receipt:
    verification = payload.get("_verification") or {}
    body = {key: value for key, value in payload.items() if key != "_verification"}
    receipt = Receipt.from_payload(body)
    if verification:
        return receipt.with_verification(
            {key: Verification(value) for key, value in verification.items()}
        )
    return receipt
