"""Logfire init, spans, scrubbing. Stdout JSON when the token is unset."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.config import Settings
from app.domain.grounding import ground_receipt as domain_ground
from app.domain.parse import parse_model_output as domain_parse
from app.domain.receipt import Receipt
from app.domain.sanity import check_totals as domain_check

logger = logging.getLogger("vlm-extraction")


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    if not any(isinstance(item, JsonLogFormatter) for item in (h.formatter for h in root.handlers if h.formatter)):
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload)


def configure_logfire(settings: Settings) -> Any:
    configure_logging()
    try:
        import logfire
    except ImportError:
        logger.warning("logfire not installed; stdout logging only")
        return None
    send = settings.send_to_logfire
    try:
        logfire.configure(
            send_to_logfire=send,
            service_name="vlm-extraction",
            token=settings.logfire_token or None,
            console=False,
        )
    except Exception as exc:  # noqa: BLE001 — startup must succeed if Logfire is unreachable
        logger.warning("logfire configure failed: %s", exc)
        try:
            logfire.configure(send_to_logfire=False, service_name="vlm-extraction")
        except Exception:
            return None
    return logfire


def instrument_app(app: Any, logfire: Any) -> None:
    if logfire is None:
        return
    try:
        logfire.instrument_fastapi(app)
        logfire.instrument_pydantic()
    except Exception as exc:  # noqa: BLE001
        logger.warning("logfire instrumentation skipped: %s", exc)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    """Manual span. Never attach document bytes, images, text, or field values."""
    safe = {key: value for key, value in attrs.items() if key not in _FORBIDDEN}
    try:
        import logfire

        with logfire.span(name, **safe):
            yield
            return
    except Exception:
        pass
    logger.info("span %s %s", name, json.dumps(safe, default=str))
    yield


_FORBIDDEN = {
    "pdf_bytes",
    "image",
    "document_text",
    "raw_output",
    "store_name",
    "line_items",
    "total",
}


def traced_ground(receipt: Receipt, text: str) -> Receipt:
    with span("ground"):
        return domain_ground(receipt, text)


def traced_sanity(receipt: Receipt) -> Receipt:
    with span("sanity"):
        return domain_check(receipt)


def traced_parse(text: str) -> Receipt:
    with span("json.parse"):
        return domain_parse(text)


def wrap_extractor(extractor: Any) -> Any:
    class _Traced:
        def extract_page(self, page: Any, *, repair_hint: str | None = None) -> str:
            attrs = {"repair": bool(repair_hint)}
            with span("vlm.infer", **attrs):
                if repair_hint:
                    try:
                        import logfire

                        logfire.info("json_repair")
                    except Exception:
                        logger.info("json_repair")
                return extractor.extract_page(page, repair_hint=repair_hint)

    return _Traced()


def wrap_renderer(renderer: Any) -> Any:
    class _Traced:
        def inspect(self, content: bytes) -> Any:
            return renderer.inspect(content)

        def render(self, content: bytes, long_edge: int) -> Any:
            with span("pdf.render", long_edge=long_edge):
                return renderer.render(content, long_edge)

    return _Traced()


def wrap_cache(cache: Any) -> Any:
    class _Traced:
        def get(self, key: str) -> dict | None:
            return cache.get(key)

        def set(self, key: str, value: dict, ttl_s: int) -> None:
            with span("cache.write"):
                cache.set(key, value, ttl_s)

    return _Traced()


def bind_request_attributes(
    *,
    request_id: str,
    page_count: int,
    cache_hit: bool,
    queue_wait_ms: float,
    inference_ms: float,
    backend: str,
    adapter_revision: str,
    verified: int,
    unverified: int,
    not_found: int,
) -> None:
    attrs = {
        "request_id": request_id,
        "page_count": page_count,
        "cache": "HIT" if cache_hit else "MISS",
        "queue_wait_ms": round(queue_wait_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "backend": backend,
        "adapter_revision": adapter_revision,
        "verified": verified,
        "unverified": unverified,
        "not_found": not_found,
    }
    try:
        import logfire

        logfire.info("extract.complete", **attrs)
    except Exception:
        logger.info("extract.complete %s", json.dumps(attrs))
