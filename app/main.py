"""Composition root. The only module that wires adapters to ports."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.application.extract_receipt import ExtractReceipt
from app.config import Settings
from app.domain.errors import DomainError
from app.infrastructure.cache.memory import MemoryCache
from app.infrastructure.gpu.single_flight import SingleFlightGpuSlot
from app.infrastructure.observability.logfire_setup import (
    configure_logfire,
    instrument_app,
    traced_ground,
    traced_parse,
    traced_sanity,
    wrap_cache,
    wrap_extractor,
    wrap_renderer,
)
from app.infrastructure.ocr.tesseract import TesseractTextReader
from app.infrastructure.pdf.renderer import PyMuPdfRenderer
from app.interfaces.http.errors import domain_error_handler, error_payload, status_for
from app.interfaces.http.routes import build_router

logger = logging.getLogger("vlm-extraction")


def _build_cache(settings: Settings):
    memory = MemoryCache(maxsize=128, ttl_s=settings.ttl_s)
    if settings.redis_url:
        from app.infrastructure.cache.redis import RedisCache

        return wrap_cache(RedisCache(settings.redis_url, fallback=memory))
    return wrap_cache(memory)


def _build_extractor(settings: Settings):
    backend = settings.model_backend.lower()
    if backend == "qwen":
        from app.infrastructure.inference.qwen import QwenExtractor

        return QwenExtractor(
            model_id=settings.model_id,
            adapter_path=settings.adapter_path,
            max_pixels=settings.max_pixels,
            timeout_s=settings.inference_timeout_s,
            hf_home=settings.hf_home,
        )
    from app.infrastructure.inference.dummy import DummyExtractor

    return DummyExtractor(timeout_s=settings.inference_timeout_s)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logfire = configure_logfire(settings)
    gpu = SingleFlightGpuSlot(depth=settings.queue_depth, timeout_s=settings.queue_timeout_s)
    state = {"ready": False, "accepting": True, "gpu": gpu}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            extractor = _build_extractor(settings)
            usecase = ExtractReceipt(
                renderer=wrap_renderer(PyMuPdfRenderer()),
                extractor=wrap_extractor(extractor),
                ocr=TesseractTextReader(),
                cache=_build_cache(settings),
                gpu=gpu,
                max_bytes=settings.max_bytes,
                max_pages=settings.max_pages,
                long_edge=settings.long_edge,
                ttl_s=settings.ttl_s,
                adapter_revision=settings.adapter_revision,
                schema_version=settings.schema_version,
                prompt_version=settings.prompt_version,
                ground=traced_ground,
                sanity=traced_sanity,
                parse=traced_parse,
            )
            app.state.settings = settings
            app.state.usecase = usecase
            app.state.extractor = extractor
            router = build_router(
                usecase=usecase,
                model_id=settings.model_id,
                adapter_revision=settings.adapter_revision,
                schema_version=settings.schema_version,
                backend=settings.model_backend,
                prompt_version=settings.prompt_version,
                is_ready=lambda: state["ready"],
                is_accepting=lambda: state["accepting"],
            )
            app.include_router(router)
            state["ready"] = True
            yield
        finally:
            state["accepting"] = False
            state["ready"] = False
            deadline = asyncio.get_event_loop().time() + 30
            while gpu.occupied and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.05)

    app = FastAPI(title="vlm-extraction", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        incoming = request.headers.get("X-Request-Id")
        try:
            request_id = str(uuid.UUID(incoming)) if incoming else str(uuid.uuid4())
        except (ValueError, TypeError):
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except DomainError as exc:
            status, code = status_for(exc)
            response = JSONResponse(
                status_code=status,
                content=error_payload(code, str(exc), request_id),
            )
        except Exception:
            logger.exception("unhandled error")
            response = JSONResponse(
                status_code=500,
                content=error_payload("INTERNAL_ERROR", "internal error", request_id),
            )
        response.headers["X-Request-Id"] = request_id
        return response

    app.add_exception_handler(DomainError, domain_error_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    instrument_app(app, logfire)
    app.state.settings = settings
    return app


app = create_app()
