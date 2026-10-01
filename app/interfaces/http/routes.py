"""HTTP adapters. No grounding, merge, sanity, or cache logic."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.application.extract_receipt import ExtractReceipt
from app.interfaces.http.errors import error_payload
from app.interfaces.http.schemas import ReceiptOut, VersionResponse


def build_router(
    *,
    usecase: ExtractReceipt,
    model_id: str,
    adapter_revision: str,
    schema_version: str,
    backend: str,
    prompt_version: str,
    is_ready: Callable[[], bool],
    is_accepting: Callable[[], bool],
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/ready")
    async def ready() -> JSONResponse:
        if not is_ready():
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    @router.get("/version")
    async def version() -> VersionResponse:
        return VersionResponse(
            model_id=model_id,
            adapter_revision=adapter_revision,
            schema_version=schema_version,
            backend=backend,
            prompt_version=prompt_version,
        )

    @router.post("/extract")
    async def extract(
        request: Request,
        file: UploadFile,
        force_refresh: bool = Form(False),
    ) -> JSONResponse:
        if not is_accepting():
            request_id = getattr(request.state, "request_id", "unknown")
            return JSONResponse(
                status_code=503,
                content=error_payload("SHUTTING_DOWN", "server is shutting down", request_id),
            )
        content = await file.read()
        result = await usecase.execute(
            content=content,
            content_type=file.content_type or "",
            filename=file.filename or "upload.pdf",
            force_refresh=force_refresh,
        )
        payload = ReceiptOut.from_domain(result.receipt)
        request_id = getattr(request.state, "request_id", "unknown")
        from app.infrastructure.observability.logfire_setup import bind_request_attributes

        bind_request_attributes(
            request_id=request_id,
            page_count=result.page_count,
            cache_hit=result.cache_hit,
            queue_wait_ms=result.queue_wait_ms,
            inference_ms=result.inference_ms,
            backend=backend,
            adapter_revision=adapter_revision,
            verified=result.verified,
            unverified=result.unverified,
            not_found=result.not_found,
        )
        return JSONResponse(
            content=payload.model_dump(by_alias=True),
            headers={
                "X-Request-Id": request_id,
                "X-Cache": "HIT" if result.cache_hit else "MISS",
                "X-Queue-Wait-Ms": f"{result.queue_wait_ms:.1f}",
                "X-Inference-Ms": f"{result.inference_ms:.1f}",
            },
        )

    return router
