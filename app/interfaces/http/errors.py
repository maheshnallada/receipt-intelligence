"""Domain error → stable JSON error body. No stack traces in HTTP."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from app.domain.errors import (
    CorruptDocument,
    DomainError,
    FileTooLarge,
    GpuBusy,
    InferenceTimeout,
    PasswordProtected,
    QueueFull,
    TooManyPages,
    UnsupportedDocument,
)
from app.interfaces.http.schemas import ErrorBody, ErrorResponse

_STATUS: dict[type[DomainError], tuple[int, str]] = {
    PasswordProtected: (400, "PDF_PASSWORD_PROTECTED"),
    CorruptDocument: (400, "PDF_CORRUPT"),
    FileTooLarge: (413, "FILE_TOO_LARGE"),
    TooManyPages: (422, "TOO_MANY_PAGES"),
    UnsupportedDocument: (415, "UNSUPPORTED_MEDIA_TYPE"),
    InferenceTimeout: (504, "INFERENCE_TIMEOUT"),
    QueueFull: (429, "QUEUE_FULL"),
    GpuBusy: (503, "QUEUE_TIMEOUT"),
}


def status_for(error: DomainError) -> tuple[int, str]:
    for cls, mapped in _STATUS.items():
        if isinstance(error, cls):
            return mapped
    return 500, "INTERNAL_ERROR"


def error_payload(code: str, message: str, request_id: str) -> dict:
    return ErrorResponse(error=ErrorBody(code=code, message=message, request_id=request_id)).model_dump()


async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    status, code = status_for(exc)
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=status,
        content=error_payload(code, str(exc), request_id),
    )
