from app.domain.errors import (
    CorruptDocument,
    FileTooLarge,
    GpuBusy,
    InferenceTimeout,
    PasswordProtected,
    QueueFull,
    TooManyPages,
    UnsupportedDocument,
)
from app.interfaces.http.errors import error_payload, status_for


def test_domain_errors_map_to_stable_codes() -> None:
    assert status_for(PasswordProtected("x")) == (400, "PDF_PASSWORD_PROTECTED")
    assert status_for(CorruptDocument("x")) == (400, "PDF_CORRUPT")
    assert status_for(FileTooLarge(9, 1)) == (413, "FILE_TOO_LARGE")
    assert status_for(TooManyPages(9, 5)) == (422, "TOO_MANY_PAGES")
    assert status_for(UnsupportedDocument("image/png")) == (415, "UNSUPPORTED_MEDIA_TYPE")
    assert status_for(QueueFull("x")) == (429, "QUEUE_FULL")
    assert status_for(GpuBusy("x")) == (503, "QUEUE_TIMEOUT")
    assert status_for(InferenceTimeout("x")) == (504, "INFERENCE_TIMEOUT")


def test_error_body_shape() -> None:
    body = error_payload("PDF_CORRUPT", "bad pdf", "req-1")
    assert body == {"error": {"code": "PDF_CORRUPT", "message": "bad pdf", "request_id": "req-1"}}
