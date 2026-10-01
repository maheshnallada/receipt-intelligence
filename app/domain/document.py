"""Uploaded document and page value objects."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.domain.errors import FileTooLarge, TooManyPages, UnsupportedDocument

PDF_CONTENT_TYPES = frozenset(
    {
        "application/pdf",
        "application/x-pdf",
        "application/acrobat",
    }
)


@dataclass(frozen=True)
class Page:
    """One rendered page plus its PDF text layer (may be empty)."""

    index: int
    image_png: bytes
    text_layer: str
    width: int
    height: int


@dataclass(frozen=True)
class DocumentInspection:
    page_count: int
    encrypted: bool


@dataclass(frozen=True)
class UploadedDocument:
    """Validated upload. Bytes hash is the cache identity."""

    content: bytes
    content_type: str
    filename: str
    sha256: str
    page_count: int


def validate_upload(
    content: bytes,
    content_type: str,
    filename: str,
    *,
    max_bytes: int,
) -> str:
    """Size and content-type only. Returns the normalized media type."""
    media = (content_type or "").split(";")[0].strip().lower()
    name = filename or "upload.pdf"
    if media and media not in PDF_CONTENT_TYPES and not name.lower().endswith(".pdf"):
        raise UnsupportedDocument(content_type or media or "unknown")
    if media and media not in PDF_CONTENT_TYPES and name.lower().endswith(".pdf"):
        if media not in {"application/octet-stream", "binary/octet-stream", ""}:
            raise UnsupportedDocument(content_type)
    if not media and not name.lower().endswith(".pdf"):
        raise UnsupportedDocument(content_type or "unknown")
    if len(content) > max_bytes:
        raise FileTooLarge(len(content), max_bytes)
    return media or "application/pdf"


def create_uploaded_document(
    content: bytes,
    content_type: str,
    filename: str,
    *,
    max_bytes: int,
    max_pages: int,
    page_count: int,
) -> UploadedDocument:
    """Factory: size, type, and page cap. Raises domain errors."""
    media = (content_type or "").split(";")[0].strip().lower()
    name = filename or "upload.pdf"
    if media and media not in PDF_CONTENT_TYPES and not name.lower().endswith(".pdf"):
        raise UnsupportedDocument(content_type or media or "unknown")
    if media and media not in PDF_CONTENT_TYPES and name.lower().endswith(".pdf"):
        # Browsers sometimes send application/octet-stream for PDFs.
        if media not in {"application/octet-stream", "binary/octet-stream", ""}:
            raise UnsupportedDocument(content_type)
    if not media and not name.lower().endswith(".pdf"):
        raise UnsupportedDocument(content_type or "unknown")
    size = len(content)
    if size > max_bytes:
        raise FileTooLarge(size, max_bytes)
    if page_count > max_pages:
        raise TooManyPages(page_count, max_pages)
    digest = hashlib.sha256(content).hexdigest()
    return UploadedDocument(
        content=content,
        content_type=media or "application/pdf",
        filename=name,
        sha256=digest,
        page_count=page_count,
    )
