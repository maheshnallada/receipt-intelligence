"""Ports. The use case depends on these protocols, not on Qwen, Redis, or PyMuPDF."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from app.domain.document import DocumentInspection, Page


class PageRenderer(Protocol):
    def inspect(self, content: bytes) -> DocumentInspection:
        """Page count and encryption. Raises PasswordProtected or CorruptDocument."""

    def render(self, content: bytes, long_edge: int) -> list[Page]:
        """Rasterize pages and extract the PDF text layer."""


class ReceiptExtractor(Protocol):
    def extract_page(self, page: Page, *, repair_hint: str | None = None) -> str:
        """Return model text for one page. Temperature 0. May raise InferenceTimeout."""


class TextReader(Protocol):
    def read(self, image_png: bytes) -> str:
        """OCR fallback when the PDF text layer is empty."""


class ExtractionCache(Protocol):
    def get(self, key: str) -> dict | None:
        """Return a schema-valid receipt payload or None."""

    def set(self, key: str, value: dict, ttl_s: int) -> None:
        """Store a schema-valid success only."""


class GpuSlot(Protocol):
    def hold(self) -> AbstractAsyncContextManager[float]:
        """
        Acquire the single GPU slot.

        Yields queue wait in milliseconds. Raises QueueFull or GpuBusy.
        Must release in a finally block.
        """
