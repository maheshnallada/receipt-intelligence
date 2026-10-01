"""PyMuPDF renderer: password, corrupt, text layer, page images."""

from __future__ import annotations

import io

from app.domain.document import DocumentInspection, Page
from app.domain.errors import CorruptDocument, PasswordProtected


class PyMuPdfRenderer:
    def inspect(self, content: bytes) -> DocumentInspection:
        doc = self._open(content)
        try:
            return DocumentInspection(page_count=doc.page_count, encrypted=bool(doc.is_encrypted))
        finally:
            doc.close()

    def render(self, content: bytes, long_edge: int) -> list[Page]:
        doc = self._open(content)
        try:
            pages: list[Page] = []
            for index, page in enumerate(doc):
                rect = page.rect
                longest = max(rect.width, rect.height) or 1.0
                scale = long_edge / longest
                pix = page.get_pixmap(matrix=_identity_scale(scale), alpha=False)
                image = pix.tobytes("png")
                text = page.get_text("text") or ""
                pages.append(
                    Page(
                        index=index,
                        image_png=image,
                        text_layer=text,
                        width=pix.width,
                        height=pix.height,
                    )
                )
            return pages
        finally:
            doc.close()

    def _open(self, content: bytes):
        import fitz

        if not content.startswith(b"%PDF"):
            raise CorruptDocument("file is not a PDF")
        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:  # noqa: BLE001 — MuPDF raises many parse errors
            raise CorruptDocument("PDF is corrupt or unreadable") from exc
        if doc.is_encrypted or doc.needs_pass:
            # Authenticate with an empty password; still locked → protected.
            try:
                unlocked = doc.authenticate("")
            except Exception:
                unlocked = 0
            if not unlocked:
                doc.close()
                raise PasswordProtected("PDF is password-protected")
        try:
            _ = doc.page_count
            if doc.page_count < 1:
                raise CorruptDocument("PDF has no pages")
        except PasswordProtected:
            raise
        except CorruptDocument:
            raise
        except Exception as exc:  # noqa: BLE001
            doc.close()
            raise CorruptDocument("PDF is corrupt or unreadable") from exc
        return doc


def _identity_scale(scale: float):
    import fitz

    return fitz.Matrix(scale, scale)


def pdf_from_text_pages(pages: list[str], *, password: str | None = None) -> bytes:
    """Build a tiny synthetic receipt PDF for tests and the desk samples."""
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=420, height=640)
        page.draw_rect(page.rect, color=(0.15, 0.12, 0.1), fill=(0.99, 0.97, 0.92))
        page.insert_textbox(
            fitz.Rect(28, 36, 392, 610),
            text,
            fontsize=13,
            fontname="helv",
            color=(0.12, 0.1, 0.08),
        )
    sink = io.BytesIO()
    if password:
        doc.save(
            sink,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw=password,
            user_pw=password,
        )
    else:
        doc.save(sink)
    doc.close()
    return sink.getvalue()
