"""Bounded OCR fallback with receipt-layout retries for empty recognition."""

from __future__ import annotations

import logging
from io import BytesIO

logger = logging.getLogger(__name__)
OCR_TIMEOUT_S = 15
OCR_LONG_EDGE = 3000


class TesseractTextReader:
    def read(self, image_png: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image, ImageOps
        except ImportError:
            logger.warning("pytesseract or pillow missing; OCR disabled")
            return ""
        def recognize(image, config="") -> str:
            try:
                text = pytesseract.image_to_string(image, config=config, timeout=OCR_TIMEOUT_S) or ""
                return text if any(character.isalnum() for character in text) else ""
            except RuntimeError as exc:
                logger.warning("OCR attempt failed (%s); trying remaining layouts", type(exc).__name__)
                return ""

        try:
            with Image.open(BytesIO(image_png)) as image:
                text = recognize(image)
                if text:
                    return text
                prepared = ImageOps.autocontrast(ImageOps.grayscale(image))
                scale = min(3.0, OCR_LONG_EDGE / max(prepared.size))
                size = tuple(max(1, round(dimension * scale)) for dimension in prepared.size)
                prepared = prepared.resize(size, Image.Resampling.LANCZOS)
                for mode in (6, 11):
                    text = recognize(prepared, config=f"--psm {mode} --dpi 300 -c preserve_interword_spaces=1")
                    if text:
                        return text
                logger.warning("OCR returned no usable text after three bounded attempts")
                return ""
        except Exception as exc:  # noqa: BLE001 — tesseract binary may be absent
            logger.warning("tesseract unavailable: %s", exc)
            return ""
