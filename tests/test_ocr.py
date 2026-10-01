from io import BytesIO

import pytest
from PIL import Image

from app.infrastructure.ocr.tesseract import TesseractTextReader


@pytest.fixture
def page_png():
    with Image.new("RGB", (713, 1008), "white") as image:
        output = BytesIO()
        image.save(output, format="PNG")
    return output.getvalue()


def test_empty_default_ocr_retries_receipt_layout(page_png, monkeypatch):
    calls = []

    def recognize(image, **kwargs):
        calls.append((image.size, image.mode, kwargs))
        return " \n\f" if len(calls) == 1 else "STORE\nTOTAL 845.00\n"

    monkeypatch.setattr("pytesseract.image_to_string", recognize)
    assert TesseractTextReader().read(page_png) == "STORE\nTOTAL 845.00\n"
    assert len(calls) == 2
    assert calls[0][0] == (713, 1008)
    assert calls[1][0][1] == 3000
    assert calls[1][1] == "L"
    assert "--psm 6" in calls[1][2]["config"]
    assert all(0 < call[2]["timeout"] <= 20 for call in calls)


@pytest.mark.parametrize("last_text", ["", "SPARSE RECEIPT\nTOTAL 845.00\n"])
def test_sparse_retry_and_empty_exhaustion(page_png, monkeypatch, last_text):
    calls = []

    def recognize(image, **kwargs):
        calls.append(kwargs)
        return last_text if len(calls) == 3 else " \n\f"

    monkeypatch.setattr("pytesseract.image_to_string", recognize)
    assert TesseractTextReader().read(page_png) == last_text
    assert len(calls) == 3
    assert "--psm 11" in calls[-1]["config"]


def test_successful_default_ocr_is_unchanged(page_png, monkeypatch):
    calls = []

    def recognize(image, **kwargs):
        calls.append(image.size)
        return "Original text\n"

    monkeypatch.setattr("pytesseract.image_to_string", recognize)
    assert TesseractTextReader().read(page_png) == "Original text\n"
    assert calls == [(713, 1008)]


def test_timed_out_attempt_can_recover(page_png, monkeypatch):
    calls = []

    def recognize(image, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("Tesseract process timeout")
        return "TOTAL 845.00"

    monkeypatch.setattr("pytesseract.image_to_string", recognize)
    assert TesseractTextReader().read(page_png) == "TOTAL 845.00"
    assert len(calls) == 2


def test_missing_tesseract_does_not_retry(page_png, monkeypatch):
    import pytesseract

    calls = []

    def recognize(image, **kwargs):
        calls.append(kwargs)
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", recognize)
    assert TesseractTextReader().read(page_png) == ""
    assert len(calls) == 1


def test_invalid_image_stays_empty(monkeypatch):
    def recognize(image, **kwargs):
        pytest.fail("Tesseract must not run on invalid image bytes")

    monkeypatch.setattr("pytesseract.image_to_string", recognize)
    assert TesseractTextReader().read(b"not an image") == ""