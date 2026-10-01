from __future__ import annotations

import asyncio
import threading
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.domain.document import Page
from app.infrastructure.pdf.renderer import PyMuPdfRenderer, pdf_from_text_pages
from app.main import create_app
from scripts.diagnose_pdf import diagnose_document


def _client(settings: Settings | None = None) -> TestClient:
    app = create_app(settings or Settings(model_backend="dummy", redis_url=""))
    return TestClient(app)


def test_sample_pdf_returns_valid_json(sample_ok: bytes) -> None:
    with _client() as client:
        response = client.post(
            "/extract",
            files={"file": ("receipt_ok.pdf", sample_ok, "application/pdf")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["store_name"] == "Oak & Ember Cafe"
    assert body["date"] == "2026-03-15"
    assert body["total"] == 13.5
    assert body["_verification"]["store_name"] == "verified"
    assert body["_verification"]["total"] == "verified"
    assert response.headers["X-Cache"] == "MISS"
    second = None
    with _client() as client:
        # New app = empty cache. Use one client for hit test below.
        pass
    del second


def test_cache_hit_on_second_call(sample_ok: bytes) -> None:
    with _client() as client:
        first = client.post("/extract", files={"file": ("ok.pdf", sample_ok, "application/pdf")})
        second = client.post("/extract", files={"file": ("ok.pdf", sample_ok, "application/pdf")})
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert second.json()["total"] == first.json()["total"]


def test_password_protected() -> None:
    protected = pdf_from_text_pages(["STORE: Secret"], password="secret")
    with _client() as client:
        response = client.post(
            "/extract",
            files={"file": ("secret.pdf", protected, "application/pdf")},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "PDF_PASSWORD_PROTECTED"


def test_corrupt_pdf() -> None:
    with _client() as client:
        response = client.post(
            "/extract",
            files={"file": ("bad.pdf", b"%PDF-1.4 corrupted", "application/pdf")},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "PDF_CORRUPT"


def test_too_many_pages() -> None:
    pdf = pdf_from_text_pages(["page"] * 6)
    with _client() as client:
        response = client.post("/extract", files={"file": ("long.pdf", pdf, "application/pdf")})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "TOO_MANY_PAGES"


def test_oversize() -> None:
    settings = Settings(model_backend="dummy", max_upload_mb=0, redis_url="")
    settings.max_upload_mb = 0
    with _client(Settings(model_backend="dummy", redis_url="", max_upload_mb=0)) as client:
        response = client.post(
            "/extract",
            files={"file": ("ok.pdf", b"%PDF-1.4 tiny", "application/pdf")},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_unsupported_type() -> None:
    with _client() as client:
        response = client.post(
            "/extract",
            files={"file": ("photo.png", b"\x89PNG", "image/png")},
        )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_health_ready_version() -> None:
    with _client() as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        version = client.get("/version").json()
        assert version["backend"] == "dummy"


def test_missing_ocr_is_read_once_and_remains_unverified(sample_ok, monkeypatch) -> None:
    with _client() as client:
        usecase = client.app.state.usecase
        page = Page(index=0, image_png=b"image", text_layer="", width=1, height=1)
        monkeypatch.setattr(usecase._renderer, "render", lambda content, long_edge: [page])
        monkeypatch.setattr(
            usecase._extractor, "extract_page",
            lambda page, **kwargs: '{"store_name":"DMART","total":845}',
        )
        read_ocr = Mock(return_value="")
        monkeypatch.setattr(usecase._ocr, "read", read_ocr)
        response = client.post(
            "/extract", data={"force_refresh": "true"},
            files={"file": ("scan.pdf", sample_ok, "application/pdf")},
        )
    assert response.status_code == 200
    assert response.json()["total"] is None
    assert response.json()["_verification"]["total"] == "unverified"
    assert response.json()["_verification"]["store_name"] == "unverified"
    read_ocr.assert_called_once_with(b"image")


@pytest.mark.asyncio
@pytest.mark.parametrize("raw, expected_parsed", [('{"store_name":"DMART","total":845}', True), ("not JSON", False)])
async def test_diagnostics_distinguish_parse_failure_from_empty_ocr(sample_ok, raw, expected_parsed) -> None:
    renderer = Mock(wraps=PyMuPdfRenderer())
    renderer.render.return_value = [Page(index=0, image_png=b"image", text_layer="", width=1, height=1)]
    extractor = Mock()
    extractor.extract_page.return_value = raw
    ocr = Mock()
    ocr.read.return_value = ""
    trace = await diagnose_document(
        Settings(model_backend="dummy", redis_url=""), sample_ok, "scan.pdf",
        extractor=extractor, renderer=renderer, ocr=ocr,
    )
    assert "error" not in trace
    assert trace["attempts"][0]["raw_output"] == raw
    assert ("parsed" in trace["attempts"][0]) is expected_parsed
    assert len(trace["attempts"]) == (1 if expected_parsed else 2)
    assert len(trace["ocr_reads"]) == 1
    assert trace["result"]["total"] is None
    assert trace["result"]["_verification"]["total"] == ("unverified" if expected_parsed else "not_found")


@pytest.mark.asyncio
async def test_blocking_inference_keeps_health_and_queue_responsive(sample_ok, monkeypatch) -> None:
    app = create_app(Settings(model_backend="dummy", redis_url="", queue_depth=2, queue_timeout_s=0.1))
    started = threading.Event()
    release = threading.Event()
    stalled = []
    async with app.router.lifespan_context(app):
        extractor = app.state.usecase._extractor
        original = extractor.extract_page

        def blocking_extract(page, *, repair_hint=None):
            started.set()
            if not release.wait(timeout=2):
                stalled.append(True)
            return original(page, repair_hint=repair_hint)

        monkeypatch.setattr(extractor, "extract_page", blocking_extract)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            async def extract():
                return await client.post(
                    "/extract", data={"force_refresh": "true"},
                    files={"file": ("ok.pdf", sample_ok, "application/pdf")},
                )

            first = asyncio.create_task(extract())
            requests = [first]
            try:
                assert await asyncio.to_thread(started.wait, 1)
                assert not stalled, "Inference blocked the event loop until the worker's safety timeout"
                health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
                assert health.status_code == 200
                second = asyncio.create_task(extract())
                requests.append(second)
                async with asyncio.timeout(1):
                    while app.state.usecase._gpu.occupied != 2:
                        await asyncio.sleep(0.001)
                overflow = await asyncio.wait_for(extract(), timeout=0.5)
                assert overflow.status_code == 429
                timed_out = await asyncio.wait_for(second, timeout=0.5)
                assert timed_out.status_code == 503
                assert not first.done()
            finally:
                release.set()
                await asyncio.gather(*requests, return_exceptions=True)
            assert first.result().status_code == 200
            assert app.state.usecase._gpu.occupied == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_holder", [False, True])
async def test_blocking_workers_keep_single_flight_and_measure_wait(sample_ok, monkeypatch, cancel_holder) -> None:
    app = create_app(Settings(model_backend="dummy", redis_url="", queue_depth=8, queue_timeout_s=2))
    started = threading.Event()
    release = threading.Event()
    calls = []
    async with app.router.lifespan_context(app):
        usecase = app.state.usecase
        original = usecase._extractor.extract_page

        def blocking_extract(page, *, repair_hint=None):
            calls.append(threading.get_ident())
            started.set()
            assert release.wait(timeout=3)
            return original(page, repair_hint=repair_hint)

        monkeypatch.setattr(usecase._extractor, "extract_page", blocking_extract)

        async def extract():
            return await usecase.execute(
                content=sample_ok, content_type="application/pdf", filename="ok.pdf", force_refresh=True,
            )

        first = asyncio.create_task(extract())
        requests = [first]
        try:
            assert await asyncio.to_thread(started.wait, 1)
            requests.extend(asyncio.create_task(extract()) for _ in range(4))
            async with asyncio.timeout(1):
                while usecase._gpu.occupied != 5:
                    await asyncio.sleep(0.001)
            if cancel_holder:
                first.cancel()
                await asyncio.sleep(0.02)
                first.cancel()
            await asyncio.sleep(0.02)
            assert not first.done()
            assert len(calls) == 1
            assert usecase._gpu.occupied == 5
        finally:
            release.set()
            results = await asyncio.gather(*requests, return_exceptions=True)
        if cancel_holder:
            assert isinstance(results[0], asyncio.CancelledError)
        else:
            assert results[0].receipt.total.value == 13.5
        for result in results[1:]:
            assert result.receipt.total.value == 13.5
            assert result.queue_wait_ms >= 10
            assert result.cache_hit is False
        assert len(calls) == 5
        assert usecase._gpu.occupied == 0
