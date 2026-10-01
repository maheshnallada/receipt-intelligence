import sys

import httpx

from scripts.concurrency_test import main, parse_summary


def test_summary_parser_reads_fixture() -> None:
    fixture = """
request 1 status=200 cache=MISS
request 2 status=200 cache=MISS
request 3 status=429 cache=-
SUMMARY statuses=200,200,429 max_active_requests=3
"""
    parsed = parse_summary(fixture)
    assert parsed["statuses"] == ["200", "200", "429"]
    assert parsed["max_active_requests"] == 3


def test_main_writes_capture(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "receipt.pdf"
    pdf.write_bytes(b"test input")
    output = tmp_path / "capture.txt"
    real_client = httpx.Client

    def respond(request):
        assert request.url.path == "/extract"
        assert b'force_refresh' in request.content
        return httpx.Response(200, headers={"X-Cache": "MISS"}, json={})

    monkeypatch.setattr(
        httpx, "Client", lambda: real_client(transport=httpx.MockTransport(respond))
    )
    monkeypatch.setattr(
        sys, "argv", ["concurrency_test.py", "--pdf", str(pdf), "--out", str(output)]
    )
    main()
    capture = output.read_text()
    assert parse_summary(capture)["statuses"] == ["200"] * 5
    assert capture.count("cache=MISS") == 5
