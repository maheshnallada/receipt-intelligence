"""Fire 5 parallel /extract calls and record single-flight behaviour."""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SUMMARY_RE = re.compile(
    r"SUMMARY statuses=(?P<statuses>[\d,]+) max_active_requests=(?P<max_active_requests>\d+)"
)


def parse_summary(text: str) -> dict:
    match = SUMMARY_RE.search(text)
    if not match:
        raise ValueError("no SUMMARY line")
    return {
        "statuses": match.group("statuses").split(","),
        "max_active_requests": int(match.group("max_active_requests")),
    }


def _one(
    client: httpx.Client,
    url: str,
    pdf: bytes,
    index: int,
    active: dict[str, int],
    active_lock: threading.Lock,
) -> dict:
    import httpx

    started = time.perf_counter()
    with active_lock:
        active["current"] += 1
        active["peak"] = max(active["peak"], active["current"])
    try:
        response = client.post(
            url,
            data={"force_refresh": "true"},
            files={"file": (f"receipt-{index}.pdf", pdf, "application/pdf")},
            timeout=180,
        )
    finally:
        with active_lock:
            active["current"] -= 1
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "index": index,
        "status": response.status_code,
        "cache": response.headers.get("X-Cache", "-"),
        "queue_wait_ms": response.headers.get("X-Queue-Wait-Ms", "-"),
        "inference_ms": response.headers.get("X-Inference-Ms", "-"),
        "elapsed_ms": f"{elapsed:.1f}",
    }


def main() -> None:
    import httpx

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument(
        "--pdf",
        default=str(Path(__file__).resolve().parents[1] / "samples" / "receipt_ok.pdf"),
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "reports" / "concurrency_output.txt"),
    )
    args = parser.parse_args()
    pdf = Path(args.pdf).read_bytes()
    url = args.base_url.rstrip("/") + "/extract"
    rows: list[dict] = []
    active = {"current": 0, "peak": 0}
    active_lock = threading.Lock()
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(_one, client, url, pdf, index, active, active_lock)
                for index in range(1, 6)
            ]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row["index"])
    lines = [
        f"request {row['index']} status={row['status']} cache={row['cache']} "
        f"queue_wait_ms={row['queue_wait_ms']} inference_ms={row['inference_ms']} elapsed_ms={row['elapsed_ms']}"
        for row in rows
    ]
    statuses = ",".join(str(row["status"]) for row in rows)
    lines.append(f"SUMMARY statuses={statuses} max_active_requests={active['peak']}")
    text = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
