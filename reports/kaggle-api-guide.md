# Minimal Kaggle API validation

This bundle runs the existing FastAPI backend, not a second notebook-only
extractor. It contains `app/`, the concurrency probe, CPU requirements, package
metadata, and three synthetic sample PDFs. No UI, training notebooks, model
weights, `.env`, or credentials are included. It needs Python 3.11 or newer.

Create a **private Kaggle dataset** from `kaggle-api.zip`, attach it to a fresh
notebook, and enable **GPU** and **Internet**. Attach your own receipt PDF
separately. Run the Python blocks below as separate notebook cells, in order.
Do not load another model in that session: the API subprocess owns the GPU.
Kaggle may unpack the ZIP during dataset creation; both layouts are supported.

## 1. Copy the bundle and install dependencies

```python
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

assert sys.version_info >= (3, 11), "Python 3.11+ is required."
BUNDLE_PATH = ""
PROJECT = Path("/kaggle/working/kaggle-api")
if not BUNDLE_PATH:
    roots = sorted({
        path.parent for path in Path("/kaggle/input").rglob("pyproject.toml")
        if (path.parent / "app/main.py").is_file()
    })
    candidates = roots or sorted(Path("/kaggle/input").rglob("kaggle-api.zip"))
    print("Bundle candidates:", *candidates, sep="\n")
    if len(candidates) != 1:
        raise ValueError("Set BUNDLE_PATH to the exact extracted bundle folder or ZIP above.")
    bundle = candidates[0]
else:
    bundle = Path(BUNDLE_PATH)
if PROJECT.exists():
    raise RuntimeError("Project folder already exists. Skip this cell or start a fresh session.")
if bundle.is_dir():
    shutil.copytree(bundle, PROJECT)
else:
    with zipfile.ZipFile(bundle) as archive:
        destination = PROJECT.parent.resolve()
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(PROJECT.resolve()):
                raise ValueError("Unexpected archive member outside kaggle-api/.")
        archive.extractall(destination)
assert (PROJECT / "app/main.py").is_file()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(PROJECT / "requirements.txt")])
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "transformers>=4.57,<5", "peft>=0.19.1,<0.20", "accelerate>=1.2,<2",
    "bitsandbytes>=0.46,<1", "huggingface_hub>=0.34,<1",
])
if not shutil.which("tesseract"):
    subprocess.check_call(["apt-get", "update", "-qq"])
    subprocess.check_call(["apt-get", "install", "-y", "tesseract-ocr", "tesseract-ocr-eng"])
subprocess.check_call(["tesseract", "--version"])
print("Project ready:", PROJECT)
```

Kaggle GPU images normally include PyTorch. If an install requires a kernel
restart, restart before continuing and restore the `PROJECT` path. If system
package installation is prohibited, use an image with Tesseract installed;
do not claim OCR-fallback validation when the executable is absent. This
minimal bundle intentionally excludes training libraries such as TRL/Unsloth.

## 2. Download the adapter and start the real API

The public adapter requires no token. `ADAPTER_REVISION` in the service is a
cache/version label, not a download pin; `snapshot_download(revision=...)`
below performs the actual pin. The revision matches the saved uploaded-PDF
notebook run, not necessarily the earlier CORD benchmark.

```python
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

PROJECT = Path("/kaggle/working/kaggle-api")
assert torch.cuda.is_available(), "Enable a Kaggle GPU."
assert torch.cuda.memory_allocated(0) == 0, "Restart: another model may be loaded in this notebook."
ADAPTER_ID = "Mahesh-Nallada/vlm-receipt-extraction-lora"
ADAPTER_REVISION = "e9d1eb7ff9a11273a87a5d27a9fec7ba35b4ee92"
adapter_dir = snapshot_download(
    ADAPTER_ID, revision=ADAPTER_REVISION,
    allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
)
assert (Path(adapter_dir) / "adapter_model.safetensors").is_file()
OUTPUT = PROJECT / "reports"
OUTPUT.mkdir(exist_ok=True)
with socket.socket() as port_check:
    port_check.bind(("127.0.0.1", 8000))
environment = os.environ.copy()
settings = {
    "MODEL_BACKEND": "qwen",
    "MODEL_ID": "Qwen/Qwen2.5-VL-3B-Instruct",
    "ADAPTER_PATH": adapter_dir,
    "ADAPTER_REVISION": ADAPTER_REVISION,
    "QUEUE_DEPTH": "8",
    "QUEUE_TIMEOUT_S": "60",
    "INFERENCE_TIMEOUT_S": "120",
    "REDIS_URL": "",
    "LOGFIRE_SEND_TO_LOGFIRE": "false",
    "LOGFIRE_TOKEN": "",
    "HF_HOME": "/kaggle/working/hf-cache",
}
environment.update(settings)
(OUTPUT / "run_settings.json").write_text(json.dumps(settings, indent=2))
with (OUTPUT / "api.log").open("w") as log:
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"],
        cwd=PROJECT, env=environment, stdout=log, stderr=subprocess.STDOUT,
    )
print("API PID:", server.pid, "Log:", OUTPUT / "api.log")
```

Do not rerun this cell while the server is alive. Stop it with the final cell
first. The first startup downloads the base model; allow several minutes.

## 3. Wait for readiness and verify the backend

```python
import time
import httpx

BASE_URL = "http://127.0.0.1:8000"
deadline = time.monotonic() + 900
with httpx.Client(timeout=5) as client:
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"API exited with {server.returncode}; inspect {OUTPUT / 'api.log'}.")
        try:
            ready = client.get(BASE_URL + "/ready")
            if ready.status_code == 200:
                break
        except httpx.TransportError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Startup exceeded 15 minutes; inspect {OUTPUT / 'api.log'}.")
        time.sleep(3)
    version = client.get(BASE_URL + "/version")
    version.raise_for_status()
    version_data = version.json()
assert version_data["backend"] == "qwen"
assert version_data["adapter_revision"] == ADAPTER_REVISION
(OUTPUT / "version.json").write_text(json.dumps(version_data, indent=2))
print(version_data)
```

## 4. Submit a PDF and save evidence

Start with the synthetic sample, then change `PDF_PATH` to your attached PDF.
`force_refresh=true` bypasses the extraction cache. The API invokes its actual
rendering, parsing, grounding, merge, and sanity-check path. Verified text
presence is still not a guarantee of semantic correctness; inspect the values
and `_verification` flags rather than treating HTTP 200 as an accuracy score.

```python
import time
import httpx

PDF_PATH = PROJECT / "samples/receipt_ok.pdf"
# Example replacement: Path("/kaggle/input/your-private-dataset/dmart.pdf")
started = time.perf_counter()
with PDF_PATH.open("rb") as pdf, httpx.Client(timeout=1500) as client:
    response = client.post(
        BASE_URL + "/extract", data={"force_refresh": "true"},
        files={"file": (PDF_PATH.name, pdf, "application/pdf")},
    )
capture = {
    "filename": PDF_PATH.name,
    "status": response.status_code,
    "elapsed_s": round(time.perf_counter() - started, 3),
    "headers": {key: value for key, value in response.headers.items() if key.lower().startswith("x-")},
    "body": response.text,
}
(OUTPUT / "extraction_capture.json").write_text(json.dumps(capture, indent=2))
print(json.dumps(capture, indent=2))
response.raise_for_status()
(OUTPUT / "receipt.json").write_text(json.dumps(response.json(), indent=2))
```

This cell overwrites the previous capture on rerun: download it first or use
distinct output filenames for multiple receipts. Outputs may contain personal
data; keep them private. Do not include the input PDF in a public repository
without permission.

## 5. Capture concurrent requests

Run after the preceding request has completed, initially with the bundled
one-page sample. This sends five cache-bypassed requests. The configured queue
wait is 60 seconds, so slow GPU runs may legitimately produce 503 responses;
record them instead of calling the run an all-success pass. The script's HTTP
timeout is 180 seconds per request; client timeouts remain failures to report.

For a separate five-completion experiment at roughly 25 seconds per request,
set `QUEUE_TIMEOUT_S` to `"120"` in the API-start cell and restart the server.
This changes queue waiting, not total request duration; it does not guarantee
five successes on slower inputs. Keep the 60-second load-shedding experiment
as separate evidence. Before each rerun, download the entire reports folder or
copy it to a uniquely named run folder, including `run_settings.json`,
`version.json`, response captures, and server log. Do not pair captures with
settings from a different run.

```python
probe = subprocess.run(
    [sys.executable, "scripts/concurrency_test.py", "--base-url", BASE_URL,
     "--pdf", str(PROJECT / "samples/receipt_ok.pdf"),
     "--out", str(OUTPUT / "concurrency_output.txt")],
    cwd=PROJECT, text=True, capture_output=True,
)
(OUTPUT / "concurrency_console.txt").write_text(probe.stdout + probe.stderr)
print(probe.stdout)
if probe.returncode:
    print(probe.stderr)
    raise RuntimeError("Concurrency probe failed; retain the console capture and API log.")
```

`max_active_requests` counts overlapping **client HTTP requests**, not GPU
forward passes. It is not by itself proof of single-flight GPU execution.
Review timings, server logs, and the single-flight implementation/tests before
making that claim. A successful notebook run does not validate the Docker
image, Redis integration, public hosting, or the local UI's real-model path.

## 6. Stop the server

```python
server.terminate()
try:
    server.wait(timeout=40)
except subprocess.TimeoutExpired:
    server.kill()
    server.wait()
print("API stopped. Download evidence from:", OUTPUT)
```

The bundle was checked locally with the dummy backend, and supplied Kaggle
captures now record real-model API execution plus completion/timeout behavior.
Those observations are not a guarantee for a new environment or document.
Keep each new run's version, settings, response, timings, logs, and concurrency
capture together as evidence. See the repository evaluation report for the
current results and their provenance limitations.

## Diagnose an all-null real PDF result

HTTP 200 means the response schema is valid, not that extraction succeeded.
An empty model prediction, two failed parses, or absent/nonmatching grounding
text can all produce empty output. The updated pipeline preserves `unverified`
when OCR supplies no evidence and reads OCR once per page. It does not retain
unsupported model values or relax grounding.

The supplied DMart diagnostic confirmed that generation and parsing succeeded,
but the PDF had no text layer and Tesseract returned only whitespace. Grounding
therefore removed the predictions. The date was independently nulled because
the model returned a non-ISO string.

The OCR adapter now retains successful default recognition and retries empty
or failed recognition on a grayscale, contrast-adjusted copy with block
(`--psm 6`) and sparse-text (`--psm 11`) layouts. The retry image is scaled by
at most 3x and capped at a 3,000-pixel long edge; this is interpolation of the
existing raster, not higher-resolution PDF rendering. VLM input size is
unchanged. Each Tesseract attempt has a 15-second subprocess timeout, with
at most three attempts (45 seconds of OCR timeouts plus preprocessing/overhead).
Missing Tesseract still produces empty evidence; grounding remains strict.
The follow-up diagnostic recovered 729 characters, but the OCR remained noisy:
none of the seven predicted item names matched and the predicted total was
absent. It also exposed a false verification of zero tax from split-decimal
`00` fragments. Nonempty OCR alone does not establish successful extraction.

The latest update matches complete numeric tokens and joins OCR-split decimals
before matching. A predicted zero must match an actual zero, not a small amount
within the general numeric tolerance. Dates accept ISO or explicitly day-first
slash, hyphen, or dot formats; `04/12/2018` means 4 December. There is no
month-first fallback, and date grounding follows the same policy. Numeric
grounding remains document-wide rather than tax-label-aware.

A local replay of the saved model output and OCR now grounds the date and
removes the unsupported zero tax. Items and total remain empty. This was not a
new GPU or Tesseract run; the original PDF is still needed to test genuinely
higher-resolution OCR rendering. No successful DMart item extraction is claimed.

To apply the fix, stop the API first (Step 6). Attach
`kaggle-diagnostics-update.zip` as a private Kaggle dataset; it contains the
four changed application modules and `scripts/diagnose_pdf.py`. Kaggle may
unpack it automatically. Apply the update to the existing working bundle:

```python
import os
import shutil
import zipfile
from pathlib import Path

assert "server" not in globals() or server.poll() is not None, "Stop the API before updating or loading another model."
PROJECT = Path("/kaggle/working/kaggle-api")
assert (PROJECT / "app/main.py").is_file(), "Complete the bundle setup first."
UPDATE_PATH = ""
relative_files = (
        "app/application/extract_receipt.py",
        "app/domain/grounding.py",
        "app/infrastructure/inference/structured.py",
        "app/infrastructure/ocr/tesseract.py",
        "scripts/diagnose_pdf.py",
)
if not UPDATE_PATH:
        roots = [path.parent.parent for path in Path("/kaggle/input").rglob("diagnose_pdf.py")
                         if (path.parent.parent / "app/domain/grounding.py").is_file()]
        candidates = roots or list(Path("/kaggle/input").rglob("kaggle-diagnostics-update.zip"))
        if len(candidates) != 1:
                print(candidates)
                raise ValueError("Set UPDATE_PATH to the exact update folder or ZIP.")
        update = candidates[0]
else:
        update = Path(UPDATE_PATH)
if update.is_dir():
        for relative in relative_files:
                shutil.copy2(update / relative, PROJECT / relative)
else:
        with zipfile.ZipFile(update) as archive:
                assert set(archive.namelist()) == set(relative_files), "Unexpected update contents."
                archive.extractall(PROJECT)
print("Updated runtime files; keep the API stopped for the diagnostic run.")
```

Run one diagnostic using the API's saved settings and adapter snapshot. The
command loads one model and exits afterward; do not run it alongside the API
or a standalone notebook model. It uses the same rendering, generation, parsing,
grounding, merge, and sanity logic, without HTTP or shared-cache behavior.

```python
import json
import subprocess
import sys
from datetime import datetime, timezone

PDF_PATH = Path("/kaggle/input/datasets/maheshnallada/dmart-pdf/dmart.pdf")
settings = json.loads((PROJECT / "reports/run_settings.json").read_text())
environment = os.environ.copy()
environment.update({key: str(value) for key, value in settings.items()})
diagnostic_path = PROJECT / "diagnostics" / f"dmart-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
subprocess.run(
        [sys.executable, "-m", "scripts.diagnose_pdf", "--pdf", str(PDF_PATH),
         "--out", str(diagnostic_path)],
        cwd=PROJECT, env=environment, check=True,
)
print("Private diagnostic:", diagnostic_path)
```

The output file is created with owner-only permissions and never overwrites an
existing file. The default `diagnostics/` directory is Git-ignored in this
repository. It contains personal receipt text and raw model outputs: do not
commit, publish, or send it to telemetry. Inspect locally and share only a
redacted excerpt. Diagnostic text must not be added to public UI evidence.

- `attempts`: raw first/repair generations, parsed fields, and detailed schema
    errors. If both attempts fail, the service falls back to an empty receipt.
- `pages`: rendered dimensions, PDF text, and whether grounding chose PDF text
    or OCR. A nonempty PDF text layer currently suppresses OCR, even if irrelevant.
- `ocr_reads`: actual OCR text; empty text can mean unavailable or unsuccessful
    OCR. Also inspect any Tesseract warnings in the command output.
- `grounding`: values before and after each pass, alongside the exact evidence
    text. Nonempty but mismatching text removes unsupported values and item names.
- `result`: final receipt, or `error` if diagnostic execution failed.

After diagnosis, restart the API to load the updated code and submit the PDF
with `force_refresh=true`; an older cached all-null response would otherwise
hide the change. Do not claim the DMart extraction is fixed until a new real
run demonstrates useful, grounded values.