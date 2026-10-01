# vlm-extraction

A receipt desk: upload a PDF, a small open-source VLM reads the page images,
and the service returns **schema-valid JSON in the same HTTP response**. Every
non-null value is grounded against the PDF text layer (Tesseract fallback).
If the model cannot be repaired into the schema, the API still returns nulls
and verification flags — never invalid JSON, never a guessed value.

Hardware target: free Colab/Kaggle T4 (16 GB). This checkout runs end-to-end
in `MODEL_BACKEND=dummy` without a GPU.

## Architecture

One bounded context: receipt extraction. Four layers, one way:

| Layer | May import | Role |
|---|---|---|
| `app/domain` | stdlib only | `Receipt` aggregate, grounding, sanity, merge, ports |
| `app/application` | domain | `ExtractReceipt` use case |
| `app/infrastructure` | domain + drivers | PyMuPDF, Qwen/dummy, Redis/LRU, GPU slot, Logfire |
| `app/interfaces` | application + domain | FastAPI routes and Pydantic HTTP schema |

`app/main.py` is the composition root. Routes do not ground, merge, or cache.

## Model choice and resolution

**Qwen2.5-VL-3B-Instruct**, QLoRA 4-bit, language/attention/MLP LoRA only
(r=16, α=32, dropout=0.0). JSON-friendly and fits a T4
with one 1008-token image and one repair pass.

Vision tokens ≈ `pixels / (28×28)`. We render the long edge at **1008 px**
and set `max_pixels = 1008×28×28`. Higher (1280+) reads small type better
and OOMs sooner. CORD pages are already ~864×1296.

Fallback models if 3B does not load: SmolVLM, PaliGemma 2, Florence-2. Do
not exceed 4B. No paid extraction API.

## Grounding and hallucination

- Predicted value in normalized document text → keep, `verified`.
- Predicted value absent → **null**, `not_found` (candidate stays in logs).
- Empty text layer and failed OCR → **null**, `unverified`.
- Sum checks fail → keep the numbers (they are on the page), mark them
  `unverified`. Tolerance `0.05`.
- One JSON repair generation inside the same GPU slot, then a schema-valid
  all-null receipt.

Model dates accept ISO `YYYY-MM-DD` or day-first `D/M/YYYY`, `D-M-YYYY`,
and `D.M.YYYY`. For example, `04/12/2018` means 4 December, not April 12.
Invalid dates become null; there is no month-first fallback. Grounding follows
the same date policy.

Numeric grounding joins OCR-split decimals such as `10. 00` and matches complete
numeric tokens, not fragments inside identifiers, dates, or percentages. A
predicted zero requires an actual zero, not a nearby amount within tolerance.
Matching is still document-wide, not label-aware: a matching number alone does
not establish that it belongs to the predicted field.

## Single-flight queue

One GPU job is admitted at a time to limit memory pressure. In-process queue depth 8
(including the holder). Overflow → **429**. Wait > `QUEUE_TIMEOUT_S` → **503**.
Inference generation > `INFERENCE_TIMEOUT_S` → **504**. These are separate
limits, not a total-request deadline. Blocking extraction runs in a worker
thread while retaining the async GPU slot, including during cancellation.
Concurrent-forward OOM behavior has not been measured directly.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make samples
make test
MODEL_BACKEND=dummy uvicorn app.main:app --port 8000
```

UI (another terminal):

```bash
cd ui && npm install && npm run dev
```

Open http://localhost:5173. Session state is in the browser only.

Docker (dummy API + nginx UI + Redis):

```bash
docker compose up --build
```

UI at http://localhost:8080, API at http://localhost:8000. `/ready` is 200
only after the backend is loaded.

Colab training: open `notebooks/01_train_and_eval.ipynb` on a T4, or
`python training/train_qlora.py`. For a **single file** you can upload
alone (no repo imports), use `notebooks/03_full_pipeline.ipynb`. Push
the adapter with
`HF_PUSH=1 HF_ADAPTER_REPO=your-username/vlm-receipt-extraction-lora`.

Published adapter: [Mahesh-Nallada/vlm-receipt-extraction-lora](https://huggingface.co/Mahesh-Nallada/vlm-receipt-extraction-lora).
The captured T4 run used 200 training examples and restored checkpoint 15
after non-finite losses. A same-test-set comparison on 32 CORD receipts is
available in the evaluation and feasibility reports.
Future notebook and script runs default to `TRAIN_LR=1e-5`; set `TRAIN_LR`
or pass `--learning-rate` to override. This conservative setting is not yet
validated on T4.

Run the API with that adapter by setting `MODEL_BACKEND=qwen` and
`ADAPTER_PATH=Mahesh-Nallada/vlm-receipt-extraction-lora`. The base model is
selected separately with `MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct`.

## Uploaded PDF: real-model Kaggle evidence

[Final_Notebook_VLM.ipynb](notebooks/Final_Notebook_VLM.ipynb), Cells 59-61,
contains a saved Tesla T4 run using the published Hub adapter at revision
`e9d1eb7ff9a11273a87a5d27a9fec7ba35b4ee92`. On a one-page `dmart.pdf`, it
returned seven line items and a predicted total of 845.00. The notebook's
schema validation passed after one repair generation; elapsed page processing
time was 118.733 seconds, including the repair.

This is real model execution, not dummy output. It is a qualitative smoke test,
not a labeled accuracy evaluation: the date remains `04-12-2018` rather than
ISO format, the store field includes the address, and predicted tax 0.00 is
unverified. OCR grounding, arithmetic checks, and page merging were not applied.
See [the evaluation report](reports/evaluation.md) for the exact evidence and
limitations. The CORD benchmark and its latency figures remain separate.

For another PDF, use a fresh Kaggle GPU session with Internet enabled and run
the standalone PDF section near the end of that notebook, in order. Attach the
PDF via Add Input and set `PDF_PATH` if multiple PDFs are available. Do not run
the earlier training cells just to perform inference. Download results from
`/kaggle/working/pdf_predictions/`. Remove embedded credentials before sharing
or publishing a notebook; use Kaggle Secrets for authentication when needed.

The local UI still defaults to the dummy API. Hugging Face hosts the adapter
weights, not a prediction service. The separate API runs below do not establish
a public endpoint or end-to-end real-model UI validation.

## Real-model API and queue evidence

The actual FastAPI service ran inside a Kaggle GPU notebook with the pinned
adapter. The current [response capture](reports/extraction_capture%20%281%29.json)
records HTTP 200, cache MISS, and **25.897 seconds** for the synthetic
`receipt_ok.pdf`: total 13.50 and 13 verified field statuses. Schema, checked
sample text, and arithmetic agree. The earlier 31.619-second response remains
in [the notebook](notebooks/final-notebook-vlm.ipynb), Cells 69-73. A text-layer
synthetic receipt does not validate scanned-document OCR or general accuracy.

- [Timeout capture](reports/concurrency_output.txt): three 200/MISS responses
  with waits of 0.00, 24.76, and 49.49 seconds; two 503 responses around 60.03
  seconds, consistent with the accompanying 60-second queue setting.
- [Completion capture](reports/concurrency_completion.txt): five 200/MISS
  responses with increasing waits up to 100.61 seconds; last completion at
  125.99 seconds. Processing order is 2, 1, 5, 4, 3. This separately pasted run
  has no matching settings/version capture; the suggested 120-second queue
  setting is not verified and does not change the service's 60-second default.

These observations support serialized processing and timeout-based load
shedding after the event-loop fix. Client overlap is not GPU-forward
instrumentation, and the 503 response bodies were not saved. See
[evaluation](reports/evaluation.md) for provenance and remaining limitations.
The CORD benchmark and cost estimates remain unchanged. Reproduce the API
workflow using [the Kaggle guide](reports/kaggle-api-guide.md) with the minimal
backend bundle; preserve each run's captures and settings together before
changing configuration.

## API

For an all-null response, see the private diagnostic workflow in
[the Kaggle guide](reports/kaggle-api-guide.md#diagnose-an-all-null-real-pdf-result).
It captures raw generations, schema errors, and PDF/OCR evidence without
changing the response contract or disabling grounding. Such diagnostic output
contains receipt data and must not be published. After updating the backend,
restart it and use `force_refresh=true` to bypass older cached results.

Empty OCR results now trigger bounded block/sparse-layout retries on a
grayscale, contrast-adjusted image copy capped at a 3,000-pixel long edge.
The VLM image size is unchanged. At most three Tesseract calls run, each with
a 15-second timeout; persistent failure still returns no grounding evidence.
The DMart diagnostic confirmed empty OCR, but improved extraction on that PDF
requires a new run with the updated adapter.

`POST /extract` multipart `file`, optional `force_refresh`. Same-response JSON:

```json
{
  "store_name": "Oak & Ember Cafe",
  "date": "2026-03-15",
  "line_items": [{"name": "Latte", "qty": 2, "unit_price": 4.5, "amount": 9.0}],
  "subtotal": 12.5,
  "tax": 1.0,
  "total": 13.5,
  "_verification": {"store_name": "verified", "total": "verified"}
}
```

Headers: `X-Request-Id`, `X-Cache`, `X-Queue-Wait-Ms`, `X-Inference-Ms`.

Errors: `{ "error": { "code": "PDF_PASSWORD_PROTECTED", "message": "...", "request_id": "..." } }`

| Condition | Status | Code |
|---|---|---|
| Password / corrupt | 400 | `PDF_PASSWORD_PROTECTED` / `PDF_CORRUPT` |
| Too large | 413 | `FILE_TOO_LARGE` |
| Unsupported type | 415 | `UNSUPPORTED_MEDIA_TYPE` |
| Too many pages | 422 | `TOO_MANY_PAGES` |
| Queue full | 429 | `QUEUE_FULL` |
| Queue timeout | 503 | `QUEUE_TIMEOUT` |
| Inference timeout | 504 | `INFERENCE_TIMEOUT` |

`GET /health` liveness. `GET /ready` model/dummy ready. `GET /version`.

## UI: project story and receipt desk

The UI opens on a narrative landing page, with separate chapters for the model,
evaluation, feasibility, architecture, and PDF extraction:

- `#/story`: project overview and experiment narrative.
- `#/model`: base model, adapter, notebook settings, and service defaults.
- `#/evaluation`: zero-shot versus fine-tuned metrics, metric selection, and
  pre/post-grounding comparisons, plus separate saved API/queue evidence.
- `#/feasibility`: recorded latency and memory, plus an adjustable illustrative
  GPU cost estimate and selectable completion/timeout captures.
- `#/architecture`: interactive request-path diagram, also available directly
  at `/architecture.html`. Its editable specification is
  [reports/architecture.json](reports/architecture.json).
- `#/extract`: the receipt desk. Moving between chapters preserves the current
  browser session; refreshing does not.

The status strip reads `/api/version` and distinguishes synthetic dummy output
from the live backend. It is not a readiness check. Displayed service limits are
configuration defaults, not live telemetry. The landing image is rendered from
the project's synthetic sample PDF.

Evaluation downloads in `ui/public/evidence/` are unchanged snapshots of
`notebooks/full_comparison_pipeline.json` and `notebooks/sroie_header_metrics.json`.
Refresh those copies when publishing new evaluation runs. The UI reads its
comparison and cost inputs from the CORD snapshot; it does not run evaluation
or load model weights in the browser.

[api_validation.json](ui/public/evidence/api_validation.json) is a curated
runtime snapshot of the current response and both concurrency captures. Its
source paths and provenance distinguish recorded settings from unconfirmed
ones. Runtime results are not live telemetry and do not change the connected
backend status or the CORD-based cost calculation.

Inbox cards (store, date, total, status), a summary strip, upload with
elapsed time, and a review pane: page image vs fields with verification
chips. Raw JSON is a collapsed inspector. Approve stays in the browser.
Unverified money fields stay blank until the clerk types a value or checks
“I accept the unverified fields.”

## Evaluation

```bash
python scripts/evaluate.py
```

Writes `reports/evaluation.md` and `reports/metrics.json`. Dummy numbers are
measured on sample PDFs. The final notebook comparison evaluates zero-shot and
fine-tuned stages on the same 32 CORD test receipts with grounding. Treat the
model results as exploratory. With
`WANDB_API_KEY`, the same table is logged to wandb.

## Observability and experiment tracking

**Logfire** (API): `logfire.instrument_fastapi` + child spans
`queue.wait`, `pdf.render`, `vlm.infer`, `json.parse`, `ground`, `sanity`,
`cache.write`. Counts and timings only — no PDF bytes, images, document
text, raw model output, or field values. Unset `LOGFIRE_TOKEN` →
`send_to_logfire=False` and JSON logs on stdout.

**Weights & Biases** (train/eval only): hyperparameters, loss via
`report_to="wandb"`, zero-shot vs fine-tuned table. Unset key →
`WANDB_MODE=disabled`. `WANDB_MODE=offline` is supported. Do not upload
PDFs, the CORD dump, or LoRA weights.

## Limitations

- Notebook VLM metrics are a 32-example T4 slice, not a stable production estimate.
- The captured training run had non-finite loss after step 18 and restored checkpoint 15.
- CORD has no store/date labels, so those fields are weak on the official test set.
- No deskew, no constrained decoding, no vision-encoder LoRA (T4 headroom).
- PyMuPDF is AGPL. The public GitHub repo is compatible; a closed-source
  product needs a commercial license or a switch to pypdfium2.
- Session inbox is not persisted.

## Project resources

- [Hugging Face adapter](https://huggingface.co/Mahesh-Nallada/vlm-receipt-extraction-lora)
- [Evaluation report](reports/evaluation.md): same-test-set zero-shot vs. fine-tuned results
- [Feasibility report](reports/feasibility.md)
- [Timeout capture](reports/concurrency_output.txt) and [completion capture](reports/concurrency_completion.txt): recorded runs with provenance and instrumentation limitations documented

Outstanding validation: capture matching settings/version metadata for the
completion run, and validate Docker and the real-model UI path.
