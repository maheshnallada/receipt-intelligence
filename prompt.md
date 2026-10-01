You are a senior AI engineer. Build this take-home end to end in the current repository (/Users/uma.maheswara/AgenticAI/FineTuning_VLM). Read AI_Engineer_Assignment_VLM_Extraction_Assignment.pdf and implement every required deliverable, then add the production layers listed below. A smaller system that runs and is measured beats an ambitious system that does not run. Be honest about limitations in the README and feasibility report.

# Non-negotiable assignment constraints

- Extraction model: an open-source VLM with 4B parameters or fewer. Default: Qwen2.5-VL-3B-Instruct with QLoRA (4-bit) so it fits a free Colab/Kaggle T4 (16 GB). Justify the choice and the image resolution (accuracy vs GPU memory) in the README. Alternatives you may switch to if Qwen2.5-VL-3B does not fit: SmolVLM, PaliGemma 2, Florence-2. Do not exceed 4B.
- Do NOT use paid LLM APIs (GPT-4o, Claude, Gemini, etc.) as the extraction model. If any paid API is used only for data preparation, document that in the README. Prefer zero paid APIs.
- Dataset: CORD v2 from Hugging Face (naver-clova-ix/cord-v2). You may use SROIE or DocILE only if you explain why. Keep a held-out test split that is never used for training or prompt tuning.
- Fine-tune with LoRA or QLoRA so the model emits the schema below from page images. Training must be a notebook or script that can run on a T4, plus instructions to push the LoRA adapter to Hugging Face (use a placeholder repo id via env, do not invent credentials).
- Report zero-shot vs fine-tuned on the same test set.
- Every API response is valid JSON that passes Pydantic validation. JSON validity rate target: 100% after repair-and-revalidate. If the model output cannot be repaired into the schema, return a schema-valid response with nulls and verification flags, never invalid JSON and never a guessed value.
- Ground every non-null value against the document (PDF text layer first, lightweight OCR such as Tesseract on a crop or full page as fallback). Unverified values become null or are marked unverified. Never return a value that is not actually in the document. When unsure, return null or flag the field.
- Sanity checks: line items should sum to subtotal; subtotal + tax should equal total (tolerance for rounding). Failures flip the affected fields to unverified or null according to a documented policy.
- POST /extract accepts a PDF and returns the JSON in the same HTTP response. Multi-page PDFs merge into one result.
- Load the model once at startup.
- Only one inference job may run on the GPU at a time. Policy: a bounded in-process queue (default depth 8). If the queue is full, return 429. If a waiter exceeds QUEUE_TIMEOUT_S, return 503. Document why (GPU memory; a second concurrent forward pass OOMs a 16 GB T4).
- Include a script that sends 5 parallel requests and saves captured output.
- Reject bad input with clear errors: file too large (413), too many pages (422), corrupt PDF (400), password-protected PDF (400), unsupported type (415), inference timeout (504).
- Provide a Dockerfile.
- Write an evaluation report (metric tables, zero-shot vs fine-tuned) and a 1–2 page feasibility report backed by real numbers: accuracy, latency per page, peak GPU memory, estimated cost per 1,000 pages. Describe failures: dense tables, poor scans, rotation, small fonts, multi-page documents. Optional: a short comparison note vs an OCR + text-only pipeline, only if you actually run it.
- README: setup, design decisions, trade-offs, how to run training, API, UI, eval, and the concurrency test.

Output schema (exact). Dates are ISO YYYY-MM-DD. Numbers are JSON numbers, not strings.

{
  "store_name": string | null,
  "date": string | null,
  "line_items": [
    { "name": string, "qty": number | null, "unit_price": number | null, "amount": number | null }
  ],
  "subtotal": number | null,
  "tax": number | null,
  "total": number | null,
  "_verification": { "<field_path>": "verified" | "unverified" | "not_found" }
}

_verification must cover store_name, date, each line item field, subtotal, tax, and total.

# What “done” means

1. `docker compose up` starts API + UI + Redis. `/ready` is 200 only after the model is loaded (a tiny CPU dummy mode must exist so CI and machines without a GPU can boot: `MODEL_BACKEND=dummy` returns schema-valid fixtures and still runs grounding, queue, cache, and validation).
2. UI is a receipt desk, not a JSON playground. A reviewer can upload PDFs into an inbox, open a receipt, review fields against the page image, see why totals fail, approve or send back to review, and read a session summary. Raw JSON is a collapsed inspector, not the primary screen.
3. `notebooks/01_train_and_eval.ipynb` trains (or, if no GPU, documents the exact Colab steps and still contains the full training code) and writes eval tables.
4. `notebooks/02_api_test.ipynb` exercises the live API: health, happy path, each error case, cache hit, and points at the concurrency script.
5. `scripts/concurrency_test.py` fires 5 parallel extracts and writes `reports/concurrency_output.txt`.
6. `scripts/evaluate.py` writes `reports/evaluation.md` and `reports/metrics.json` with the metrics below.
7. `reports/feasibility.md` is 1–2 pages and cites numbers produced by the eval script, with a clear “fill after GPU run” path if this machine has no GPU.
8. Unit tests pass without a GPU, with Logfire send disabled and wandb disabled.
9. With `LOGFIRE_TOKEN` set, one `/extract` produces a Logfire trace with the pipeline child spans. With the token unset, the same request still returns JSON and logs to stdout.
10. With `WANDB_API_KEY` set, training and `scripts/evaluate.py` each create a wandb run. With the key unset, both still finish and write the report files.

# Backend architecture (domain-driven design)

One bounded context: receipt extraction. Model the language of that context (receipt, line item, verification, grounding, document page). Do not add a second bounded context, event sourcing, CQRS, a generic repository base class, or a dependency-injection framework. `app/main.py` is the composition root and the only place that wires concrete adapters to ports.

Dependency rule, enforced by imports:

- `domain` imports only the standard library and its own modules. It does not import FastAPI, Pydantic, Redis, PyMuPDF, torch, Logfire, or anything under `application`, `infrastructure`, or `interfaces`.
- `application` imports `domain` only. It orchestrates one use case. It does not import FastAPI or concrete adapters.
- `infrastructure` implements the domain ports (PDF renderer, VLM, OCR, cache, GPU gate, Logfire). It may import `domain`.
- `interfaces` translates HTTP to the use case and domain exceptions to status codes. Routes contain no grounding, merge, sanity, or cache logic.
- Tests for domain behavior import `domain` only.

Tactical model:

- `Receipt` is the aggregate root. `LineItem` is an entity inside it. `Money`, `IsoDate`, and `Verification` (`verified` | `unverified` | `not_found`) are value objects. Constructing a `Receipt` enforces the assignment schema: ISO dates, numeric money, and a verification entry for every leaf field.
- `UploadedDocument` and `Page` are value objects for the input (bytes hash, page count, image, text layer). Limits (size, page cap, content type) are checked by the document factory and raise domain errors.
- Domain services, pure and side-effect free: `ground_receipt` (normalize and mark each non-null value), `check_totals` (line items vs subtotal, subtotal + tax vs total), `merge_pages` (concatenate line items; pick store name, date, and totals from the page with the strongest grounding). They return a new `Receipt`. Document the null-out policy and the merge rule next to that code.
- Ports (typing.Protocol in `domain/ports.py`): `PageRenderer`, `ReceiptExtractor`, `TextReader`, `ExtractionCache`, `GpuSlot`. The use case depends on these protocols, not on Qwen, Redis, or PyMuPDF.
- Application service `ExtractReceipt` is the only orchestration: validate document, cache lookup, acquire GPU slot, render, extract, parse, repair once, merge, ground, sanity-check, cache write, release the slot. The pipeline section below is the body of this use case.
- The public JSON contract stays the assignment schema. Pydantic models live in `interfaces/http/schemas.py` and map to and from `Receipt`. They are the HTTP representation, not the domain model.

# Repository layout

vlm-extraction/
  README.md
  pyproject.toml or requirements.txt (pin versions)
  .env.example
  Dockerfile
  docker-compose.yml
  Makefile
  app/
    main.py                              # composition root and FastAPI factory
    config.py                            # pydantic-settings, env only
    domain/
      receipt.py                         # Receipt aggregate, LineItem, Money, IsoDate, Verification
      document.py                        # UploadedDocument, Page; size and page-count rules
      errors.py                          # domain exceptions
      ports.py                           # PageRenderer, ReceiptExtractor, TextReader, ExtractionCache, GpuSlot
      grounding.py                       # pure grounding and normalization
      sanity.py                          # pure total checks
      merge.py                           # pure multi-page merge
    application/
      extract_receipt.py                 # ExtractReceipt use case
    infrastructure/
      pdf/renderer.py                    # PyMuPDF or pypdfium2; password and corrupt detection
      inference/qwen.py                  # model singleton, temperature 0, timeout
      inference/dummy.py                 # MODEL_BACKEND=dummy
      inference/json_repair.py           # one repair generation inside the same GPU slot
      ocr/tesseract.py                   # text-layer fallback
      cache/memory.py                    # LRU when REDIS_URL is empty
      cache/redis.py                     # Redis adapter
      gpu/single_flight.py               # bounded queue, one GPU job
      observability/logfire_setup.py     # Logfire init, spans, scrubbing; stdout fallback
    interfaces/
      http/routes.py                     # /extract, /health, /ready, /version
      http/schemas.py                    # Pydantic HTTP contract, maps Receipt
      http/errors.py                     # domain error -> stable JSON error
  ui/                        # Vite + React + TypeScript
  notebooks/
    01_train_and_eval.ipynb
    02_api_test.ipynb
  training/
    prepare_cord.py          # CORD -> chat/JSON samples, frozen split
    train_qlora.py
    dataset_card.md
  eval/
    metrics.py
    run_eval.py
  scripts/
    concurrency_test.py
    evaluate.py
  tests/
    test_schema.py
    test_grounding.py
    test_sanity.py
    test_queue.py
    test_cache.py
    test_errors.py
    test_api_dummy.py
  reports/
    evaluation.md
    feasibility.md
    metrics.json
  samples/                   # 2–3 tiny synthetic receipt PDFs committed for tests and the UI

# Pipeline

`ExtractReceipt` performs these steps. The route only builds `UploadedDocument` from the upload and returns the mapped `Receipt`.

POST /extract (multipart file, optional form field force_refresh=false):

1. Validate content-type and size (MAX_UPLOAD_MB, default 10) and page count (MAX_PAGES, default 5) before any GPU work.
2. SHA-256 the bytes. Cache key = hash + adapter revision + schema version + prompt version. On hit, return the cached JSON with header X-Cache: HIT and do not take the GPU lock.
3. Acquire the single GPU slot (queue). Record queue wait on the request span.
4. Render pages to images at the chosen long-edge resolution (default 1008 or whatever you justify for the model). Also extract the PDF text layer per page.
5. Run the VLM once per page (or one tiled prompt if you document it). Instruct it to copy only text visible on the page and to use null when a field is absent. Temperature 0.
6. Parse JSON. If parse or schema validation fails, one repair retry: re-ask the model with the validation error, still inside the same GPU slot. Then validate again.
7. Merge pages: concatenate line items; store_name/date/totals from the page where grounding confidence is highest; document the merge rule.
8. Ground each non-null leaf against normalized document text (casefold, strip currency symbols and thousands separators, numeric tolerance). Status: verified, unverified, or not_found. Apply the null-out policy for unverified scalar fields and keep the raw candidate only inside logs, not in the client JSON.
9. Run sum checks. On failure, mark the inconsistent fields unverified.
10. Store only schema-valid successes in the cache (TTL_S default 24h). Do not cache 4xx.
11. Release the GPU slot in a finally block.
12. Return the JSON plus headers: X-Request-Id, X-Cache, X-Queue-Wait-Ms, X-Inference-Ms.

Error body shape, always JSON:

{ "error": { "code": "PDF_PASSWORD_PROTECTED", "message": "...", "request_id": "..." } }

# Caching

- Interface with two backends: Redis (docker-compose service) and an in-process LRU (cachetools, max 128) used when REDIS_URL is empty.
- Include adapter id and schema version in the key so a new model does not serve stale JSON.
- force_refresh=true bypasses read but still writes.
- Record cache hit/miss as a Logfire span attribute (and on the X-Cache header).

# Retries

- Inference: at most 1 retry on transient runtime errors (CUDA OOM after empty_cache, incomplete generation). Same queue slot. Then 504.
- JSON repair: at most 1 extra generation, recorded as a Logfire span event `json_repair`.
- Do not retry 4xx. The UI retries only GET /health and /ready, not POST /extract, unless the person clicks Retry on a failed upload.
- Outbound Redis calls: 2 retries with short jitter, then fall back to memory and log a warning.

# Error handling

- Domain code raises domain exceptions (`PasswordProtected`, `CorruptDocument`, `TooManyPages`, `FileTooLarge`, `UnsupportedDocument`, `InferenceTimeout`, `GpuBusy`, `QueueFull`). `interfaces/http/errors.py` maps them to the HTTP status codes and the error JSON below. No stack traces in HTTP bodies. Stack traces go to structured logs.
- Timeouts: INFERENCE_TIMEOUT_S per page (default 120), applied with a worker-side deadline.
- Corrupt, truncated, and encrypted PDFs are detected before render.
- Shutdown: stop accepting /extract, drain the queue up to 30s, then cancel.

# Observability (Logfire) and experiment tracking (Weights & Biases)

Use these two tools only in the roles below. They are feasible here. They are not hard dependencies: the service, tests, training, and eval must still finish when the tokens are unset.

## Why these are feasible

- Logfire is the service observability tool. The API is FastAPI with Pydantic models, which Logfire instruments directly (`logfire.instrument_fastapi`, `logfire.instrument_pydantic`). Pipeline stages are ordinary functions, so manual spans fit without a second tracing stack. Logfire sits on OpenTelemetry, so one SDK covers traces, logs, and metrics. Do not also run Prometheus, Grafana, or a hand-rolled OTLP exporter.
- Weights & Biases is the training and evaluation experiment tracker. Hugging Face Trainer and TRL already support `report_to="wandb"`. Use it for runs, hyperparameters, loss, GPU memory, throughput, and the zero-shot vs fine-tuned metric table. Do not use wandb to trace HTTP requests. That is Logfire's job.

## Logfire (API process)

- Configure in `app/infrastructure/observability/logfire_setup.py`, called once from the composition root before routes are hit. Span helpers may wrap infrastructure adapters. Do not import Logfire from `domain` or `application`. The use case stays free of the SDK; time the stages in infrastructure and record the attributes there.
- If `LOGFIRE_TOKEN` is set, send to Logfire (`service_name=vlm-extraction`). If it is unset, call `logfire.configure(send_to_logfire=False)` and also write JSON logs to stdout. Startup must succeed when Logfire is unreachable.
- Tests and CI set `LOGFIRE_SEND_TO_LOGFIRE=false` (or leave the token unset). Do not make network calls to Logfire from pytest.
- Instrument FastAPI and Pydantic. One server span per `/extract`, with child spans: `queue.wait`, `pdf.render`, `vlm.infer`, `json.parse`, `ground`, `sanity`, `cache.write`.
- Span attributes: request_id, http status, page count, cache hit/miss, queue_wait_ms, inference_ms, backend, adapter revision, counts of verified / unverified / not_found, error code on failure.
- Scrub content. Do not put PDF bytes, page images, document text, raw model output, or extracted field values on spans or logs. Receipts are sensitive. Log counts and timings only.
- Middleware assigns request_id (honor incoming X-Request-Id if it is a UUID) and bind it on the Logfire span.
- Query latency p50/p95, queue depth, cache hit rate, and verification counts from the Logfire UI when a token is present. The eval script and concurrency script still compute latency locally from timed HTTP calls so the reports exist with no Logfire account.
- GET /health is liveness (process up). GET /ready is readiness (model loaded or dummy backend ready). Do not add a Prometheus `/metrics` endpoint or a Grafana compose profile.

## Weights & Biases (training and eval only)

- `training/train_qlora.py` and `notebooks/01_train_and_eval.ipynb`: `wandb.init` once per run (project from `WANDB_PROJECT`, default `vlm-receipt-extraction`). Log model id, LoRA rank/alpha/dropout, bits, image resolution, train/val sizes, learning rate, and the frozen split hash.
- During training, rely on Trainer/TRL `report_to="wandb"` for loss, learning rate, grad norm, and samples per second. Also log peak GPU memory when CUDA is available.
- After zero-shot and after fine-tuning, `wandb.log` the same metric dict the report uses (field F1, exact match, JSON validity, hallucination rate before and after grounding, latency, peak GPU memory), with `config` key `stage=zero-shot` or `stage=fine-tuned`. Log a `wandb.Table` comparing the two stages.
- `scripts/evaluate.py` opens one wandb run when a key is present and logs that same table, then still writes `reports/evaluation.md` and `reports/metrics.json`. The files are the submission artifact. The wandb run is the live trace.
- If `WANDB_API_KEY` is unset, set `WANDB_MODE=disabled` and continue. Training and eval must not fail. Support `WANDB_MODE=offline` when the user wants a local run synced later.
- Do not upload PDF bytes, the full CORD dump, or LoRA weights to wandb. The Hugging Face Hub is the adapter store. Do not log raw receipt images unless `WANDB_LOG_SAMPLES=true`, and then at most a handful of training previews.

# API surface

- POST /extract
- GET /health
- GET /ready
- GET /version (model id, adapter revision, schema version, backend)
- CORS for the UI origin from env.

# UI: receipt desk (Vite + React + TypeScript)

Do not ship a page whose main job is “upload a PDF and dump JSON.” The product is a **receipt desk**: an expense inbox a clerk uses to review extracted receipts and approve only what they trust. The assignment JSON is the payload. The desk is the use case.

The only backend call for extraction remains `POST /extract`. Approve, status, and the session list live in the browser for this take-home. Do not add accounts, login, a second model, or a receipts database.

## Session model (client only)

Each successful extract becomes a desk item:

- `id`, file name, page-image preview (object URL from the first rendered page if the API returns one; otherwise a local PDF.js or pdfjs-dist first-page render of the uploaded file)
- extracted `Receipt` JSON
- `status`: `needs_review` | `ready` | `approved`
- request meta: request id, cache hit/miss, queue wait, inference time, HTTP status

Status rules, computed from `_verification` and the same sum checks the API already ran:

- `needs_review` if any field is `unverified` or `not_found`, or if line items do not sum to subtotal, or subtotal + tax does not equal total (use the same rounding tolerance as the domain).
- `ready` if every required field (`store_name`, `date`, `total`) is `verified` and the sum checks pass.
- `approved` only after the person clicks Approve. Approving does not call the model again.

A person may still approve a `needs_review` item after they explicitly accept the unverified fields (a confirm checkbox: “I accept the unverified fields”). Unverified money fields stay blank in the form until that happens or until the person types a value they read from the page. Typed overrides stay in client state only and are tagged `clerk_override` in the local item, not sent back to the model.

## Screens

**Inbox.** Default view. Cards for each item: store, date, total, status chip. Sort: needs review first, then ready, then approved. Empty state before the first upload: one sentence on what the desk does, a drop zone, and “Use sample receipt.”

**Summary strip.** Always visible: count of receipts in this session, sum of approved totals, count still needing review.

**Upload.** Drag-and-drop or file picker, PDF only, shows name and size. Disable submit while in flight. Progress: elapsed time, and “queued” vs “running” if the API exposes it. Error banner from the API error JSON, with specific copy for 413, 422, 400, 429, 503, 504. Retry resubmits the same file. A failed upload does not create an inbox card.

**Review.** Opening a card shows the page image on one side and the fields on the other (stacked on ~390px). Each field has a verification chip (`verified` / `unverified` / `not_found`). Line items are a table. A banner states a failed check in plain language, for example “Line items do not add up to the subtotal.” Approve is enabled when the item is `ready`, or when the person has checked “I accept the unverified fields.” After approve, the card moves to approved and the summary strip updates.

**Inspector.** Collapsed “Technical details” panel: raw JSON with a copy button, plus the request meta line. Reviewers should not need this to complete the flow.

## Constraints

- Responsive at desktop width and ~390px. Accessible labels, visible focus, sufficient contrast. No new design system beyond a small amount of CSS.
- Serve the built UI from Nginx in Docker, proxying /api to the FastAPI service. Vite dev server proxy for local dev.
- Session state is lost on refresh. Say so in the empty state. That is acceptable for the take-home.

# Notebooks

notebooks/01_train_and_eval.ipynb

- Markdown cells explain data, split, model choice, resolution, LoRA config, and metrics.
- Code cells, top to bottom: install notes, load CORD v2, build a frozen train/val/test split (save split ids to data/splits.json), convert labels into the target schema, init wandb (disabled when no API key), zero-shot eval on a small test subset, QLoRA training (Unsloth if it supports the chosen model, otherwise peft + trl + bitsandbytes, `report_to="wandb"` only when wandb is enabled), fine-tuned eval on the same subset, log the comparison table to wandb, save adapter, optional push to Hub behind an env flag, write reports/metrics.json.
- Guard heavy cells with a clear “needs GPU” check so the notebook opens and runs the CPU cells anywhere.
- Do not leak the test set into training.

notebooks/02_api_test.ipynb

- Configurable API_BASE_URL.
- Wait for /ready.
- Upload each sample PDF, assert schema, print verification.
- Assert error cases using generated bad files (empty, huge, not-a-pdf).
- Call the same file twice and show the second response is a cache hit.
- Leave a cell that shells out to scripts/concurrency_test.py and displays the output.
- Narrative markdown between cells so a reviewer can run it as the demo.

# Evaluation

Metrics, zero-shot and fine-tuned, same test ids:

- Field-level precision, recall, F1 for store_name, date, subtotal, tax, total, and line items (match line items by normalized name, then score qty, unit_price, amount).
- Exact match rate per field and for the full receipt (line items as a multiset).
- JSON validity rate.
- Hallucination rate = non-null predictions whose normalized value is absent from the document text, before grounding and after grounding.
- Latency p50 and p95 per page.
- Peak GPU memory (torch.cuda.max_memory_allocated; record “n/a” on CPU).

scripts/evaluate.py must be rerunnable and overwrite reports/evaluation.md and reports/metrics.json. The markdown report is a table, not a screenshot. When WANDB_API_KEY is set, also log that table to a wandb run. When it is unset, skip wandb and still write the files.

# Feasibility report

reports/feasibility.md, 1–2 pages, using the measured numbers:

- Is a small VLM a good fit for production extraction? Answer with accuracy, latency per page, peak GPU memory, and a cost estimate per 1,000 pages (state the GPU hourly price you assumed and the batch-1 throughput you measured).
- Where it fails: dense tables, poor scans, rotation, small fonts, multi-page receipts.
- Queue and timeout policy and what breaks under load.
- What you would do next if this went to production (do not implement a second architecture).

# Deployment

- Dockerfile: CUDA runtime base only for the API image; CPU-only target via build arg so dummy mode builds without a GPU. Model weights are downloaded at startup from HF (cache dir via env), not baked into the image.
- docker-compose.yml services: api, ui, redis. Pass LOGFIRE_TOKEN through env. Do not add Prometheus or Grafana. Logfire is the observability backend.
- .env.example for every setting, including LOGFIRE_TOKEN, LOGFIRE_SEND_TO_LOGFIRE, WANDB_API_KEY, WANDB_PROJECT, WANDB_MODE, WANDB_LOG_SAMPLES. Empty tokens are valid.
- Makefile targets: install, test, lint, up, down, eval, concurrency, notebook-check.
- GitHub Actions (or a documented local equivalent): ruff + pytest on dummy backend, no GPU.
- Log to stdout and to Logfire when a token is set. No secrets in the image. HF token, LOGFIRE_TOKEN, and WANDB_API_KEY only via env. Never commit them.

# Tests

pytest, no network, no GPU, Logfire send disabled, wandb disabled. Domain tests must not import FastAPI, Redis, or torch.

- Receipt aggregate accepts a valid example and rejects a bad date, a wrong money type, and a missing verification leaf.
- HTTP schema maps that aggregate to the assignment JSON and back.
- Grounding: value present, value absent, numeric formatting differences (1,200.50 vs 1200.5).
- Sanity: matching sums stay verified; mismatched sums become unverified.
- Merge: line items concatenate; totals come from the better-grounded page.
- GPU slot: 5 concurrent callers, only one runs at a time, overflow raises `QueueFull` (HTTP 429 at the edge).
- Cache port: second identical key hits; force_refresh misses; different schema version misses. Test the memory adapter.
- API dummy mode, through the use case: sample PDF returns 200 valid JSON; password, corrupt, too many pages, and oversize return the documented codes.
- An import check fails if `app.domain` imports infrastructure or interfaces.
- Concurrency script is importable and its summary parser can read a fixture.

# Implementation order

1. Domain model, ports, domain errors, grounding, sanity, merge, and their tests. Then the memory cache and GPU slot adapters.
2. `ExtractReceipt` with the PDF renderer and dummy extractor, wired in the composition root, so /extract works without a GPU.
3. Qwen extractor behind MODEL_BACKEND=qwen, still one slot, loaded once from the composition root.
4. Receipt desk UI against dummy mode (inbox, review, approve, summary strip), then the same UI against the real model.
5. Training script and notebook, eval script, reports.
6. Dockerfile, compose, Logfire wiring, concurrency script.
7. README and feasibility report filled from whatever numbers this machine can produce. If there is no GPU, run dummy latency numbers only for the API path and label VLM metrics as pending a T4 run, with the exact command to produce them.

# Code quality

- Python 3.11, type hints on public functions, ruff clean.
- Small modules. Business rules live in the domain. The use case orchestrates ports. Route handlers translate HTTP.
- Config only through environment variables, read in the composition root and passed into adapters. Domain objects do not read the environment.
- Comments only where a trade-off is not obvious (queue policy, null-out policy, merge rule, resolution).
- Do not commit data dumps, model weights, .env, HF tokens, Logfire tokens, or wandb keys. Add `wandb/` to .gitignore.

# README sections

- What this is
- Architecture: the receipt-extraction bounded context, the four layers, and the dependency rule
- Model choice and resolution trade-off
- Grounding and hallucination policy
- Single-flight queue policy
- Quick start (local dummy, Docker, Colab training)
- API examples
- UI: the receipt desk (inbox, review, approve, what stays in the browser)
- Evaluation and how to reproduce the tables
- Observability (Logfire: what is traced, how to run with no token) and experiment tracking (wandb: what each run logs, offline/disabled mode)
- Limitations
- Submission checklist: GitHub structure, HF adapter link placeholder, eval report, feasibility report, concurrency output, notebooks

Start implementing now. After each phase, run the relevant tests and fix failures before moving on. Finish with the commands you ran and what is left that requires a T4.