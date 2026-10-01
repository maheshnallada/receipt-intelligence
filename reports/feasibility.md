# Feasibility: small VLM for receipt extraction

**Online resources:** [Deployed frontend](https://receipt-intelligence-chi.vercel.app)
(frontend only; no extraction API or GPU inference backend is deployed there) and
[Weights & Biases experiment report](https://forge.coreweave.com/wandb/maheshnallada/vlm-receipt-extraction/reports/Receipt-Intelligence--VmlldzoxODAzNzk0OA)
(may require sign-in or permission from the report owner).

[full_comparison_pipeline.json](../notebooks/full_comparison_pipeline.json).
The latest uploaded-PDF inference evidence is in
[final-notebook-vlm.ipynb](../notebooks/final-notebook-vlm.ipynb), Cells 59-61;
Cells 69-75 also record real-model FastAPI execution and concurrent requests.
The API response and runtime identity are saved in
[extraction_capture (1).json](extraction_capture%20%281%29.json) and [version.json](version.json).
The adapter is published at
[Mahesh-Nallada/vlm-receipt-extraction-lora](https://huggingface.co/Mahesh-Nallada/vlm-receipt-extraction-lora).
The current artifact contains a clean same-test-set comparison on 32 CORD
receipts plus a separate SROIE header probe.

## Is a 3B VLM a good fit?

A 4-bit Qwen2.5-VL-3B with language-only LoRA keeps the base model below 4B parameters.
The saved runs demonstrate execution on a Tesla T4, including an uploaded PDF
with one repair generation. They do not establish that 3B is the smallest
capable model or guarantee memory headroom for every document.

| Measure | Current T4 artifact |
|---|---:|
| Test receipts | 32 |
| Receipt exact match, zero-shot / fine-tuned | 0.188 / 0.219 |
| Line-item name F1, zero-shot / fine-tuned | 0.766 / 0.766 |
| Line-item qty F1, zero-shot / fine-tuned | 0.859 / 0.859 |
| Total F1, zero-shot / fine-tuned | 0.621 / 0.655 |
| Hallucination before / after grounding | 0.144 / 0.000, both stages |
| JSON validity | 1.000, both stages |
| Latency p50 / p95, zero-shot | 23.3 / 47.0 s |
| Latency p50 / p95, fine-tuned | 39.7 / 81.2 s |
| Peak GPU memory | 1.24 GiB, both stages |

Using p50 as a throughput proxy gives roughly 155 pages/hour zero-shot and 91
pages/hour fine-tuned. At AWS `g4dn.xlarge` T4 on-demand **$0.526/hour**, that
is approximately **$3.40 and $5.81 per 1,000 pages**, respectively, before
storage, networking, retries, and idle time. This is illustrative only: the
sample is small and p95 is much higher than p50. Confirm the current cloud
quote before using the rate commercially.

The conclusion is: **T4 model execution, the same-test-set comparison, and a
real-model FastAPI smoke test inside Kaggle have saved evidence. The local UI
was tested in dummy mode. Follow-up concurrency timings support serialized
processing and timeout-based load shedding; production deployment and
production quality/latency remain unvalidated.** Fine-tuning
provided a modest gain on totals and receipt exact match, with no line-item
gain and a substantial latency cost.

## Uploaded PDF execution on Kaggle

The published adapter at revision
`e9d1eb7ff9a11273a87a5d27a9fec7ba35b4ee92` ran on a Tesla T4 against the
one-page `dmart.pdf`. It returned seven line items and a predicted total of
845.00 after one schema-repair retry, in **118.733 seconds** of elapsed page
processing. Rendering, preview display, generation, and repair are included;
this is not a model-only latency percentile. No new memory peak was printed.

This establishes an on-demand notebook execution path without a paid managed
endpoint, subject to Kaggle GPU availability and quota. It is not always-on
hosting for the local UI. The result is ungrounded, the date is non-ISO, and
tax 0.00 is unverified. Keep it separate from the 32-receipt CORD benchmark;
do not recompute throughput or per-1,000-page cost from this single sample.

The separate FastAPI run used a pinned adapter snapshot and the synthetic
`receipt_ok.pdf`. The current response capture returned HTTP 200 with a cache miss in **25.897 seconds**,
two line items, a total of 13.50, an ISO date, and 13 verified field statuses.
The saved values pass schema and arithmetic checks. This closes the basic
real-model API smoke-test gap, not production deployment: the API was a
loopback subprocess inside Kaggle. Its sample has a text layer, so OCR fallback
and accuracy on real scanned uploads still need validation.

Remaining gates include real-upload/date robustness, OCR fallback, multi-page
behavior, matching settings/version metadata for the five-completion run,
and end-to-end API/UI and Docker validation. Neither single-sample run establishes production
throughput, new latency percentiles, or peak GPU memory.

## Where it fails

- **Dense tables:** long-edge 1008 px / 1008 vision tokens drop small cells.
- **Poor scans and rotation:** CORD is relatively clean; we do not deskew.
- **Small fonts:** first thing to lose when we cap `max_pixels`.
- **Multi-page:** we infer once per page and merge. Totals on a later page win when they are better grounded; a hallucinated header on page 2 is dropped if page 1 has a verified store name.
- **CORD labels:** no store name or date. Those fields stay weak on the official test set; synthetic samples cover them for the desk.

## Queue and timeouts

The service is designed for one GPU job at a time; the Qwen executor also has
one worker. Default queue depth is 8 including the holder. Overflow maps to
429, a wait beyond `QUEUE_TIMEOUT_S` to 503, and an inference timeout to 504.
These controls avoid concurrent model execution; the saved runs do not prove
that every second concurrent forward pass would cause an OOM.

The [timeout capture](concurrency_output.txt), matching
[concurrency_console.txt](concurrency_console.txt), has three HTTP 200 cache
misses with waits of 0.00, 24.76, and 49.49 seconds, plus two 503 responses at
about 60.03 seconds. This is consistent with the saved 60-second queue setting.
The 503 bodies were not captured, so their exact error codes are not proven.
The last successful request finishes at 74.24 seconds because the timeout
limits queue waiting, not total duration.

The [separate completion capture](concurrency_completion.txt) has five HTTP
200 cache misses. Processing order is 2, 1, 5, 4, 3; waits are 0.00, 25.37,
50.41, 75.62, and 100.61 seconds. Each request processes for about 25 seconds,
and the last completes at 125.99 seconds. A 120-second queue setting was
suggested but is not verified: this user-pasted run has no matching saved
settings/version metadata. The 60-second settings belong to the earlier run.

The corrected pipeline runs in a thread while holding the async GPU slot.
Cancellation retains the slot until the worker exits. Local blocking-extractor
tests pass for responsive health checks, 429 overflow, 503 queue timeout,
five serialized callers with nonzero queue waits, and repeated cancellation.
The follow-up GPU timing observations now support the queue fix. They do not
directly instrument simultaneous GPU forwards, measure health responsiveness
under GPU load, or establish 429 overflow and production capacity. Client
overlap alone is not proof of GPU exclusivity. Both the three-success timeout
run and the five-success completion run are useful evidence; five 200s are not
a required outcome of correct load shedding. Keep these observations separate
from the CORD latency percentiles and illustrative cost calculation.

## What we would do next (not implemented)

Serve with a vision-capable engine and constrained JSON decoding, add a deskew/binarize step, mix SROIE for store/date, and train vision-layer LoRA on a 24 GB card. Prioritize these improvements within the existing architecture.
