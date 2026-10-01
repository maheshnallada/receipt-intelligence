# Evaluation

## T4 Notebook Run

The final standalone notebook evaluates the same frozen CORD test slice
(`n=32`) with both the base model and the LoRA adapter. Test ids were never
used for training. Both stages use the same prompt, grounding, sanity checks,
and receipt ids `0` through `31`.

| Field | Zero-shot F1 | Fine-tuned F1 | Delta F1 |
|---|---:|---:|---:|
| store_name | 0.000 | 0.000 | 0.000 |
| date | 0.000 | 0.000 | 0.000 |
| subtotal | 0.718 | 0.718 | 0.000 |
| tax | 0.560 | 0.560 | 0.000 |
| total | 0.621 | 0.655 | +0.034 |
| line_items.name | 0.766 | 0.766 | 0.000 |
| line_items.qty | 0.859 | 0.859 | 0.000 |
| line_items.unit_price | 0.400 | 0.400 | 0.000 |
| line_items.amount | 0.625 | 0.625 | 0.000 |

| Metric | Zero-shot | Fine-tuned |
|---|---:|---:|
| Receipt exact match | 0.188 | 0.219 |
| JSON validity | 1.000 | 1.000 |
| Hallucination before grounding | 0.144 | 0.144 |
| Hallucination after grounding | 0.000 | 0.000 |
| Latency p50 / p95 (ms/page) | 23,264 / 47,048 | 39,746 / 81,182 |
| Peak GPU memory | 1.24 GiB | 1.24 GiB |

CORD does not label `store_name` or `date`; their F1 remains 0.000 and the exact-match values mostly reflect whether the model returned `null` consistently. This is a small smoke run, not a production benchmark.

## Uploaded PDF inference smoke test (Kaggle)

Saved evidence: [Final_Notebook_VLM.ipynb](../notebooks/Final_Notebook_VLM.ipynb),
Cells 59-61. This is a real-model run, separate from both the frozen CORD
comparison above and the dummy API results below.

| Observation | Saved result |
|---|---|
| Input | `dmart.pdf`, one page rendered at 713 x 1008 pixels |
| Base | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Adapter | `Mahesh-Nallada/vlm-receipt-extraction-lora` |
| Adapter revision | `e9d1eb7ff9a11273a87a5d27a9fec7ba35b4ee92` |
| GPU | Tesla T4 |
| Predicted line items | 7 |
| Predicted subtotal / tax / total | 845.00 / 0.00 / 845.00 |
| Notebook schema validation | Passed after one regeneration |
| Elapsed time | 118.733 seconds for the page, including the repair attempt |

The initial response used strings for quantities. Regeneration returned numeric
quantities and passed the notebook's Pydantic schema. The seven predicted line
amounts sum to 845.00; that internal consistency is not proof of correspondence
with every printed field. The predicted store field includes an address, and
the returned date `04-12-2018` does not satisfy the requested `YYYY-MM-DD` format.
The notebook schema accepts any date string, so `schema_valid: true` does not
establish compliance with the production API's date contract.

The run did **not** apply OCR grounding, domain arithmetic checks, or page
merging. In particular, predicted tax 0.00 is not verified evidence of zero tax.
There is no labeled gold comparison for this PDF, so no accuracy, F1, or
hallucination rate is claimed. The single timing includes rendering, preview,
generation, and repair; it is not a p50/p95 benchmark or a replacement for the
CORD latency/cost figures. The saved output reports a Kaggle `predictions.json`
path; the notebook embeds the page result, not the complete saved file.

This demonstrates that the Hub adapter can execute on an uploaded PDF in
Kaggle. This standalone run does not itself exercise FastAPI; the separate API
evidence follows below. The recorded revision must not be assumed to be the
checkpoint used in the earlier CORD comparison.

## Real-model FastAPI smoke test (Kaggle)

Saved evidence: [final-notebook-vlm.ipynb](../notebooks/final-notebook-vlm.ipynb),
Cells 69-73, [version.json](version.json), [run_settings.json](run_settings.json),
[extraction_capture (1).json](extraction_capture%20%281%29.json), and [receipt.json](receipt.json).
The notebook retains an earlier 31.619-second response; the current standalone
response capture below records 25.897 seconds. These are distinct observations.
The service ran on loopback inside the GPU notebook, not through the local UI
or a public endpoint. The version and settings identify the Qwen base and the
same adapter revision above, downloaded to a revision-pinned local snapshot.

| Observation | Saved result |
|---|---|
| Input | Bundled synthetic `receipt_ok.pdf`, one page |
| Backend | `qwen`, not `dummy` |
| HTTP status / cache | 200 / MISS (`force_refresh=true`) |
| Client elapsed time | 25.897 seconds |
| Reported processing time / queue wait | 25,756.3 ms / 0.0 ms |
| Receipt | Oak & Ember Cafe, 2026-03-15, two line items |
| Subtotal / tax / total | 12.50 / 1.00 / 13.50 |
| Verification | All 13 returned field statuses are `verified` |

The saved response passes the API schema, matches the sample's checked text
fields, and satisfies line-item and total arithmetic. This exercises the real
API's grounding and sanity-check path, but the sample has a PDF text layer:
it does not establish OCR-fallback accuracy or multi-page behavior. The
processing header includes rendering and postprocessing, not just GPU time.
One synthetic receipt is not a new accuracy benchmark or latency percentile;
these results are not pooled into the CORD comparison.

### Concurrency capture and remaining validation

[concurrency_output.txt](concurrency_output.txt) and
[concurrency_console.txt](concurrency_console.txt) now contain the matching
follow-up timeout run, not the older all-zero-wait run embedded in notebook
Cell 75. With the accompanying settings recording a 60-second queue limit,
requests 2, 4, and 5 returned 200/MISS after queue waits of 0.00, 24.76, and
49.49 seconds. Requests 1 and 3 returned 503 at about 60.03 seconds, consistent
with queue-timeout load shedding; their response bodies were not captured.
Request 5 completed at 74.24 seconds: the limit applies to queue waiting, not
total request duration.

The separate user-pasted follow-up is preserved verbatim in
[concurrency_completion.txt](concurrency_completion.txt):

| Processing order | Request | HTTP / cache | Queue wait (s) | Processing (s) | Client elapsed (s) |
|---|---:|---|---:|---:|---:|
| 1 | 2 | 200 / MISS | 0.00 | 25.37 | 25.39 |
| 2 | 1 | 200 / MISS | 25.37 | 25.04 | 50.45 |
| 3 | 5 | 200 / MISS | 50.41 | 25.21 | 75.66 |
| 4 | 4 | 200 / MISS | 75.62 | 24.99 | 100.64 |
| 5 | 3 | 200 / MISS | 100.61 | 25.36 | 125.99 |

All five requests completed successfully, with increasing waits consistent
with serialized processing. A 120-second queue timeout was suggested for this
run, but no matching settings/version capture was supplied. Do not apply the
saved 60-second settings to this completion run or label 120 seconds verified.
The captures have no per-run code identifier or GPU-forward instrumentation;
`max_active_requests=5` counts overlapping client requests only.

Together, these follow-up observations support the event-loop fix and the
intended queue behavior, not a production capacity guarantee. The corrected
pipeline runs in a worker thread while retaining its GPU slot through
cancellation. Local blocking-extractor tests additionally cover health
responsiveness, 429 overflow, 503 queue timeout, and cancellation; those
properties were not all directly probed in the GPU captures. Docker, Redis
integration, scanned-document OCR fallback, and the local UI's real-model path
remain unvalidated by these runs. CORD metrics and cost estimates are unchanged.

## Dummy Backend

Dummy-backend numbers are measured on committed sample PDFs. Command: `python scripts/evaluate.py`.

## dummy-samples (n=2)

| Field | P | R | F1 | Exact |
|---|---:|---:|---:|---:|
| store_name | 1.000 | 1.000 | 1.000 | 1.000 |
| date | 1.000 | 1.000 | 1.000 | 1.000 |
| subtotal | 1.000 | 1.000 | 1.000 | 1.000 |
| tax | 1.000 | 1.000 | 1.000 | 1.000 |
| total | 1.000 | 1.000 | 1.000 | 1.000 |
| line_items.name | 1.000 | 1.000 | 1.000 | 1.000 |
| line_items.qty | 1.000 | 1.000 | 1.000 | 1.000 |
| line_items.unit_price | 1.000 | 1.000 | 1.000 | 1.000 |
| line_items.amount | 1.000 | 1.000 | 1.000 | 1.000 |

- Receipt exact match: 1.000
- JSON validity: 1.000
- Hallucination before grounding: 0.000
- Hallucination after grounding: 0.000
- Latency p50/p95 (ms/page): 12.2 / 24.8
- Peak GPU memory: n/a

## Summary

On the current T4 artifact, fine-tuning improved receipt exact match by 3.1
percentage points and total F1 by 0.034. It did not improve line-item metrics.
Grounding reduced the hallucination rate from 0.144 to 0.000 for both stages,
while the adapter increased p50 latency by about 71% and p95 latency by about
73% without increasing measured peak GPU memory.

## Separate SROIE header probe

The 361-receipt SROIE artifact evaluates only `store_name` and `date`. It is a
header probe, not the assignment schema evaluation, and its `header-fine-tuned`
values are identical to the zero-shot values. It should therefore be reported
as exploratory evidence rather than as proof that the adapter improved the
model.
