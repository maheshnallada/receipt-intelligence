# Submission Email

## What to provide

Send the following in the email or as links in the email:

1. GitHub repository link.
2. Hugging Face adapter link:
   [Mahesh-Nallada/vlm-receipt-extraction-lora](https://huggingface.co/Mahesh-Nallada/vlm-receipt-extraction-lora)
3. Final notebook with saved PDF inference and API output:
   [final-notebook-vlm.ipynb](../notebooks/final-notebook-vlm.ipynb).
   Cells 59-61 record the standalone uploaded-PDF run; Cells 69-75 record
   real-model FastAPI execution. Remove embedded credentials before publication
   and rotate any exposed tokens.
4. Evaluation report: `reports/evaluation.md`.
5. Feasibility report: `reports/feasibility.md`.
6. Concurrency script, [timeout capture](concurrency_output.txt), and separate
   [completion capture](concurrency_completion.txt). The first records three
   200s and two 503s near the saved 60-second queue limit; the second records
   five 200 cache misses with increasing waits. The completion run was supplied
   as pasted output, without matching settings/version metadata; do not claim
   its suggested 120-second setting is verified. Also include
   [extraction_capture (1).json](extraction_capture%20%281%29.json), [receipt.json](receipt.json),
   [version.json](version.json), and [run_settings.json](run_settings.json).
7. The README contains Docker configuration and setup, architecture, API
   behavior, limitations, and model-loading instructions. Do not describe a
   Docker/GPU deployment as verified without a successful run.

## Email template

Subject: AI Engineer Round 2 Take-Home Assignment - Document Extraction Service

Hello,

Please find my submission for the AI Engineer Round 2 take-home assignment:

- GitHub repository: `<repository URL>`
- Hugging Face LoRA adapter: `https://huggingface.co/Mahesh-Nallada/vlm-receipt-extraction-lora`
- Evaluation and feasibility reports: included in the repository under `reports/`
- Training/evaluation and API-inference notebook: `notebooks/final-notebook-vlm.ipynb`
- Concurrency script and both captures: `scripts/concurrency_test.py`, `reports/concurrency_output.txt`, `reports/concurrency_completion.txt`

The implemented service accepts a PDF at `POST /extract`, performs document
grounding and sanity checks, and returns structured JSON in the same response.
It uses an in-process single-flight inference slot. Local API/UI validation
used the dummy backend. Separately, the actual FastAPI service ran with the
Qwen base and pinned published LoRA adapter inside a Kaggle GPU notebook.
For the bundled synthetic receipt, it returned HTTP 200 with a cache miss in
25.897 seconds, a total of 13.50, and 13 verified field statuses. The saved
response passes schema and arithmetic checks. This validates a basic real-model
API path, not general extraction accuracy, OCR fallback, or public hosting.

After correcting event-loop blocking, a follow-up capture recorded three
successful requests and two 503 responses near the 60-second queue limit,
consistent with timeout-based load shedding. A separate completion run
recorded five HTTP 200 cache misses, approximately 25 seconds of processing
each, increasing queue waits up to 100.61 seconds, and a final completion at
125.99 seconds. These timings support serialized processing. The completion
run lacks matching settings/version metadata; the suggested 120-second queue
setting is not independently verified. Neither capture directly instruments
GPU-forward exclusivity, and the 503 bodies were not saved. Local regression
tests cover health responsiveness, queue waits, overflow, timeout, and cancellation.

Separately, I ran the published LoRA adapter on an uploaded one-page DMart PDF
in Kaggle on a Tesla T4. The saved output contains seven predicted line items
and a total of 845.00, passing the notebook schema after one repair generation
in 118.733 seconds. This is real-model execution, but not a new accuracy
benchmark or a deployed endpoint: the date remains non-ISO and OCR grounding
and domain sanity checks were not applied to that notebook result.

The README documents setup, Docker configuration, design decisions, trade-offs,
and limitations. Docker deployment, the local UI's real-model path, and
production-scale quality and latency remain unvalidated.

Regards,
Mahesh Nallada