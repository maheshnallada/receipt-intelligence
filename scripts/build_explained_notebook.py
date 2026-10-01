"""Generate notebooks/04_pipeline_explained.ipynb from the Full_pipeline function list."""

from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(source),
    }


def _lines(source: str) -> list[str]:
    return (source.strip("\n") + "\n").splitlines(keepends=True)


CELLS = [
    md(
        """# 04 · `Full_pipeline.ipynb` explained

Companion to [`notebooks/Full_pipeline.ipynb`](Full_pipeline.ipynb). That notebook **runs** the pipeline. This one **explains every function**: what it does, the design decision, and the tradeoff.

It does not train a VLM and does not import `app`. A few CPU cells replay the interesting helpers so you can see the policy, not just read it.

**How to read.** Each function is `What` → `Decision` → `Tradeoff`. Constants are grouped at the top of a section because they *are* the design.

**Pipeline order** (same as `run_pipeline`):

```
image → infer_page → parse_model_output (+ 1 repair)
      → ground_receipt (per page, null-out)
      → merge_pages
      → ground_receipt (full text)
      → check_totals (keep number, mark unverified)
```"""
    ),
    md(
        """## 0. Setup constants and image math

These live in the first import cell of `Full_pipeline.ipynb`. They exist because a T4 is 16 GB and Qwen2.5-VL spends **one token per 28×28 pixels** before any JSON.

| Name | Default | Why |
|---|---|---|
| `MODEL_ID` | `unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit` | Unsloth-owned 4-bit weights. Passing a custom `BitsAndBytesConfig` into Unsloth caused NaN. |
| `HF_BASE_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | Eval/serve path uses official transformers + bnb, not Unsloth. |
| `LONG_EDGE` / `MAX_PIXELS` | 1008 / `1008×28×28` | Inference OCR quality. One 3B + one repair generate fits a T4. |
| `EVAL_MIN_PIXELS` | `256×28×28` | Stops the processor from blowing a tiny crop up to huge token counts. |
| `TRAIN_VISION_TOKENS` | 512 | Training images only. `512×28×28` px ≈ 512 vision tokens. |
| `TRAIN_MAX_SEQ` | 3072 | Room for 512 vis + prompt + compact CORD JSON. Same value on model, collator, and `SFTConfig`. |
| `LORA_R` / `α` / dropout | 16 / 32 / 0 | Language/attention/MLP only. Dropout 0 — Unsloth is more stable. |
| `EVAL_N` | 8 | T4 wall-clock. Same ids for zero-shot and fine-tuned. |
| `SANITY_TOLERANCE` | 0.05 | Rounding / Rp vs cents. Used for money equality and total checks. |
| `MAX_TRAIN` | 200 (`0` = full CORD train) | Smoke default so one Colab session finishes. |"""
    ),
    md(
        """### `smart_hw(width, height, min_pixels, max_pixels, factor=28)`

**What.** Qwen `smart_resize`: sides multiple of 28, area in `[min_pixels, max_pixels]`.

**Decision.** Match the processor, not a naive long-edge scale. Token count is then `(h//28)×(w//28)`.

**Tradeoff.** Slightly more code than `thumbnail()`. Wrong factor (14 vs 28) mis-estimates tokens and you truncate or OOM."""
    ),
    md(
        """### `resize_image(image, min_pixels, max_pixels)`

**What.** RGB convert + `smart_hw` + bicubic resize.

**Decision.** Resize **before** the processor so train and eval token budgets are deterministic.

**Tradeoff.** Bicubic is slower than nearest; receipts stay readable. Train uses a tighter pixel cap than eval — small print can vanish in training but stay at 1008 for scoring."""
    ),
    md(
        """### `vision_token_count(image)`

**What.** `(h//28)×(w//28)` after resize.

**Decision.** Cheap estimate used by `filter_to_budget`. Avoids running the full processor on 200 images just to drop 0–2 outliers.

**Tradeoff.** Off by a few tokens vs `image_grid_thw`. `CHAT_OVERHEAD=48` covers the slack."""
    ),
    code(
        r"""
import math

def smart_hw(width, height, min_pixels, max_pixels, factor=28):
    h = max(factor, int(round(height / factor) * factor))
    w = max(factor, int(round(width / factor) * factor))
    if h * w > max_pixels:
        scale = math.sqrt(max_pixels / max(1, height * width))
        h = max(factor, int(math.floor(height * scale / factor) * factor))
        w = max(factor, int(math.floor(width * scale / factor) * factor))
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / max(1, height * width))
        h = max(factor, int(math.ceil(height * scale / factor) * factor))
        w = max(factor, int(math.ceil(width * scale / factor) * factor))
    return int(w), int(h)

# A tall CORD-like page at train vs eval budgets
print("train 512 vis", smart_hw(864, 1296, 128 * 28 * 28, 512 * 28 * 28),
      "tokens", (smart_hw(864, 1296, 128 * 28 * 28, 512 * 28 * 28)[1] // 28)
      * (smart_hw(864, 1296, 128 * 28 * 28, 512 * 28 * 28)[0] // 28))
print("eval 1008 edge", smart_hw(864, 1296, 256 * 28 * 28, 1008 * 28 * 28))
"""
    ),
    md(
        """## 1. Schema, CORD mapping, JSON parse

`EXTRACT_PROMPT` is the only instruction the VLM sees. It asks for visible text, JSON numbers, and `null` when absent. Constrained decoding (Outlines) was **not** added: JSON validity is already 1.0 after parse+repair; it would not raise field F1.

`QTY_RE` / `FENCE` pull a number out of `"2x"` and a `{...}` out of a markdown fence."""
    ),
    md(
        """### `parse_qty(raw)` / `parse_money(raw)`

**What.** Coerce CORD/model strings to `float | None`. `parse_money` handles `Rp`, `1.234,56` vs `1,234.56`.

**Decision.** The receipt schema requires JSON numbers. CORD stores money as messy strings. Fail → `null`, never a guessed float.

**Tradeoff.** Locale heuristics can misread a rare format. Better than crashing the schema or keeping `"12.50"` as a string (which fails Pydantic in the API)."""
    ),
    md(
        """### `as_menu_list(menu)`

**What.** CORD `menu` is sometimes one dict, sometimes a list.

**Decision.** Normalize to `list[dict]` so `cord_to_schema` has one loop.

**Tradeoff.** Non-dict entries are dropped. Those are unlabeled junk in CORD, not line items."""
    ),
    md(
        """### `cord_to_schema(gt_parse)`

**What.** Map `gt_parse` → receipt schema. **`store_name` and `date` are always `null`.** Menu `nm/cnt/unitprice/price` → line items. Subtotal/tax/total from `sub_total` / `total`.

**Decision.** Do not invent store/date labels CORD does not have. Fine-tune cannot learn those fields from CORD; the API uses synthetic samples + grounding for them.

**Tradeoff.** Zero-shot/fine-tuned **F1 on store/date stays 0** even when Exact is 1.0 (both null). That is correct scoring, not a bug. Ignored: void lines, cash, change, service charge."""
    ),
    md(
        """### `ground_truth_of(row)`

**What.** `row["ground_truth"]` may be a JSON string; return `gt_parse`.

**Decision.** Hugging Face CORD v2 stores the blob as a string. One unwrap, used by train and eval.

**Tradeoff.** Assumes the official key `gt_parse`. A different receipt dataset would need its own mapper."""
    ),
    md(
        """### `split_hash(ids)`

**What.** First 16 hex chars of SHA-256 over comma-joined ids.

**Decision.** Freeze “which images were train vs test” in `splits.json` so prompt edits cannot silently use test.

**Tradeoff.** Order-sensitive. We hash official split order, not a sorted set."""
    ),
    md(
        """### `extract_json_object(text)`

**What.** Prefer a ` ```json ` fence, else first `{` through last `}`.

**Decision.** VLMs wrap JSON in prose. Taking the outer braces is enough; a full JSON parser on the whole string fails on the preface.

**Tradeoff.** Extra `{` in a store name inside a broken string can yield a bad slice. `json_repair` and the one VLM retry sit on top."""
    ),
    md(
        """### `parse_model_output(text)`

**What.** Slice → `json.loads` → optional `json_repair` → coerce line items / money / ISO date. Drop items with empty names.

**Decision.** Schema-valid dict or raise. The service never returns invalid JSON. Date must be `%Y-%m-%d` or the parse fails (then repair generate).

**Tradeoff.** `json_repair` can “fix” garbage into a plausible object. Grounding nulls anything not on the page. We did **not** use Outlines: validity is already 1.0; a tight schema without `null` would *force* hallucinations."""
    ),
    md(
        """## 2. Grounding, sanity, merge

Extraction policy, encoded here:

- **Grounding:** a non-null leaf that is not in the page text is **nulled** and marked `not_found`.
- **Sanity:** line-sum ≠ subtotal or subtotal+tax ≠ total → keep the number, mark **`unverified`**. Do not null a printed total just because arithmetic failed (discounts, rounding)."""
    ),
    md(
        """### `normalize_text` / `normalize_number_token` / `parse_number` / `numeric_variants`

**What.** Casefold, strip currency words, collapse spaces; then emit ways a float might be printed (`12`, `12.00`, `12,00`, `1,234.50`).

**Decision.** Grounding is substring match on page text (PDF layer or OCR). Receipts print the same amount many ways.

**Tradeoff.** `12` matches any token that parses to 12.00 within 0.05. False verify on a different `12` on the page is possible; better than nulling a real total because of `Rp` or grouping."""
    ),
    md(
        """### `numbers_match(predicted, document_text)` / `text_present` / `date_present`

**What.** Is this number/string/ISO date actually on the page?

**Decision.** Dates try ISO, `D/M/Y`, `M/D/Y`, and unpadded day/month. Text match allows space-stripped (`OAK & EMBER` vs `OAK&EMBER`).

**Tradeoff.** `M/D/Y` vs `D/M/Y` can both match `01/02/2024`. We accept that; the value still had to appear. We do not run a second VLM to “confirm” — that would be another GPU job."""
    ),
    md(
        """### `verification_paths(n)` / `_status(value, present, empty)`

**What.** Every leaf gets `verified | unverified | not_found`. `_status`: `None` → `not_found`; empty page text → keep nothing, `unverified`; present → keep + `verified`; else null + `not_found`.

**Decision.** Empty OCR/PDF layer is not “not in document” — we cannot prove presence, so unverified/null rather than fake verified.

**Tradeoff.** A blank text layer nulls everything even if the VLM read the image correctly. The API’s Tesseract fallback exists to reduce that. The notebook uses CORD `valid_line` words as page text."""
    ),
    md(
        """### `ground_receipt(receipt, document_text)`

**What.** Per-leaf grounding. A line item whose **name** is not on the page is dropped entirely (not kept with nulled amounts).

**Decision.** Null-out hallucinations: never return a value that is not in the document.

**Tradeoff.** A real item with an OCR miss on the name disappears. Amounts on that line never reach sanity. Alternative (keep + unverified) would leak invented names into the UI."""
    ),
    md(
        """### `check_totals(receipt, tolerance=0.05)`

**What.** If item amounts exist and disagree with subtotal, mark those amounts and subtotal `unverified`. If `subtotal + (tax or 0) ≠ total`, mark the money fields `unverified`. **Values stay.**

**Decision.** Printed totals can be right when the line list is incomplete (voids, discounts). Nulling them would hide the number a reviewer needs.

**Tradeoff.** An inconsistent receipt still shows numbers. The UI must surface `_verification`, not treat every float as approved."""
    ),
    md(
        """### `page_score(receipt, fields)` / `merge_pages(pages)`

**What.** Concatenate line items. Pick **header** (store, date) from the page with most verified header fields (tie → first page). Pick **money** from the page with most verified money fields (tie → last page — totals often sit on the last sheet).

**Decision.** One overall “best page” let a totals-heavy last page overwrite a verified store on page 1. Split scores fixed that.

**Tradeoff.** Two pages both claiming different totals: last verified wins. No LLM judge. Duplicate items across pages are not de-duplicated (receipts rarely reprint the same SKU block)."""
    ),
    md(
        """## 3. Extract pipeline"""
    ),
    md(
        """### `empty_receipt()`

**What.** All-null schema plus `not_found` verification.

**Decision.** Parse/repair both failed → still valid JSON. Never 500 with a broken body.

**Tradeoff.** Looks like “empty receipt” not “model failed”. Logs/spans record the failure; the HTTP contract stays schema-valid."""
    ),
    md(
        """### `extract_one_page(raw_text, repair_text=None)`

**What.** Parse raw; on failure parse the repair generate; on second failure return `empty_receipt()`.

**Decision.** Allow at most **one** extra GPU generate for JSON repair, same slot.

**Tradeoff.** Two broken generations → empty, not a third try. Latency p95 in eval is often that one retry."""
    ),
    md(
        """### `run_pipeline(page_outputs)`

**What.** For each page: parse → copy `before` → ground on that page’s text. Merge. Ground again on concatenated text. Sanity. Return `(final, before, combined_text)`.

**Decision.** Ground **before** merge so a hallucinated store on page 2 cannot win a merge score. Ground **after** merge so a header copied from page 1 is re-checked against the full document. `before` is for the CPU demo / debugging, not the API response.

**Tradeoff.** Two grounding passes. Cheap compared to the VLM. Multi-page “see page 1 for store” works only if page-1 text is in `combined`."""
    ),
    md(
        """## 4. CPU demo

The demo cell is not a function. It builds a gold-like JSON, a hallucination (`store_name` + `total: 99` not on the page), and a totals-mismatch receipt, then prints `run_pipeline` results.

**Decision.** Reviewers without a GPU can still see null-out vs keep-and-unverify.

**Tradeoff.** Synthetic text, not CORD. It proves policy, not VLM accuracy."""
    ),
    md(
        """## 5. Metrics"""
    ),
    md(
        """### `values_equal(left, right)`

**What.** Both null → equal. Strings → `normalize_text`. Numbers → within `SANITY_TOLERANCE`.

**Decision.** Same tolerance as sanity, so eval and product agree on “12.00 vs 12”.

**Tradeoff.** 0.05 is loose for unit prices in some currencies. Tight enough to reject 12 vs 21."""
    ),
    md(
        """### `prf(tp, fp, fn, exact, n)`

**What.** Precision / recall / F1; Exact = `exact / n`.

**Decision.** If there are no positives, P=R=F1 = **0** (not 1). That is why CORD `date` can be Exact 1.0 and F1 0.0.

**Tradeoff.** Easy to misread the table. The comparison cell prints this caveat."""
    ),
    md(
        """### `field_scores(golds, preds, field)`

**What.** Per receipt, one header/money field. null/null → Exact only. Gold null / pred value → FP. Gold value / pred null → FN. Both set and equal → TP+Exact. Both set and different → FP+FN.

**Decision.** Standard information-extraction scoring with an extra Exact that credits honest nulls.

**Tradeoff.** A model that always emits null scores Exact high and F1 0 on labeled fields — visible on store/date."""
    ),
    md(
        """### `line_item_scores(golds, preds)`

**What.** Pair gold items to pred items by **normalized name**, then score each leaf. Extra pred items are unmatched (FP on every leaf).

**Decision.** Names are the only stable key. Order on the page is not reliable after merge.

**Tradeoff.** Two lines named “Tea” collapse. A typo in the name counts as a miss + an extra, which tanks name F1 and every leaf on both rows."""
    ),
    md(
        """### `exact_receipt(gold, pred)`

**What.** Every header field and every line-item leaf must match (items sorted by name).

**Decision.** Report strict exact match: 0/8 on the T4 slice.

**Tradeoff.** One wrong qty makes the whole receipt fail even if totals are perfect."""
    ),
    md(
        """### `hallucination_rate(preds, texts)`

**What.** Among non-null emitted leaves, fraction not found in page text (same rules as grounding).

**Decision.** Measured **after** grounding in `summarize`, so 0.0 means the pipeline stripped inventions, not that the VLM never guessed.

**Tradeoff.** Logging this before grounding measures raw model output; this metric measures output after grounding."""
    ),
    md(
        """### `latency_percentiles` / `summarize` / `show_table`

**What.** p50/p95 of `extract_image` wall time (includes optional repair). Peak CUDA bytes if a GPU is present. Markdown table.

**Decision.** Latency is per page, not per PDF. GPU peak is for the eval load only (one 4-bit 3B).

**Tradeoff.** p95 on n=8 is dominated by one retry. Do not quote it as “typical page time”."""
    ),
    md(
        """## 6. CORD load and wandb

Not much function surface. The load cell writes `splits.json` with official ids. **`wandb_init`** no-ops without `WANDB_API_KEY`. Config records `TRAIN_LR` (5e-5), not the old 2e-4 that overflowed fp16 on T4.

**Tradeoff.** Offline wandb still works if you set `WANDB_MODE=offline`. Default is disabled so CI stays quiet."""
    ),
    md(
        """## 7. VLM load and one-page infer"""
    ),
    md(
        """### `load_vlm(adapter_path=None)`

**What.** `AutoProcessor` + 4-bit `Qwen2_5_VLForConditionalGeneration` on `HF_BASE_ID`. Optional PEFT adapter. Clears `generation_config.temperature` so greedy decode does not warn.

**Decision.** Eval/serve is plain transformers. Unsloth is train-only. Processor `min_pixels` / `max_pixels` set at load (writable). Do not `setattr` `min_pixels` later — transformers 5.x raises `_min_pixels has no setter`.

**Tradeoff.** Two code paths (Unsloth train vs HF infer). One 3B on GPU. A second copy OOMs the T4."""
    ),
    md(
        """### `free_vlm()`

**What.** Drop model/processor refs, `gc.collect()`, `empty_cache()`, reset peak stats.

**Decision.** Zero-shot must not still sit on the GPU when QLoRA starts.

**Tradeoff.** You reload weights for fine-tuned eval (~30–60s). Cheaper than swapping or CPU offload on Colab."""
    ),
    md(
        """### `infer_page(image, repair_hint=None)`

**What.** Resize to eval pixels, chat template (image + prompt or “previous failed: …”), greedy `generate`, decode new tokens only.

**Decision.** Temperature 0. `max_new_tokens=1024` covers a dense line list. Repair is a **new** user message, not a conversation history, so the image is sent again.

**Tradeoff.** Sending the image twice on repair costs vision tokens. History-only repair would be shorter but Unsloth/Qwen chat + 4-bit was flakier."""
    ),
    md(
        """### `extract_image(image, page_text="")`

**What.** `infer_page` → if parse fails, `infer_page(..., repair_hint)` → `run_pipeline` on that one page.

**Decision.** Notebook analogue of `ExtractReceipt` without PDF/cache/queue.

**Tradeoff.** Page text for CORD comes from `cord_page_text` (word boxes), not Tesseract. Grounding is optimistic vs a scanned PDF with an empty text layer."""
    ),
    md(
        """## 8. Zero-shot eval"""
    ),
    md(
        """### `cord_page_text(row)`

**What.** Join CORD `valid_line` word strings.

**Decision.** Official words are the grounding document. No extra OCR in the notebook.

**Tradeoff.** Words can miss printed totals that the image shows; then a correct VLM total gets nulled. That is grounding working as specified."""
    ),
    md(
        """### `cord_eval_rows(n=EVAL_N)`

**What.** First `n` official **test** rows: image, `cord_to_schema` gold, page text.

**Decision.** Frozen prefix of the official test split. Same list after training.

**Tradeoff.** n=8 is directional, not a leaderboard. One receipt moves F1 a lot."""
    ),
    md(
        """### `eval_vlm(rows, stage)`

**What.** Loop `extract_image`, take schema keys only (no `_verification` in the scored pred), `summarize` + `show_table`.

**Decision.** Score the post-pipeline receipt, not raw model text. That matches the API.

**Tradeoff.** You cannot see “VLM-only F1” without a second table. Grounding can hide a good total or a bad store."""
    ),
    md(
        """## 9. QLoRA

Train-cell helpers exist because **truncation, OOM, and NaN** showed up on the T4 in that order."""
    ),
    md(
        """### `compact_target(target)` / `to_messages(image, target)`

**What.** Compact JSON (`separators=(",", ":")`). Chat: user = image **object** + prompt; assistant = JSON.

**Decision.** Image inside the message (Unsloth collator). Compact JSON saves tokens so `TRAIN_MAX_SEQ=3072` holds long receipts.

**Tradeoff.** Serve-time VLM may emit pretty JSON. That is fine — we train content, not whitespace."""
    ),
    md(
        """### `convert_split(split, limit=0)`

**What.** Resize each train image to `TRAIN_*_PIXELS`, map gold, wrap messages. Prints vision-token and JSON-char p50/max.

**Decision.** Prove the budget before Unsloth loads.

**Tradeoff.** `limit=200` is not the full 800. Faster, more forgetting on tax/total (see metrics)."""
    ),
    md(
        """### `estimate_seq_len` / `filter_to_budget`

**What.** `vision + tok(prompt+json) + 48`. Drop rows over `TRAIN_MAX_SEQ`.

**Decision.** Do not truncate. Truncation cuts the assistant JSON or image placeholders → empty labels / Unsloth image-token mismatch / NaN.

**Tradeoff.** A few long receipts never train. Raising seq to 8192 OOMs. Shrinking pixels further loses OCR."""
    ),
    md(
        """### `_sft_length_kwargs` / `_warmup_steps` / `_bind_processor_pixels`

**What.** Pass `max_length` and/or `max_seq_length` depending on TRL. Warmup ≥5 steps or 20% of optimizer steps. Bind pixels via `size["shortest_edge"|"longest_edge"]` only.

**Decision.** TRL renamed the arg. `warmup_ratio` is deprecated. **Never setattr `min_pixels`** on `Qwen2VLImageProcessor` (no setter).

**Tradeoff.** Size-dict mutate is enough because images are pre-resized. If a large image slipped through, the processor would still downscale via `longest_edge`."""
    ),
    md(
        """### `_nan_callback()` / `_save_finite_adapter(...)`

**What.** `GuardNaNCallback` zeros grads if any are non-finite before Adam. After train, if the last logged loss is NaN, reload the latest `checkpoint-*` at or before the last finite step, then `save_pretrained`.

**Decision.** fp16 on T4 overflowed at step 19 after a healthy 3.66→0.20 drop. Saving the NaN tail made fine-tuned money worse. Checkpoints every 5 steps.

**Tradeoff.** Skipping a step wastes that batch. Reloading step 15 misses steps 16–18. Still better than publishing NaN LoRA weights. `adamw_torch` + `5e-5` is slower/stabler than `adamw_8bit` + `1e-4`."""
    ),
    md(
        """### `train_unsloth(train_rows)` / `train_hf(train_rows)`

**What.** Unsloth first: `FastVisionModel`, language LoRA only, `UnslothVisionDataCollator(resize="max", completion_only_loss=True)`, `fp16`, `save_steps=5`. On failure: free GPU, HF 4-bit + same collator if Unsloth already patched `SFTTrainer` (otherwise it demands `formatting_func`).

**Decision.** `import unsloth` at the top of the setup cell, before transformers. `completion_only_loss` so we do not train on image tokens. Collator `resize="max"` so it does not undo our resize.

**Tradeoff.** Unsloth is faster but version-fragile. HF fallback after a failed Unsloth load used to OOM (two 3Bs) and then crash on `formatting_func`. The fallback now frees memory and uses the vision collator."""
    ),
    md(
        """## 10. Fine-tuned eval and Hub"""
    ),
    md(
        """### `_field_score` / `compare_stages`

**What.** Print Δ F1 / Exact for store, date, money, and each line-item leaf.

**Decision.** Report the comparison table, not a single accuracy. CORD store/date F1 is expected 0.

**Tradeoff.** n=8. Use it to see “items up, totals down”, not to claim a SOTA delta."""
    ),
    md(
        """The rest of the last cell is orchestration: `free_vlm` → `load_vlm(ADAPTER_DIR)` → `eval_vlm` → write `metrics.json` → optional wandb table → optional `HF_PUSH=1` upload.

**Decision.** Push is opt-in. Auto-push during experiments uploaded a NaN adapter once.

**Tradeoff.** Fine-tuned eval reloads the 3B. Peak GPU in that stage looks lower (~3.4 GiB) because peak stats were reset; do not compare it to zero-shot 6.6 GiB as “the adapter is smaller”."""
    ),
    md(
        """## Design map (whole notebook)

| Choice | Picked | Rejected | Why |
|---|---|---|---|
| Model | Qwen2.5-VL-3B QLoRA | SmolVLM / PaliGemma / 7B | ≤4B, JSON + OCR, T4 4-bit |
| Train pixels | 512 vision tokens | 1008 long-edge | Fits 3072 seq; 8192 OOMs |
| Truncation | Drop overflow rows | Lower `max_length` only | Lowering cuts JSON / image tokens |
| Grounding | Null if not on page | Keep + unverified | Never return absent text |
| Sanity | Keep + unverified | Null broken totals | Printed total can be right |
| Merge | Header score ≠ money score | One best page | Totals page stole verified store |
| Structured decode | No (parse + 1 retry) | Outlines | Validity already 1.0; risk of forced guesses |
| LoRA target | Language only | + vision encoder | T4 headroom |
| LR | 5e-5 + skip NaN grads | 1e-4 / 2e-4 | fp16 overflow after step 18 |
| Eval n | 8 same test ids | Full test | Colab time; honest small-n |

Open [`Full_pipeline.ipynb`](Full_pipeline.ipynb) to run this. Use [`03_full_pipeline.ipynb`](03_full_pipeline.ipynb) if you want the same pipeline without Colab experiment residue."""
    ),
]


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks" / "04_pipeline_explained.ipynb"
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": CELLS,
    }
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("wrote", out, "cells", len(CELLS))


if __name__ == "__main__":
    main()
