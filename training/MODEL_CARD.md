---
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-3B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - qwen2.5-vl
  - qwen
  - lora
  - qlora
  - receipt-extraction
  - document-ai
  - vision-language
datasets:
  - naver-clova-ix/cord-v2
language:
  - en
  - id
---

# Receipt extraction LoRA (Qwen2.5-VL-3B)

QLoRA adapter on [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) for turning a **receipt image** into structured receipt JSON:

`store_name`, `date` (`YYYY-MM-DD`), `line_items[]` (`name`, `qty`, `unit_price`, `amount`), `subtotal`, `tax`, `total`.

This repo is **the adapter only**. Load it on the base 3B instruct checkpoint. It is not a merged full model.

## Project links

- [Weights & Biases experiment report](https://forge.coreweave.com/wandb/maheshnallada/vlm-receipt-extraction/reports/Receipt-Intelligence--VmlldzoxODAzNzk0OA): may require sign-in or permission from the report owner.
- [Deployed frontend](https://receipt-intelligence-chi.vercel.app): frontend only, not a hosted model inference endpoint.

## Intended use

- Research baseline for receipt field extraction.
- Pair with a **post-processor**: parse/repair JSON, ground every non-null leaf against page text (null out hallucinations), then sanity-check totals. Scores below are **after** that pipeline.

Not intended as a standalone production OCR product. Do not use for KYC, tax filing, or any decision that needs audited store/date without a second source.

## Training target

The training pipeline maps official [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) **train** images to the schema above. The surviving training capture does not establish the provenance of the published adapter; see the training notes below.

CORD **does not label store name or date**. Those training targets are `null`, so this dataset does not supervise extraction of those fields. For live store/date you need other data (or keep the zero-shot base and only trust grounded text).

Ignored CORD fields: voided lines, cash/change, service charge, discounts.

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER = "Mahesh-Nallada/vlm-receipt-extraction-lora"

processor = AutoProcessor.from_pretrained(
    BASE, min_pixels=256 * 28 * 28, max_pixels=1008 * 28 * 28
)
quant = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE, quantization_config=quant, device_map="auto", torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

PROMPT = (
    "Extract the receipt as JSON with keys store_name, date (YYYY-MM-DD), "
    "line_items (name, qty, unit_price, amount), subtotal, tax, total. "
    "Copy only text visible on the page. Use null when a field is absent. "
    "Numbers must be JSON numbers, not strings. Do not guess."
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},  # PIL RGB
        {"type": "text", "text": PROMPT},
    ],
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

Greedy decode. Ground the JSON against OCR/page text before you trust a field.

## Training

The table describes the configuration in the surviving notebook, not a verified
record of the run that produced the published adapter. Defaults may differ from
the settings used for saved historical outputs.

| | |
|---|---|
| Base | `Qwen/Qwen2.5-VL-3B-Instruct` (Unsloth 4-bit load: `unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit`) |
| Method | QLoRA 4-bit NF4, language / attention / MLP only (`finetune_vision_layers=False`) |
| LoRA | r=16, α=32, dropout=0 |
| Data | CORD v2 official train, 200-image T4 smoke run (`MAX_TRAIN=200`). Test ids never used. |
| Image budget | Training default: up to 768 vision tokens (`768×28×28` px). Evaluation budget: up to 1008 vision tokens; standalone PDF rendering uses a 1008-pixel long edge. |
| Sequence | `max_seq_length=3072` on model + collator + `SFTConfig` (same value). Completions that still overflow are dropped, not truncated. |
| Optim | AdamW (torch), lr=`1e-5` default, cosine, warmup ~20% of steps, grad clip 1.0, accum 8, batch 1, 1 epoch |
| Hardware | NVIDIA Tesla T4 16 GB (saved Kaggle run). The CORD comparison reports 1.24 GiB peak GPU memory. Free the eval model before training. |
| Framework | Unsloth `FastVisionModel` + `UnslothVisionDataCollator` (`completion_only_loss=True`), TRL `SFTTrainer`, PEFT |

Do not “fix truncation” by setting `max_length=8192` on a T4. Qwen2.5-VL spends `pixels/(28×28)` tokens on the image first; shrink training pixels, then keep one shared sequence budget.

The surviving CORD training capture records non-finite losses, a last finite
logged step of 3, and no finite checkpoint available for recovery. It does not
verify that the published adapter came from a successful or recovered training
run. A clean training log and matching adapter revision are required to resolve
this provenance gap. Treat the saved comparison as historical, exploratory
evidence rather than proof of successful training in this capture.

## Evaluation

The final attached metrics artifact evaluates **32** frozen CORD v2 **test**
receipts with both the zero-shot base model and this adapter. Test ids were
never used in training. Scores are field P/R/F1 and Exact after parse → ground
→ sanity. Null=null counts as Exact and is skipped for F1. Line items are
paired by normalized name.

Hardware: Tesla T4. Both stages have JSON validity 1.0 and hallucination falls
from 0.144 before grounding to 0.000 after grounding.

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

Receipt-level exact match improves from **0.188 to 0.219**. Latency increases
from **23.3 / 47.0 seconds** p50/p95 to **39.7 / 81.2 seconds**. Peak GPU
memory is **1.24 GiB** for both stages.

### How to read this

- **Total F1 improves modestly** after fine-tuning, but subtotal and tax remain unchanged.
- **Line-item metrics do not improve** on this slice, so the adapter should not be described as a general line-item quality improvement.
- **store_name / date F1 stay 0** because CORD gold is always null in the official dataset. Exact-match values mostly reflect whether the model returned `null` consistently rather than actual store/date extraction.
- **unit_price F1 0.400**: CORD often omits `unitprice`, so exact-match can rise even when the field itself still cannot be recovered reliably.
- **n=32** is still too small for a production or leaderboard claim. One receipt can swing the field-level metric substantially.
- **Latency increases materially** after loading the adapter; this quality/latency trade-off should be measured on a larger test set before deployment.

The separate 361-receipt SROIE artifact is header-only and exploratory. Its
reported header-fine-tuned values are identical to zero-shot and must not be
presented as a demonstrated adapter improvement.

## Uploaded PDF inference evidence

The saved Kaggle run in
[final-notebook-vlm.ipynb](../notebooks/final-notebook-vlm.ipynb), in "Predict your own PDF on Kaggle",
loads the official Qwen base and this Hub adapter on a Tesla T4. Its recorded
adapter revision is `e9d1eb7ff9a11273a87a5d27a9fec7ba35b4ee92`.

For one page of `dmart.pdf` (713 x 1008 rendered pixels), it predicted seven
line items, subtotal 845.00, tax 0.00, and total 845.00. The first output used
string quantities; one repair generation produced numeric quantities and
passed the notebook schema. Recorded elapsed time was 118.733 seconds,
including the repair and page-processing overhead.

This is a real-model inference smoke test, not an accuracy benchmark. The
predicted date `04-12-2018` is not ISO-formatted, the store name includes an
address, and zero tax is not verified. The notebook accepts any date string
and does not run OCR grounding, arithmetic checks, or page merging. The seven
predicted amounts sum to 845.00, which shows internal consistency only.
There are no gold labels or zero-shot comparison for this PDF. Do not pool
this observation with CORD metrics or assume the recorded Hub revision is
the historical benchmark/training checkpoint described above.

The Hub repository distributes weights; this notebook run does not establish
an online inference endpoint or a validated real-model API deployment.

## Real-model API evidence

Separately, the FastAPI service loaded the pinned adapter inside Kaggle.
The current [response capture](../reports/extraction_capture%20%281%29.json)
records HTTP 200 with a cache miss in 25.897 seconds on a synthetic receipt,
total 13.50, and 13 verified field statuses. Schema and arithmetic checks pass;
the sample has a text layer and does not establish OCR-fallback accuracy.

The [timeout capture](../reports/concurrency_output.txt) has three successful
cache misses and two 503 responses near the saved 60-second queue limit. The
separate [completion capture](../reports/concurrency_completion.txt) has five
successful cache misses with sequential timing and a last completion at
125.99 seconds. Its suggested 120-second timeout has no matching saved
settings/version evidence. These observations support the queue fix, but do
not directly instrument GPU-forward exclusivity or establish production
capacity. They are not part of the CORD accuracy or latency benchmark.
Public hosting, Docker, and the local UI's real-model path remain unvalidated.

## Limitations

- Indonesian-heavy CORD training receipts; one additional uploaded DMart PDF has a qualitative inference result, not a systematic cross-layout, currency, or language evaluation.
- 200-sample / 1-epoch smoke train, not the full 800-image official train split.
- The comparison uses only 32 CORD test receipts and is not a production-scale benchmark.
- Vision encoder is frozen. The default 768-token training image budget can lose small-print detail relative to the 1008-token evaluation budget.
- Adapter can overwrite useful base-model money extraction. Prefer the base for tax/total if you only need those fields.
- Numbers must still pass grounding (`±0.05`) and arithmetic sanity in the service layer.

## License and data

- Base model: Qwen2.5-VL-3B-Instruct, Apache-2.0.
- Adapter: Apache-2.0.
- Training images/labels: CORD v2 (`naver-clova-ix/cord-v2`). Follow that dataset’s terms; this adapter does not redistribute CORD images.

## Citation

```bibtex
@misc{vlm-receipt-extraction-lora,
  title  = {Receipt extraction LoRA for Qwen2.5-VL-3B},
  author = {Mahesh Nallada},
  year   = {2026},
  url    = {https://huggingface.co/Mahesh-Nallada/vlm-receipt-extraction-lora}
}
```

CORD: Park et al., *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing*, 2019.  
Qwen2.5-VL: Qwen Team, 2025.
