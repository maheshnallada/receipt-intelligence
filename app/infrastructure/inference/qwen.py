"""Qwen2.5-VL-3B singleton. Loaded once at startup. Temperature 0."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from io import BytesIO
from typing import Any

from app.domain.document import Page
from app.domain.errors import InferenceTimeout

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You extract receipt fields from a page image. Copy only text that is "
    "visible on the page. If a field is absent, use null. Never guess or copy "
    "a product name into store_name. Only return store_name or date when the "
    "receipt clearly shows it; use null for unknown or placeholder text. "
    "Return a single JSON object with keys store_name, date (YYYY-MM-DD), "
    "line_items (name, qty, unit_price, amount), subtotal, tax, total. "
    "qty is the printed quantity; keep it null when it is not printed. "
    "unit_price is the price for one unit; amount is the total for the line. "
    "Do not swap them. Numbers must be JSON numbers, not strings. Do not guess."
)

REPAIR_PREFIX = (
    "Your previous output failed validation. Return only one valid JSON object, "
    "with no markdown, commentary, placeholders, or guessed values. Use null "
    "for unknown fields. Error: "
)


class QwenExtractor:
    def __init__(
        self,
        *,
        model_id: str,
        adapter_path: str = "",
        max_pixels: int,
        timeout_s: float,
        hf_home: str = "",
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_pixels = max_pixels
        self.timeout_s = timeout_s
        self.ready = False
        if hf_home:
            import os

            os.environ.setdefault("HF_HOME", hf_home)
        self._model, self._processor = self._load()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")
        self.ready = True

    def extract_page(self, page: Page, *, repair_hint: str | None = None) -> str:
        prompt = SYSTEM_PROMPT
        if repair_hint:
            prompt = f"{REPAIR_PREFIX}{repair_hint}\n{SYSTEM_PROMPT}"
        future = self._pool.submit(self._generate, page.image_png, prompt)
        try:
            return future.result(timeout=self.timeout_s)
        except FutureTimeout as exc:
            raise InferenceTimeout(f"inference exceeded {self.timeout_s}s") from exc

    def _generate(self, image_png: bytes, prompt: str) -> str:
        from PIL import Image

        image = Image.open(BytesIO(image_png)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        try:
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._empty_cache()
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=None,
                )
            else:
                raise
        trimmed = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(trimmed, skip_special_tokens=True)
        return decoded[0] if decoded else ""

    def _load(self) -> tuple[Any, Any]:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=self.max_pixels,
        )
        quant = None
        try:
            from transformers import BitsAndBytesConfig

            if torch.cuda.is_available():
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
        except Exception:  # noqa: BLE001
            quant = None
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quant,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        return model, processor

    def _empty_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            logger.warning("cuda empty_cache failed")
