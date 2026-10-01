"""QLoRA fine-tune for Qwen2.5-VL-3B. Unsloth first, peft+trl+bnb fallback.

Do not "fix truncation" by raising max_length to 8192. Qwen2.5-VL spends
pixels/(28×28) tokens on the image first; a full-res CORD page plus JSON
will not fit in 2048, and 8192 OOMs a T4. Cap training vision tokens, set
the same max_seq_length on the model, collator, and SFTConfig, then drop
the few leftovers that still overflow.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE") or ("disabled" if not os.environ.get("WANDB_API_KEY") else "online"))

from training.prepare_cord import cord_to_schema, ground_truth_of, to_messages

TRAIN_VISION_TOKENS = int(os.environ.get("TRAIN_VISION_TOKENS", "512"))
TRAIN_MIN_PIXELS = 128 * 28 * 28
TRAIN_MAX_PIXELS = TRAIN_VISION_TOKENS * 28 * 28
TRAIN_MAX_SEQ = int(os.environ.get("TRAIN_MAX_SEQ", "3072"))
CHAT_OVERHEAD = 48
EXTRACT_PROMPT = (
    "Extract the receipt as JSON with keys store_name, date (YYYY-MM-DD), "
    "line_items (name, qty, unit_price, amount), subtotal, tax, total. "
    "Copy only visible text. Use null when absent. Numbers must be JSON numbers."
)


def maybe_wandb(config: dict) -> None:
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
        os.environ["WANDB_MODE"] = "disabled"
    try:
        import wandb

        if os.environ.get("WANDB_MODE") == "disabled":
            return
        wandb.init(project=os.environ.get("WANDB_PROJECT", "vlm-receipt-extraction"), config=config)
    except Exception as exc:  # noqa: BLE001
        print(f"wandb skipped: {exc}")


def load_cord():
    from datasets import load_dataset

    return load_dataset("naver-clova-ix/cord-v2")


def smart_hw(width: int, height: int, min_pixels: int, max_pixels: int, factor: int = 28) -> tuple[int, int]:
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


def resize_for_train(image: Any) -> Any:
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        image = image.convert("RGB") if hasattr(image, "convert") else PILImage.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    w, h = image.size
    nw, nh = smart_hw(w, h, TRAIN_MIN_PIXELS, TRAIN_MAX_PIXELS)
    if (nw, nh) != (w, h):
        image = image.resize((nw, nh), PILImage.Resampling.BICUBIC)
    return image


def vision_token_count(image: Any) -> int:
    w, h = image.size
    return max(1, (h // 28) * (w // 28))


def compact_target(target: dict[str, Any]) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def messages_with_image(image: Any, target: dict[str, Any]) -> list[dict[str, Any]]:
    base = to_messages(target)
    base[0]["content"] = [
        {"type": "image", "image": image},
        {"type": "text", "text": EXTRACT_PROMPT},
    ]
    base[1]["content"] = [{"type": "text", "text": compact_target(target)}]
    return base


def convert_split(split) -> list[dict[str, Any]]:
    rows = []
    for row in split:
        image = resize_for_train(row["image"])
        target = cord_to_schema(ground_truth_of(row))
        rows.append({"messages": messages_with_image(image, target)})
    return rows


def estimate_seq_len(tokenizer, row: dict[str, Any]) -> int:
    image = row["messages"][0]["content"][0]["image"]
    completion = row["messages"][1]["content"][0]["text"]
    text_ids = tokenizer(EXTRACT_PROMPT + completion, add_special_tokens=False)["input_ids"]
    return vision_token_count(image) + len(text_ids) + CHAT_OVERHEAD


def filter_to_budget(tokenizer, rows: list[dict[str, Any]], max_seq: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped = 0
    lengths: list[int] = []
    for row in rows:
        n = estimate_seq_len(tokenizer, row)
        lengths.append(n)
        if n <= max_seq:
            kept.append(row)
        else:
            dropped += 1
    if lengths:
        ordered = sorted(lengths)
        print(
            f"token budget {max_seq} p50/p95/max "
            f"{ordered[len(ordered) // 2]}/{ordered[int(len(ordered) * 0.95)]}/{ordered[-1]} "
            f"kept {len(kept)} dropped {dropped}"
        )
    if not kept:
        raise RuntimeError("every training row exceeds TRAIN_MAX_SEQ; lower TRAIN_VISION_TOKENS or raise TRAIN_MAX_SEQ")
    return kept


def _sft_length_kwargs(max_seq: int) -> dict[str, int]:
    from trl import SFTConfig

    params = inspect.signature(SFTConfig.__init__).parameters
    kwargs: dict[str, int] = {}
    if "max_length" in params:
        kwargs["max_length"] = max_seq
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq
    return kwargs


def _warmup_steps(n_rows: int, accum: int = 8) -> int:
    return max(5, int(0.2 * max(1, math.ceil(n_rows / accum))))


def _nan_callback():
    from transformers import TrainerCallback

    class GuardNaNCallback(TrainerCallback):
        """Zero non-finite grads before Adam so one overflow cannot poison later steps."""

        def __init__(self) -> None:
            self.skipped = 0
            self.last_finite_step = 0

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
            del args, control, kwargs
            if model is None:
                return
            import torch

            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    self.skipped += 1
                    print(f"skip step {state.global_step}: non-finite grad (#{self.skipped})")
                    model.zero_grad(set_to_none=True)
                    return

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            del args, control, kwargs
            loss = (logs or {}).get("loss")
            if loss is not None and math.isfinite(float(loss)):
                self.last_finite_step = state.global_step

    return GuardNaNCallback()


def _save_finite_adapter(model, processor, trainer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = [row for row in trainer.state.log_history if "loss" in row]
    last = logs[-1]["loss"] if logs else None
    last_good = 0
    for row in logs:
        if math.isfinite(float(row["loss"])):
            last_good = int(row.get("step", last_good))
    print("last loss", last, "last finite step", last_good)
    if last is not None and not math.isfinite(float(last)):
        ckpts = []
        for path in out_dir.glob("checkpoint-*"):
            try:
                step = int(path.name.split("-")[-1])
            except ValueError:
                continue
            if step <= last_good:
                ckpts.append((step, path))
        if ckpts:
            step, ckpt = max(ckpts)
            print("restoring finite adapter from", ckpt)
            try:
                trainer._load_from_checkpoint(str(ckpt))
            except Exception as exc:  # noqa: BLE001
                print("checkpoint reload failed:", exc)
        else:
            print("no finite checkpoint found; saved adapter may be NaN")
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


def _bind_processor_pixels(processor, min_pixels: int, max_pixels: int) -> None:
    """Newer Qwen2VLImageProcessor exposes min/max_pixels as read-only properties.

    Images are already resized; only mutate the writable size dict.
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    size = getattr(image_processor, "size", None)
    if not isinstance(size, dict):
        return
    if "shortest_edge" in size:
        size["shortest_edge"] = min_pixels
    if "longest_edge" in size:
        size["longest_edge"] = max_pixels
    size.pop("min_pixels", None)
    size.pop("max_pixels", None)


def train_unsloth(args: argparse.Namespace, train_rows: list) -> Path:
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    model, processor = FastVisionModel.from_pretrained(
        args.model_id,
        load_in_4bit=True,
        max_seq_length=args.max_seq,
        use_gradient_checkpointing="unsloth",
    )
    _bind_processor_pixels(processor, TRAIN_MIN_PIXELS, TRAIN_MAX_PIXELS)
    tokenizer = getattr(processor, "tokenizer", processor)
    train_rows = filter_to_budget(tokenizer, train_rows, args.max_seq)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    FastVisionModel.for_training(model)
    report = "wandb" if os.environ.get("WANDB_MODE") not in {"", "disabled"} and os.environ.get("WANDB_API_KEY") else "none"
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(
            model,
            processor,
            max_seq_length=args.max_seq,
            resize="max",
            completion_only_loss=True,
        ),
        train_dataset=train_rows,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=_warmup_steps(len(train_rows)),
            max_grad_norm=1.0,
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=args.output,
            report_to=report,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            fp16=True,
            bf16=False,
            dataloader_num_workers=0,
            save_strategy="steps",
            save_steps=5,
            save_total_limit=4,
            logging_nan_inf_filter=False,
            **_sft_length_kwargs(args.max_seq),
        ),
        callbacks=[_nan_callback()],
    )
    trainer.train()
    out = Path(args.output)
    _save_finite_adapter(model, processor, trainer, out)
    return out


def train_hf(args: argparse.Namespace, train_rows: list) -> Path:
    import gc

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=TRAIN_MIN_PIXELS, max_pixels=TRAIN_MAX_PIXELS
    )
    _bind_processor_pixels(processor, TRAIN_MIN_PIXELS, TRAIN_MAX_PIXELS)
    train_rows = filter_to_budget(processor.tokenizer, train_rows, args.max_seq)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    report = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    trainer_kwargs = {"model": model, "train_dataset": train_rows}
    sft_kwargs = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "warmup_steps": max(1, int(0.05 * max(1, math.ceil(len(train_rows) / 8)))),
        "max_grad_norm": 0.3,
        "logging_steps": 1,
        "output_dir": args.output,
        "report_to": report,
        "remove_unused_columns": False,
        "fp16": True,
        "dataloader_num_workers": 0,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }
    sft_kwargs.update(_sft_length_kwargs(args.max_seq))
    try:
        from unsloth.trainer import UnslothVisionDataCollator

        trainer_kwargs["tokenizer"] = processor
        trainer_kwargs["data_collator"] = UnslothVisionDataCollator(
            model, processor, max_seq_length=args.max_seq, resize="max", completion_only_loss=True
        )
    except Exception:
        trainer_kwargs["processing_class"] = processor
    trainer = SFTTrainer(args=SFTConfig(**sft_kwargs), **trainer_kwargs)
    trainer.train()
    out = Path(args.output)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    return out


def maybe_push(out: Path, repo: str) -> None:
    if os.environ.get("HF_PUSH") != "1":
        print(f"adapter saved at {out}. To push: HF_PUSH=1 huggingface-cli login && ...")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, exist_ok=True, private=True)
    api.upload_folder(folder_path=str(out), repo_id=repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"),
    )
    parser.add_argument("--output", default="outputs/lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-seq", type=int, default=TRAIN_MAX_SEQ)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    if not _has_cuda():
        print("No GPU. Training code is here; run this script on a T4 (Colab/Kaggle).")
        print("See notebooks/01_train_and_eval.ipynb for the exact Colab steps.")
        return

    dataset = load_cord()
    train_rows = convert_split(dataset["train"])
    if args.max_samples:
        train_rows = train_rows[: args.max_samples]
    config = {
        "model_id": args.model_id,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bits": 4,
        "train_vision_tokens": TRAIN_VISION_TOKENS,
        "train_max_seq": args.max_seq,
        "train_size": len(train_rows),
        "val_size": len(dataset["validation"]),
        "learning_rate": args.learning_rate,
        "split_hash": "see data/splits.json",
    }
    maybe_wandb(config)
    try:
        out = train_unsloth(args, train_rows)
    except Exception as exc:  # noqa: BLE001
        print(f"Unsloth failed ({exc}); falling back to peft+trl+bitsandbytes")
        out = train_hf(args, train_rows)
    _log_gpu_memory()
    maybe_push(out, os.environ.get("HF_ADAPTER_REPO", "your-username/vlm-receipt-extraction-lora"))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _log_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"peak GPU memory: {peak:.2f} GiB")
            try:
                import wandb

                if wandb.run:
                    wandb.log({"peak_gpu_memory_gib": peak})
            except Exception:
                pass
    except Exception:
        pass


if __name__ == "__main__":
    main()
