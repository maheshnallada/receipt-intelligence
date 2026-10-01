from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("MODEL_BACKEND", "dummy")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
os.environ.setdefault("LOGFIRE_TOKEN", "")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_API_KEY", "")

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture
def sample_ok() -> bytes:
    return (SAMPLES / "receipt_ok.pdf").read_bytes()


@pytest.fixture
def sample_mismatch() -> bytes:
    return (SAMPLES / "receipt_mismatch.pdf").read_bytes()


@pytest.fixture
def sample_multipage() -> bytes:
    return (SAMPLES / "receipt_multipage.pdf").read_bytes()
