# CUDA runtime for the API image. CPU-only: --build-arg BASE_IMAGE=python:3.11-slim-bookworm
ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_BACKEND=dummy \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv tesseract-ocr libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY app ./app
COPY samples ./samples
COPY scripts ./scripts
COPY eval ./eval
COPY training ./training
RUN pip3 install --no-cache-dir -r requirements.txt && pip3 install --no-cache-dir -e .

# Weights are downloaded at startup, not baked in.
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
