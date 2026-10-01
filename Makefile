.PHONY: install test lint up down eval concurrency notebook-check samples

install:
	python3 -m pip install -e ".[dev]"

test:
	LOGFIRE_SEND_TO_LOGFIRE=false WANDB_MODE=disabled MODEL_BACKEND=dummy python3 -m pytest -q

lint:
	python3 -m ruff check app tests scripts eval training

up:
	docker compose up --build

down:
	docker compose down

eval:
	python3 scripts/evaluate.py

concurrency:
	python3 scripts/concurrency_test.py

notebook-check:
	python3 -c "import json,pathlib; [json.loads(p.read_text()) for p in pathlib.Path('notebooks').glob('*.ipynb')]; print('notebooks ok')"

samples:
	python3 scripts/generate_samples.py
