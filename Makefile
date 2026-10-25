.PHONY: help install dev fmt lint type test test-int eval run ui docker up down bootstrap openapi clean
PY ?= python

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1 | tr '\n' ' '; echo

install:
	$(PY) -m pip install -U pip && $(PY) -m pip install -e ".[dev]"

dev: install
	pre-commit install

fmt:
	ruff format src tests && ruff check --fix src tests

lint:
	ruff check src tests && ruff format --check src tests

type:
	mypy src

test:
	pytest -m "not llm and not integration" -q --cov=src/emrag --cov-report=term-missing

test-int:
	pytest -m integration -q

eval:
	$(PY) -m emrag.eval.run_eval --mode retrieval --golden tests/golden/v1.jsonl

run:
	uvicorn emrag.api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py

docker:
	docker build -t emrag:local .

up:
	docker compose up --build

down:
	docker compose down -v

bootstrap:
	$(PY) scripts/bootstrap_store.py

openapi:
	$(PY) scripts/export_openapi.py docs/api/openapi.json

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
