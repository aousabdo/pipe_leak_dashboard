.PHONY: install install-frontend dev backend frontend simulate train test lint format clean

install:
	pip install -e ".[dev]"

install-frontend:
	cd frontend && npm install --legacy-peer-deps

dev: ## Start both backend and frontend (use two terminals, or run each separately)
	@echo "Run in two terminals:"
	@echo "  make backend"
	@echo "  make frontend"

backend:
	uvicorn pipe_leak.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

simulate:
	python scripts/run_simulation.py

train:
	python scripts/train_model.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

clean:
	rm -rf data/processed/*.parquet models/*.joblib
	find . -type d -name __pycache__ -exec rm -rf {} +
