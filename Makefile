.PHONY: install run simulate train test lint clean

install:
	pip install -e ".[dev]"

run:
	streamlit run src/pipe_leak/dashboard/app.py

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
