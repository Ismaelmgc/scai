PYTHON = .venv/bin/python3
PIP = $(PYTHON) -m pip

.PHONY: install lint format typecheck test test-cov run-api download-data build-features train backtest

install:
	$(PIP) install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/unit -v

test-all:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=app --cov-report=html

run-api:
	PYTHONPATH=src $(PYTHON) -m uvicorn app.web.server:app --host 127.0.0.1 --port 8501 --reload

download-data:
	$(PYTHON) scripts/download_data.py

build-features:
	$(PYTHON) scripts/build_features.py

train:
	$(PYTHON) scripts/train_models.py

backtest:
	$(PYTHON) scripts/run_backtest.py

real-backtest:
	$(PYTHON) scripts/run_real_backtest.py

report:
	$(PYTHON) scripts/generate_report.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf htmlcov/ .coverage
