# Local developer shortcuts. Production runs in GitHub Actions (.github/workflows/).
PYTHON ?= python
export PYTHONPATH := src
export PYTHONIOENCODING := utf-8

.PHONY: install lint format typecheck test test-cov web dashboard-static \
        smallcap-dry liquidcap-dry btc-dry bounce-dry leak-check clean

install:          ## editable install + dev tools
	$(PYTHON) -m pip install -e ".[dev]"

lint:             ## what CI runs
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

test:             ## what CI runs
	$(PYTHON) -m pytest tests/unit -q

test-cov:
	$(PYTHON) -m pytest tests/unit --cov=app --cov-report=term-missing

web:              ## local dashboard on http://localhost:8501
	$(PYTHON) -m uvicorn app.web.server:app --host 127.0.0.1 --port 8501

dashboard-static: ## render the Pages login shell into site/
	$(PYTHON) scripts/render_static_dashboard.py

smallcap-dry:     ## small-cap model pipeline, no writes (needs the committed OHLCV panel)
	$(PYTHON) scripts/daily_pipeline.py --dry-run --skip-download

liquidcap-dry:    ## S&P 500 book, no writes
	$(PYTHON) scripts/liquidcap/daily_liquidcap.py --dry-run

btc-dry:          ## BTC <-> managed-futures book, no writes
	$(PYTHON) scripts/btc_trend_daily.py --dry-run

bounce-dry:       ## S&P 500 dip-reversion shadow book, no writes
	$(PYTHON) scripts/bounce_daily.py --dry-run

leak-check:       ## anti-leakage gate the production retrain must pass
	$(PYTHON) scripts/v3/18_verify_no_leak.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
