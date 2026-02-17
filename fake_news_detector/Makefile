.PHONY: lint test build all clean

# ─── Code Quality ──────────────────────────────────────
lint:
	flake8 fake_news_detector/ \
		--max-line-length=120 \
		--extend-ignore=E402,W503 \
		--count \
		--statistics

# ─── Tests ─────────────────────────────────────────────
test:
	pytest fake_news_detector/tests/ \
		-v \
		--tb=short \
		--cov=fake_news_detector \
		--cov-report=term-missing

# ─── Docker ────────────────────────────────────────────
build:
	docker build -t fake-news-detector:local .

# ─── Train Models ──────────────────────────────────────
train-nb:
	python scripts/train_nb.py

train-svm:
	python scripts/train_svm.py

# ─── Combined ─────────────────────────────────────────
all: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache coverage.xml htmlcov/
