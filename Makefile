.PHONY: data download prepare baseline report test lint

download:
	python -m src.download

prepare:
	python -m src.prepare

data: download prepare

baseline:
	python -m src.evaluate --all

report:
	python -m src.report

test:
	pytest -q

lint:
	ruff check .
