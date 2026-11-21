.PHONY: data download prepare baseline train report test lint

download:
	python -m src.download

prepare:
	python -m src.prepare

data: download prepare

baseline:
	python -m src.evaluate --all

train:
	python -m src.train --run qwen2.5-0.5b-lora --evaluate

report:
	python -m src.report

test:
	pytest -q

lint:
	ruff check .
