.PHONY: data download prepare baseline train ablations report analyze test lint

download:
	python -m src.download

prepare:
	python -m src.prepare

data: download prepare

baseline:
	python -m src.evaluate --all

train:
	python -m src.train --run qwen2.5-0.5b-lora --evaluate

# one process per run, so each starts with the full memory of the 8 GB machine
ABLATIONS = qwen2.5-1.5b-lora-2k qwen2.5-0.5b-lora-2k qwen2.5-0.5b-lora-1k qwen2.5-0.5b-lora-2k-r4 qwen2.5-0.5b-lora-2k-r16

ablations:
	for run in $(ABLATIONS); do python -m src.train --run $$run --evaluate || exit 1; done

report:
	python -m src.report

analyze:
	python -m src.analyze

test:
	pytest -q

lint:
	ruff check .
