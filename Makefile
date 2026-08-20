.PHONY: setup api test train
setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
api:
	uvicorn backend.app.main:app --reload --port 8000
test:
	pytest -q
train:
	python -m ml.training.train --config ml/training/config.yaml
