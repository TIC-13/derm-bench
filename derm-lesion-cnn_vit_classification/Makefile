PY     ?= python3
CONFIG ?= ./configuration/config.yaml

.PHONY: all train summary plots run clean help

all: run

train:
	$(PY) -m scripts.model_train --config $(CONFIG)

summary:
	$(PY) -m scripts.results_summary --config $(CONFIG)

plots:
	$(PY) -m scripts.results_plots --config $(CONFIG)

run: train summary plots

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete || true

wrong_classifications:
	$(PY) -m scripts.wrong_classifications

help:
	@echo "Usage:"
	@echo "  make train               # run training"
	@echo "  make summary             # generate results summary"
	@echo "  make plots               # generate plots"
	@echo "  make run                 # train + summary + plots"
	@echo "  make CONFIG=... train    # use a different config"
	@echo "  make PY=python train     # use a different python"
	@echo "  make wrong_classifications # generate wrong classifications report"

