.PHONY: help install install-ollama pull-models dataset run eval summary plots all

help:
	@echo "Available commands:"
	@echo "  make install         		-> Install Python dependencies"
	@echo "  make install-ollama  		-> Install Ollama"
	@echo "  make pull-models     		-> Download Ollama models"
	@echo "  make dataset-merge   		-> Process / merge datasets"
	@echo "  make eval            		-> Run evaluation pipeline"
	@echo "  make summary         		-> Generate results summary"
	@echo "  make plots           		-> Generate result plots"
	@echo "  make alucinations_plots    -> Generate alucinations plots"

install:
	pip install -r requirements.txt
	pip install -U transformers

install-ollama:
	curl -fsSL https://ollama.com/install.sh | sh

pull-models:
	ollama pull llama3.2-vision:11b
	ollama pull llama3.2-vision:90b
	ollama pull llama4:16x17b
	ollama pull qwen2.5vl:72b
	ollama pull gemma3:27b

dataset-merge:
	python3 scripts/merge_datasets.py --config ./configuration_yaml/setup_config.yaml

eval:
	python3 -m scripts.evaluation --config ./configuration_yaml/setup_config.yaml

summary:
	python3 -m scripts.results_summary --config ./configuration_yaml/setup_config.yaml

plots:
	python3 -m scripts.results_plots --config ./configuration_yaml/setup_config.yaml

alucinations_plots:
	python3 -m scripts.alucination_plots --config configuration_yaml/setup_config.yaml