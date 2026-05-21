.PHONY: help embeddings plots param-search training summary plots

help:
	@echo "  help                Show this help message"
	@echo "  embeddings          Generate embedding dataset"
	@echo "  param-search        Run hyperparameter search (DL + ML)"
	@echo "  training            Train classifiers"
	@echo "  summary             Generate training summaries and plots"
	@echo "  plots 			 	 Generate metrics plots (by dataset and label)"
	@echo "  wrong-predictions   Generate wrong predictions analysis"

embeddings:
	@echo "Generating embedding dataset..."
	python3 -m scripts.dataset.embedding_dataset --config ./configuration/config.yaml

param-search:
	@echo "Running hyperparameter search (DL)..."
	python3 -m scripts.parameters_tunning.parameters_search_dl --config ./configuration/config.yaml
	@echo "Running hyperparameter search (ML)..."
	python3 -m scripts.parameters_tunning.parameters_search_ml --config ./configuration/config.yaml

training:
	@echo "Training classifiers..."
	python3 -m scripts.pipeline.classificator_training --config ./configuration/config.yaml
	rm -rf ./runs/

summary:
	@echo "Generating training summary..."
	python3 -m scripts.pipeline.metrics_summary --config ./configuration/config.yaml

plots:
	python3 -m scripts.pipeline.plot_summary --config ./configuration/config.yaml

wrong-predictions:
	@echo "Generating wrong predictions..."
	python3 -m scripts.pipeline.wrong_predictions --config ./configuration/config.yaml
