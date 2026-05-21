import yaml
import argparse

from src.eval.eval_pipeline import EvaluationPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", [])
    datasets = cfg.get("datasets", [])
    configs = cfg.get("configs", [])
    dataset_path = cfg.get("datasets_root_path", "../datasets")
    ollama_url = cfg.get("ollama_url", None)

    evaluator = EvaluationPipeline()
    evaluator.evaluation(models, datasets, configs, dataset_path, ollama_url)
