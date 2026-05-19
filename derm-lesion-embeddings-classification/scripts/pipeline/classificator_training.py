import argparse

import yaml

from src.training.trainer import TrainingPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training pipeline runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_pipeline = TrainingPipeline(
        resources=config["resources"],
        datasets=config["datasets"],
        models=config["models"],
        undersampling_cfg=config.get("undersampling", {})
    )
    train_pipeline.run()
