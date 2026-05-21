import argparse

import yaml

from src.training.trainer import TrainingPipeline
from src.config.seed_setting import set_global_seed


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

    set_global_seed()

    train_pipeline = TrainingPipeline(
        resources=config["resources"],
        datasets=config["datasets"],
        models=config["models"],
        undersampling_cfg=config.get("undersampling", {}),
        smote=config.get("smote", {})
    )
    train_pipeline.run()
