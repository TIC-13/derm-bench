import argparse

import yaml

from src.training.train_orquestrator import TrainOrquestrator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training with YAML configuration.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    trainer = TrainOrquestrator(
        models=config["models"],
        datasets=config["datasets"],
        classes=config["classes"],
        hyperparams=config["hyperparams"],
        runs_dir=config["runs_output_path"],
        datasets_root=config["dataset_path"]
    )
    trainer.run_all()