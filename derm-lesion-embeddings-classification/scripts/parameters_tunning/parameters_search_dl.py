import argparse

from src.parameters_search.dl_search import DLPParameterSearch
from src.config.seed_setting import set_global_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna MLP hyperparameter search (early stopping with min_delta, 10 trials)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    set_global_seed()

    search = DLPParameterSearch(
        config_path=args.config,
        n_trials=10
    )
    search.run()
