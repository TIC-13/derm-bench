import argparse

from src.parameters_search.ml_search import MLParameterSearch
from src.config.seed_setting import set_global_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna ML hyperparameter search (RandomForest/XGBoost/SVM) with CV"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials (default: 10)")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV splits (default: 5)")
    parser.add_argument("--scoring", type=str, default="f1_macro", help="sklearn scoring metric (default: f1_macro)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    set_global_seed()

    search = MLParameterSearch(
        config_path=args.config,
        n_trials=args.trials,
        cv_splits=args.cv,
        scoring=args.scoring,
        random_state=args.seed,
    )
    search.run()
