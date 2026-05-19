from pathlib import Path

import json
import yaml
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.embeddings.data_utils import DataUtils
from src.embeddings.undersampling import UndersamplingUtils

from src.architectures.heads.custom_ml import (
    RandomForestModel,
    XGBoostModel,
    SVMModel,
)


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


class MLParameterSearch:
    def __init__(
        self,
        config_path: str,
        n_trials: int = 10,
        cv_splits: int = 5,
        scoring: str = "f1_macro",
        random_state: int = 42,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.resources = self.config["resources"]
        self.datasets = self.config["datasets"]
        self.models_to_run: List[str] = self.config.get("models", [])

        self.undersampling_cfg = self.config.get("undersampling", {})
        self.n_trials = int(n_trials)
        self.cv_splits = int(cv_splits)
        self.scoring = scoring
        self.random_state = int(random_state)

        self.out_root = Path("results_pure") / "parameters search" / "ml"
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.model_map = {
            "random_forest": RandomForestModel,
            "xgboost": XGBoostModel,
            "svm": SVMModel,
        }

        print(
            f"[Init] MLParameterSearch | trials={self.n_trials} | cv_splits={self.cv_splits} "
            f"| scoring={self.scoring} | seed={self.random_state}"
        )

    def _resolve_models_for_resource(self, resource: Dict[str, Any]) -> List[str]:
        """Resolve which models to run for this resource/dataset loop."""
        if self.models_to_run:
            return self.models_to_run
        if "models" in resource:
            return list(resource["models"])
        if "model_name" in resource:
            return [resource["model_name"]]
        return list(self.model_map.keys())

    def _load_splits_and_encode(
        self, dataset_path: str, dataset: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
        """
        Load train/validation DataFrames, apply optional undersampling to train,
        encode labels consistently, and return (X_train, y_train, X_val, y_val, label_encoder).
        """
        train_df = DataUtils.load_and_clean_h5(dataset_path, dataset, "train")
        val_df = DataUtils.load_and_clean_h5(dataset_path, dataset, "validation")

        train_df = UndersamplingUtils.apply_undersampling_if_enabled(train_df, self.undersampling_cfg)

        label_encoder = LabelEncoder()
        DataUtils.encode_labels(label_encoder, train_df, val_df)

        X_train, y_train = DataUtils.load_embeddings_and_labels(train_df)
        X_val, y_val = DataUtils.load_embeddings_and_labels(val_df)
        return X_train, y_train, X_val, y_val, label_encoder

    def _evaluate_on_validation(
        self,
        estimator,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Compute metrics on validation set; AUC only if predict_proba is available."""
        y_pred = estimator.predict(X_val)

        try:
            y_prob_pos = estimator.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob_pos)
        except Exception:
            auc = float("nan")

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision_macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
            "auc": auc,
        }
        return {k: float(v) for k, v in metrics.items()}

    def run(self):
        summary: List[Dict[str, Any]] = []

        for resource_key, resource in self.resources.items():
            dataset_path = resource["dataset_path"]
            backbone_name = resource.get("model_name", "<unknown>")
            out_base = self.out_root / _safe_name(resource_key)
            out_base.mkdir(parents=True, exist_ok=True)

            model_names = self._resolve_models_for_resource(resource)

            for dataset in self.datasets:
                print(
                    f"\n=== OPTUNA ML SEARCH ===\n"
                    f"Resource: {resource_key} | Backbone: {backbone_name} | Dataset: {dataset}\n"
                    f"Trials={self.n_trials} | CV={self.cv_splits} | Scoring={self.scoring}\n"
                    f"Output: {out_base}"
                )

                X_train, y_train, X_val, y_val, label_encoder = self._load_splits_and_encode(
                    dataset_path, dataset
                )
                print(
                    f"[Data] train={X_train.shape}, val={X_val.shape} | "
                    f"classes={list(map(str, label_encoder.classes_))}"
                )

                for model_name in model_names:
                    if model_name not in self.model_map:
                        print(f"[Skip] Unknown model: {model_name}")
                        continue

                    ModelCls = self.model_map[model_name]
                    model_wrapper = ModelCls()

                    out_dir = out_base / f"{_safe_name(model_name)}_{_safe_name(dataset)}"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    print(
                        f"[RUN] Algo={model_name} | Resource={resource_key} | Dataset={dataset} | "
                        f"Backbone={backbone_name} | trials={self.n_trials} | cv={self.cv_splits} | "
                        f"scoring={self.scoring}"
                    )

                    best_params = model_wrapper.tune_with_optuna(
                        X=X_train,
                        y=y_train,
                        n_trials=self.n_trials,
                        save_path=str(out_dir / "best_params.pkl"),
                        scoring=self.scoring,
                        cv_splits=self.cv_splits,
                        random_state=self.random_state,
                    )

                    estimator = model_wrapper.build_model(best_params)
                    estimator.fit(X_train, y_train)

                    model_path = out_dir / "best_model.pkl"
                    joblib.dump(estimator, model_path)
                    print(f"[Saved] Trained model → {model_path}")

                    metrics_val = self._evaluate_on_validation(estimator, X_val, y_val)
                    print(
                        f"[RESULT] Algo={model_name} | Resource={resource_key} | Dataset={dataset} | "
                        f"f1={metrics_val['f1_macro']:.4f} acc={metrics_val['accuracy']:.4f} "
                        f"prec={metrics_val['precision_macro']:.4f} rec={metrics_val['recall_macro']:.4f} "
                        f"auc={metrics_val['auc']:.4f}"
                    )

                    result = {
                        "resource": resource_key,
                        "backbone": backbone_name,
                        "model_name": model_name,
                        "dataset": dataset,
                        "n_trials": int(self.n_trials),
                        "cv_splits": int(self.cv_splits),
                        "scoring": self.scoring,
                        "class_order": list(map(str, label_encoder.classes_)),
                        "best_params": best_params,
                        "metrics_on_validation": {k: round(v, 6) for k, v in metrics_val.items()},
                    }

                    out_json = out_dir / "best_result.json"
                    with open(out_json, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"[Saved] {out_json}")

                    summary.append(result)

        summary_path = self.out_root / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Saved summary] {summary_path}")
