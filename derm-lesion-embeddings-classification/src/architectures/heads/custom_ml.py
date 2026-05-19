import os
import joblib
import optuna
import numpy as np
import random
from typing import Any, Dict, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb


# Base class
class BaseModel:
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model does not support probability prediction.")

    def tune_with_optuna(
        self,
        X,
        y,
        n_trials: int = 5,
        save_path: Optional[str] = None,
        scoring: str = "f1_macro",
        cv_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:

        # Reproducibility setup
        np.random.seed(random_state)
        random.seed(random_state)
        try:
            import torch
            torch.manual_seed(random_state)
        except ImportError:
            pass

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        if not hasattr(self, "define_search_space"):
            raise NotImplementedError("Subclass must implement define_search_space(trial).")

        def objective(trial):
            params = self.define_search_space(trial)
            model = self.build_model(params)

            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return np.mean(scores)

        study.optimize(objective, n_trials=n_trials)

        print("\n===== Optuna tuning finished =====")
        print(f"Best {scoring}: {study.best_value:.4f}")
        print("Best parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(study.best_params, save_path)
            print(f"Saved best parameters to {save_path}")

        self.best_params = study.best_params
        return study.best_params

    def build_model(self, params: Dict[str, Any]):
        """To be implemented in subclasses (returns configured estimator)."""
        raise NotImplementedError


# Random Forest
class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        self.default_params = dict(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        self.default_params.update(kwargs)
        self.model = RandomForestClassifier(**self.default_params)

    def define_search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }

    def build_model(self, params):
        merged = {**self.default_params, **params}
        return RandomForestClassifier(**merged)


# XGBoost
class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        self.default_params = dict(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=1.0,
        )
        self.default_params.update(kwargs)
        self.model = xgb.XGBClassifier(**self.default_params)

    def define_search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    def build_model(self, params):
        merged = {**self.default_params, **params}
        return xgb.XGBClassifier(**merged)


# SVM
class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        self.default_params = dict(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
            class_weight="balanced",
        )
        self.default_params.update(kwargs)
        self.model = SVC(**self.default_params)

    def define_search_space(self, trial):
        return {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        }

    def build_model(self, params):
        merged = {**self.default_params, **params}
        return SVC(**merged)
