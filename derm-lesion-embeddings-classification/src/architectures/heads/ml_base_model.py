import os
import joblib
from typing import Any, Dict, Optional

import optuna
import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold


class BaseModel:
    def fit(self, X, y):
        """
        Fit the underlying model using the input features and target labels.

        Args:
            X: Input feature matrix with shape (n_samples, n_features).
            y: Target labels with shape (n_samples,).

        Returns:
            None.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict class labels for the input features.

        Args:
            X: Input feature matrix with shape (n_samples, n_features).

        Returns:
            Predicted class labels with shape (n_samples,).
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for the input features.

        Args:
            X: Input feature matrix with shape (n_samples, n_features).

        Returns:
            Predicted class probabilities with shape (n_samples, n_classes).
        """
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
        """
        Tune the model hyperparameters using Optuna and cross-validation.

        Args:
            X: Input feature matrix with shape (n_samples, n_features).
            y: Target labels with shape (n_samples,).
            n_trials: Number of Optuna trials to run.
            save_path: Optional path where the best parameters will be saved.
            scoring: Scikit-learn scoring metric used during cross-validation.
            cv_splits: Number of stratified cross-validation folds.
            random_state: Random seed used for reproducibility.

        Returns:
            Dictionary containing the best hyperparameters found by Optuna.
        """
        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        if not hasattr(self, "define_search_space"):
            raise NotImplementedError("Subclass must implement define_search_space(trial).")

        def objective(trial):
            params = self.define_search_space(trial)
            model = self.build_model(params)

            cv = StratifiedKFold(
                n_splits=cv_splits,
                shuffle=True,
                random_state=random_state
            )

            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1    
            )

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