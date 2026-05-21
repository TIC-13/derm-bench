import os
import joblib
from typing import Any, Dict, Optional

import optuna

from sklearn.metrics import get_scorer


class BaseModel:
    def fit(self, X, y):
        """
        Fit the underlying model using the given training data.

        Args:
            X: Training feature matrix with shape (n_samples, n_features).
            y: Training labels with shape (n_samples,).

        Returns:
            None.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict class labels for the given input samples.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            Predicted class labels with shape (n_samples,).
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input samples.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            Predicted class probabilities with shape
            (n_samples, n_classes).

        Raises:
            NotImplementedError: If the underlying model does not support
            probability prediction.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model does not support probability prediction.")

    def tune_with_optuna(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials: int = 5,
        save_path: Optional[str] = None,
        scoring: str = "f1_macro",
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters using Optuna and a fixed validation set.

        For each Optuna trial, this method samples a set of hyperparameters,
        builds a model using those parameters, trains it on the training set,
        and evaluates it on the validation set using the specified scikit-learn
        scoring metric.

        Args:
            X_train: Training feature matrix with shape
                (n_train_samples, n_features).
            y_train: Training labels with shape (n_train_samples,).
            X_val: Validation feature matrix with shape
                (n_val_samples, n_features).
            y_val: Validation labels with shape (n_val_samples,).
            n_trials: Number of Optuna trials to run.
            save_path: Optional path where the best hyperparameters will be
                saved using joblib.
            scoring: Scikit-learn scoring metric used to evaluate each trial.
            random_state: Random seed used by the Optuna sampler for
                reproducibility.

        Returns:
            A dictionary containing the best hyperparameters found by Optuna.

        Raises:
            NotImplementedError: If the subclass does not implement
            define_search_space(trial).
        """

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        if not hasattr(self, "define_search_space"):
            raise NotImplementedError("Subclass must implement define_search_space(trial).")

        scorer = get_scorer(scoring)

        def objective(trial):
            params = self.define_search_space(trial)
            model = self.build_model(params)

            model.fit(X_train, y_train)

            score = scorer(model, X_val, y_val)

            return score

        study.optimize(objective, n_trials=n_trials)

        print("\n===== Optuna tuning finished =====")
        print(f"Best {scoring}: {study.best_value:.4f}")
        print("Best parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            joblib.dump(study.best_params, save_path)
            print(f"Saved best parameters to {save_path}")

        self.best_params = study.best_params
        return study.best_params

    def build_model(self, params: Dict[str, Any]):
        """
        Build and return a configured model instance.

        This method must be implemented by subclasses. It should receive a
        dictionary of hyperparameters and return a scikit-learn-compatible
        estimator.

        Args:
            params: Dictionary containing model hyperparameters.

        Returns:
            A configured estimator instance.

        Raises:
            NotImplementedError: Always raised in the base class.
        """
        raise NotImplementedError