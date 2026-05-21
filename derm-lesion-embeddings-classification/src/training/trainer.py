from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder

matplotlib.use('Agg')

from src.architectures.heads.custom_ml import RandomForestModel, XGBoostModel, SVMModel
from src.architectures.heads.custom_mlp import MLPClassifier
from src.mlp_trainer.trainer import MLPTrainer
from src.mlp_trainer.dataloader import MLPDatasetLoader
from src.metrics.metrics_manager import MetricsManager
from src.embeddings.data_utils import DataUtils
from src.embeddings.undersampling import UndersamplingUtils

class TrainingPipeline:
    def __init__(
            self,
            resources,
            datasets,
            models,
            undersampling_cfg: dict = None,
            smote: dict = None
    ) -> None:
        self.resources = resources
        self.datasets = datasets
        self.models = models
        self.undersampling_cfg = undersampling_cfg or {}
        self.smote = smote or {}
        self.metrics_manager = MetricsManager()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_encoder = LabelEncoder()

        self.model_classes = {
            "random_forest": RandomForestModel,
            "xgboost": XGBoostModel,
            "svm": SVMModel,
        }

    def _get_resource_paths(
        self,
        resource: dict,
    ) -> tuple[str, str, int | None]:
        dataset_path = resource["dataset_path"]
        output_path = resource["train_output_path"]
        input_dim = resource.get("input_dim")

        return dataset_path, output_path, input_dim

    def _build_output_path(
        self,
        output_path: str,
        dataset: str,
        model_name: str,
    ) -> Path:
        return Path(output_path) / dataset / model_name

    def _infer_id_col(
            self,
            df: pd.DataFrame
        ) -> str:
        candidates = ["img_id", "image_id", "filename", "file_name", "path", "filepath", "file", "id", "image", "img"]
        for c in candidates:
            if c in df.columns:
                return c
        return df.columns[0]

    def _save_predictions_csv(
        self,
        test_df: pd.DataFrame,
        y_pred: np.ndarray,
        probs: np.ndarray,
        output_dir: Path,
    ) -> Path:
        id_col = self._infer_id_col(test_df)

        true_names = self.label_encoder.inverse_transform(
            test_df["benign_malignant"].astype(int).to_numpy()
        )
        pred_names = self.label_encoder.inverse_transform(np.asarray(y_pred).astype(int))

        out_df = pd.DataFrame({
            id_col: test_df[id_col].astype(str).to_numpy(),
            "true_label": true_names,
            "pred_label": pred_names,
            "prob": np.asarray(probs) if probs is not None else np.nan,
        })

        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "predictions.csv"
        out_df.to_csv(csv_path, index=False)
        print(f"[OK] Predictions CSV saved to: {csv_path}")
        return csv_path
    
    def _run_single_experiment(self, resource: dict, dataset: str, model_name: str) -> None:
        dataset_path, output_path, input_dim = self._get_resource_paths(resource)
        current_output_path = self._build_output_path(output_path, dataset, model_name)

        if current_output_path.exists():
            return

        print(f"\n=== {model_name.upper()} on {dataset} ===")

        train_df, val_df, test_df = self._load_prepare_data(dataset_path, dataset)

        if model_name == "mlp":
            results = self._train_eval_mlp(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                input_dim=input_dim,
                dataset=dataset,
                model_name=model_name,
                output_path=current_output_path,
            )
        else:
            results = self._train_eval_classic_ml(
                train_df=train_df,
                test_df=test_df,
                model_name=model_name,
                output_path=current_output_path,
            )

        self._save_common_outputs(
            results=results,
            test_df=test_df,
            dataset=dataset,
            model_name=model_name,
            output_path=current_output_path,
        )
    
    def _load_prepare_data(
        self,
        dataset_path: str,
        dataset: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = DataUtils.load_and_clean_h5(dataset_path, dataset, "train")
        val_df = DataUtils.load_and_clean_h5(dataset_path, dataset, "validation")
        test_df = DataUtils.load_and_clean_h5(dataset_path, dataset, "test")

        train_df = UndersamplingUtils.apply_undersampling_if_enabled(
            train_df,
            self.undersampling_cfg,
        )

        self.label_encoder = LabelEncoder()

        DataUtils.encode_labels(
            self.label_encoder,
            train_df,
            val_df,
            test_df,
        )

        return train_df, val_df, test_df
    
    def _train_eval_mlp(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        input_dim: int,
        dataset: str,
        model_name: str,
        output_path: Path,
    ) -> dict:
        data_loader = MLPDatasetLoader(
            train_df,
            val_df,
            test_df,
            use_smote=self.smote.get("enabled"),
            smote_ratio=self.smote.get("ratio"),
            smote_seed=self.smote.get("random_state"),
        )

        train_loader, val_loader, test_loader = data_loader.get_loaders()

        model = MLPClassifier(
            input_dim=input_dim,
        ).to(self.device)

        trainer = MLPTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=self.device,
        )

        trainer.train()
        print("Training completed!")

        output_path.mkdir(parents=True, exist_ok=True)
        model_save_path = output_path / "mlp_model.pth"
        torch.save(model.state_dict(), model_save_path)

        test_acc, test_prec, test_rec, test_f1, test_report, y_test, y_test_pred, test_probs, _ = trainer.test(
            label_encoder=self.label_encoder
        )

        self.metrics_manager.save_roc_curve(
            y_true=y_test,
            y_score=test_probs,
            output_path=output_path,
            title=f"ROC Curve - {model_name} ({dataset})",
        )

        pred_class_prob = np.where(
            np.asarray(y_test_pred) == 1,
            np.asarray(test_probs),
            1.0 - np.asarray(test_probs),
        )

        return {
            "acc": test_acc,
            "prec": test_prec,
            "rec": test_rec,
            "f1": test_f1,
            "report": test_report,
            "y_true": y_test,
            "y_pred": y_test_pred,
            "probs": pred_class_prob,
        }
    
    def _train_eval_classic_ml(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_name: str,
        output_path: Path,
    ) -> dict:
        X_train, y_train = DataUtils.load_embeddings_and_labels(train_df)
        X_test, y_test = DataUtils.load_embeddings_and_labels(test_df)

        model = self.model_classes[model_name]()
        tuned_model = model.build_model(model.default_params)
        tuned_model.fit(X_train, y_train)

        output_path.mkdir(parents=True, exist_ok=True)
        model_save_path = output_path / f"{model_name}_model.pkl"
        joblib.dump(tuned_model, model_save_path)

        y_test_pred = tuned_model.predict(X_test)

        try:
            proba_matrix = tuned_model.predict_proba(X_test)
            pred_class_prob = proba_matrix[np.arange(len(y_test_pred)), y_test_pred]
        except Exception:
            pred_class_prob = np.full_like(y_test_pred, fill_value=np.nan, dtype=np.float64)

        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

        test_report = classification_report(
            y_test,
            y_test_pred,
            target_names=self.label_encoder.classes_,
        )

        return {
            "acc": test_acc,
            "prec": test_prec,
            "rec": test_rec,
            "f1": test_f1,
            "report": test_report,
            "y_true": y_test,
            "y_pred": y_test_pred,
            "probs": pred_class_prob,
        }
    
    def _save_common_outputs(
        self,
        results: dict,
        test_df: pd.DataFrame,
        dataset: str,
        model_name: str,
        output_path: Path,
    ) -> None:
        self._save_predictions_csv(
            test_df=test_df,
            y_pred=results["y_pred"],
            probs=results["probs"],
            output_dir=output_path,
        )

        self.metrics_manager.save_confusion_matrix(
            y_true=results["y_true"],
            y_pred=results["y_pred"],
            classes=self.label_encoder.classes_,
            output_path=output_path,
        )

        self.metrics_manager.save_report(
            output_path=output_path,
            model_name=f"{model_name}_test",
            dataset_name=dataset,
            classes=self.label_encoder.classes_,
            acc=results["acc"],
            prec=results["prec"],
            rec=results["rec"],
            f1=results["f1"],
            report=results["report"],
        )
    
    def run(self):
        for _, resource in self.resources.items():
            for dataset in self.datasets:
                for model_name in self.models:
                    self._run_single_experiment(resource, dataset, model_name)