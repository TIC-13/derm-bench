import os
import json
import yaml
import optuna
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.mlp_trainer.trainer import MLPTrainer
from src.mlp_trainer.dataloader import MLPDatasetLoader
from src.embeddings.data_utils import DataUtils
from src.embeddings.undersampling import UndersamplingUtils


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")

class DynamicMLPFactory:
    ACTIVATIONS = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "leaky_relu": lambda: torch.nn.LeakyReLU(negative_slope=0.1),
    }

    @staticmethod
    def build(
        input_dim: int,
        n_hidden_layers: int,
        hidden_units: int,
        width_scheme: str,      # "constant" | "pyramid"
        activation: str,        # "relu" | "gelu" | "leaky_relu"
        use_batch_norm: bool,
        use_dropout: bool,
        dropout: float,
    ) -> torch.nn.Module:
        Act = DynamicMLPFactory.ACTIVATIONS[activation]

        if n_hidden_layers <= 0:
            hidden_sizes = []
        else:
            if width_scheme == "constant":
                hidden_sizes = [hidden_units] * n_hidden_layers
            else:
                h1 = hidden_units
                h2 = max(hidden_units // 2, 16)
                h3 = max(hidden_units // 4, 8)
                hidden_sizes = [h1, h2, h3][:n_hidden_layers]

        layers = []
        in_dim = input_dim

        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, h))

            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(h))

            layers.append(Act())

            if use_dropout:
                layers.append(torch.nn.Dropout(dropout))

            in_dim = h

        layers.append(torch.nn.Linear(in_dim, 2))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def type_signature(hparams: Dict[str, Any]) -> str:
        use_do = bool(hparams.get("use_dropout", False))
        dropout_str = f"({hparams.get('dropout', 0.0):.2f})" if use_do else ""
        return (
            f"layers={hparams.get('n_hidden_layers', '?')}, "
            f"units={hparams.get('hidden_units', '?')}, "
            f"width={hparams.get('width_scheme', '?')}, "
            f"act={hparams.get('activation', '?')}, "
            f"bn={hparams.get('use_batch_norm', '?')}, "
            f"dropout={'on' if use_do else 'off'}{dropout_str}"
        )

class DLPParameterSearch:
    def __init__(
            self,
            config_path: str,
            n_trials: int = 10,
            patience: int = 10,
            min_delta: int = 1e-4
        ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.resources = self.config["resources"]
        self.datasets = self.config["datasets"]
        self.undersampling_cfg = self.config.get("undersampling", {})
        self.batch_size = self.config.get("extraction_bath_size", 512)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            try:
                print(f"[Device] CUDA enabled: {torch.cuda.get_device_name(0)}")
            except Exception:
                print("[Device] CUDA enabled")
        else:
            print("[Device] CUDA not available; using CPU")

        self.n_trials = n_trials
        self.patience = patience
        self.min_delta = min_delta

        self.out_root = Path("results_pure") / "parameters search" / "mlp"
        self.out_root.mkdir(parents=True, exist_ok=True)
        Path(".optuna_tmp").mkdir(parents=True, exist_ok=True)


    def _train_until_early_stop(self, trainer: MLPTrainer, train_loader, val_loader, ckpt_path: Path):
        """Early stopping with min_delta (no per-epoch printing)."""
        best_val = float("inf")
        no_improve = 0
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            trainer.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                trainer.optimizer.zero_grad()
                out = trainer.model(xb)
                loss = trainer.criterion(out, yb)
                loss.backward()
                trainer.optimizer.step()

            val_loss = trainer.validation()
            trainer.scheduler.step(val_loss)

            if (best_val - val_loss) > self.min_delta:
                best_val = val_loss
                no_improve = 0
                torch.save(trainer.model.state_dict(), ckpt_path)
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    trainer.model.load_state_dict(torch.load(ckpt_path))
                    break

    def _evaluate_loader(self, model: torch.nn.Module, loader) -> Dict[str, float]:
        """Compute metrics silently on a loader (no prints)."""
        model.eval()
        probs_all, preds_all, targets_all = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                probs_all.extend(probs.tolist())
                preds_all.extend(preds.tolist())
                targets_all.extend(yb.numpy().tolist())

        y_true = np.array(targets_all)
        y_prob = np.array(probs_all)
        y_pred = np.array(preds_all)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan"),
        }

    def _objective(
        self,
        trial: optuna.trial.Trial,
        input_dim: int,
        model_name: str,
        dataset: str,
        train_df,
        val_df,
    ) -> float:

        hparams = {
            "n_hidden_layers": trial.suggest_int("n_hidden_layers", 0, 3),
            "hidden_units": trial.suggest_categorical("hidden_units", [128, 256, 512, 1024]),
            "width_scheme": trial.suggest_categorical("width_scheme", ["constant", "pyramid"]),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu"]),
            "use_batch_norm": trial.suggest_categorical("use_batch_norm", [True, False]),
            "use_dropout": trial.suggest_categorical("use_dropout", [True, False]),
        }
        hparams["dropout"] = trial.suggest_float("dropout", 0.0, 0.6) if hparams["use_dropout"] else 0.0

        use_weight_decay = trial.suggest_categorical("use_weight_decay", [True, False])
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True) if use_weight_decay else 0.0

        train_hp = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
            "weight_decay": weight_decay,
            "use_weight_decay": use_weight_decay,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }

        print(
            f"[RUN] Trial#{trial.number} | Model={model_name} | Dataset={dataset} | "
            f"type={DynamicMLPFactory.type_signature(hparams)} | "
            f"opt={train_hp['optimizer']} lr={train_hp['lr']:.2e} "
            f"wd={'on' if use_weight_decay else 'off'} "
            f"patience={self.patience} min_delta={self.min_delta:.1e} "
            f"batch={self.batch_size}"
        )

        model = DynamicMLPFactory.build(
            input_dim=input_dim,
            n_hidden_layers=hparams["n_hidden_layers"],
            hidden_units=hparams["hidden_units"],
            width_scheme=hparams["width_scheme"],
            activation=hparams["activation"],
            use_batch_norm=hparams["use_batch_norm"],
            use_dropout=hparams["use_dropout"],
            dropout=hparams["dropout"],
        ).to(self.device)

        loader = MLPDatasetLoader(train_df, val_df, val_df, batch_size=self.batch_size)
        train_loader, val_loader, _ = loader.get_loaders()

        trainer = MLPTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,
            device=self.device,
            lr=train_hp["lr"],
            patience=self.patience,
        )
        if train_hp["optimizer"] == "adamw":
            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(),
                lr=train_hp["lr"],
                weight_decay=train_hp["weight_decay"]
            )
        else:
            trainer.optimizer = torch.optim.Adam(
                trainer.model.parameters(),
                lr=train_hp["lr"],
                weight_decay=train_hp["weight_decay"]
            )

        trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer,
            mode="min",
            patience=max(1, self.patience // 3),
            factor=0.5
        )

        ckpt = Path(".optuna_tmp") / f"best_{_safe_name(model_name)}_{_safe_name(dataset)}_trial{trial.number}.pth"
        trainer.best_model_path = str(ckpt)

        self._train_until_early_stop(trainer, train_loader, val_loader, ckpt)

        metrics = self._evaluate_loader(trainer.model, val_loader)
        print(f"[RESULT] Trial#{trial.number} | "
              f"f1={metrics['f1_macro']:.4f} acc={metrics['accuracy']:.4f} "
              f"prec={metrics['precision_macro']:.4f} rec={metrics['recall_macro']:.4f} "
              f"auc={metrics['auc']:.4f}")

        return float(metrics["f1_macro"])

    def _load_splits(self, dataset_path: str, dataset: str):
        train_df = DataUtils.load_and_clean_h5(dataset_path, dataset, "train")
        val_df = DataUtils.load_and_clean_h5(dataset_path, dataset, "validation")
        train_df = UndersamplingUtils.apply_undersampling_if_enabled(train_df, self.undersampling_cfg)
        label_encoder = LabelEncoder()
        DataUtils.encode_labels(label_encoder, train_df, val_df)
        return train_df, val_df, label_encoder

    def _retrain_and_report(
        self,
        best_params: Dict[str, Any],
        input_dim: int,
        model_name: str,
        dataset: str,
        train_df,
        val_df,
    ) -> Dict[str, float]:
        best_model = DynamicMLPFactory.build(
            input_dim=input_dim,
            n_hidden_layers=best_params["n_hidden_layers"],
            hidden_units=best_params["hidden_units"],
            width_scheme=best_params["width_scheme"],
            activation=best_params["activation"],
            use_batch_norm=best_params["use_batch_norm"],
            use_dropout=best_params["use_dropout"],
            dropout=best_params.get("dropout", 0.0),
        ).to(self.device)

        loader = MLPDatasetLoader(train_df, val_df, val_df, batch_size=self.batch_size)
        train_loader, val_loader, _ = loader.get_loaders()

        lr = best_params["lr"]
        optimizer = best_params["optimizer"]
        weight_decay = best_params.get("weight_decay", 0.0)

        trainer = MLPTrainer(
            model=best_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,
            device=self.device,
            lr=lr,
            patience=self.patience,
        )
        if optimizer == "adamw":
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr, weight_decay=weight_decay)
        trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer, mode="min", patience=max(1, self.patience // 3), factor=0.5
        )

        final_ckpt = Path(".optuna_tmp") / f"best_final_{_safe_name(model_name)}_{_safe_name(dataset)}.pth"
        trainer.best_model_path = str(final_ckpt)

        self._train_until_early_stop(trainer, train_loader, val_loader, final_ckpt)

        metrics = self._evaluate_loader(trainer.model, val_loader)
        print(f"[BEST] {model_name} | {dataset} | "
              f"f1={metrics['f1_macro']:.4f} acc={metrics['accuracy']:.4f} "
              f"prec={metrics['precision_macro']:.4f} rec={metrics['recall_macro']:.4f} "
              f"auc={metrics['auc']:.4f}")
        return metrics

    def run(self):
        summary = []

        for _, resource in self.resources.items():
            model_name = resource["model_name"]
            dataset_path = resource["dataset_path"]
            input_dim = resource["input_dim"]

            for dataset in self.datasets:
                print(f"\n=== OPTUNA MLP SEARCH | MODEL: {model_name} | DATASET: {dataset} | "
                      f"BATCH={self.batch_size} | patience={self.patience} | min_delta={self.min_delta} ===")

                train_df, val_df, _ = self._load_splits(dataset_path, dataset)

                sampler = optuna.samplers.TPESampler(seed=42)
                study = optuna.create_study(direction="maximize", sampler=sampler)
                study.optimize(
                    lambda t: self._objective(t, input_dim, model_name, dataset, train_df, val_df),
                    n_trials=self.n_trials,
                    show_progress_bar=False,
                )

                best_f1 = float(study.best_value)
                best_params = study.best_params
                # include fixed training constants for transparency
                best_params["lr"] = best_params.get("lr")  # already there
                best_params["optimizer"] = best_params.get("optimizer")
                best_params["weight_decay"] = best_params.get("weight_decay", 0.0)
                best_params["use_weight_decay"] = best_params.get("use_weight_decay", best_params["weight_decay"] > 0)
                best_params["patience"] = self.patience
                best_params["min_delta"] = self.min_delta

                print("\n===== OPTUNA FINISHED =====")
                print(f"[BEST F1] {model_name} | {dataset} → {best_f1:.4f}")
                print(f"Best type: {DynamicMLPFactory.type_signature(best_params)}")

                metrics = self._retrain_and_report(best_params, input_dim, model_name, dataset, train_df, val_df)

                result = {
                    "model_name": model_name,
                    "dataset": dataset,
                    "batch_size": int(self.batch_size),
                    "best_f1_macro": round(best_f1, 6),
                    "best_type": DynamicMLPFactory.type_signature(best_params),
                    "best_params": best_params,
                    "metrics_on_validation": {k: round(v, 6) for k, v in metrics.items()},
                }

                out_json = self.out_root / f"best_{_safe_name(model_name)}_{_safe_name(dataset)}.json"
                with open(out_json, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"[Saved] {out_json}")

                summary.append(result)

        summary_path = self.out_root / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Saved summary] {summary_path}")
