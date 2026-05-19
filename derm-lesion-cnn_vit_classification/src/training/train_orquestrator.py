import os
from typing import List, Dict, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.architectures.torch_models import ClassifierLoader
from src.data.data_loaders import DataLoaderFactory
from src.training.trainer import Trainer
from src.metrics.visualizer import Visualizer


class TrainOrquestrator:
    def __init__(
        self,
        models: List[str],
        datasets: List[str],
        classes: List[str],
        hyperparams: Dict[str, object],
        runs_dir: str,
        datasets_root: str
    ) -> None:
        
        self.models = models
        self.datasets = datasets
        self.classes = classes
        self.hyperparams = hyperparams
        self.runs_dir = runs_dir
        self.datasets_root = datasets_root

        torch.manual_seed(self.hyperparams.get("seed", 42))
        random.seed(self.hyperparams.get("seed", 42))
        np.random.seed(self.hyperparams.get("seed", 42))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} for training")

    def _build_model(self, model_name: str) -> nn.Module:
        loader = ClassifierLoader(model_name, self.classes)
        return loader.get_model().to(self.device)

    def _prepare_dataloaders(self, dataset: str) -> Tuple[torch.utils.data.DataLoader, ...]:
        factory = DataLoaderFactory(
            root_dir=os.path.join(self.datasets_root, dataset),
            batch_size=self.hyperparams["batch_size"],
            num_workers=self.hyperparams["num_workers"],
            input_size=self.hyperparams["input_size"],
            images_folder="images",
            train_csv="train_metadata.csv",
            val_csv="validation_metadata.csv",
            test_csv="test_metadata.csv"
        )
        return factory.get_dataloaders()

    def _configure_training(self, model: nn.Module) -> Tuple[nn.Module, optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.hyperparams["learning_rate"],
            weight_decay=self.hyperparams["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hyperparams.get("t_max", self.hyperparams["num_epochs"]),
            eta_min=self.hyperparams.get("eta_min", 1e-6)
        )
        return criterion, optimizer, scheduler

    def _train_single_model(self, model_name: str, dataset: str) -> None:
        print(f"\n\nTraining {model_name} on {dataset}\n")

        run_dir = os.path.join(self.runs_dir, model_name, dataset)
        os.makedirs(run_dir, exist_ok=True)

        model = self._build_model(model_name)
        train_loader, val_loader, test_loader = self._prepare_dataloaders(dataset)
        criterion, optimizer, scheduler = self._configure_training(model)

        model_save_path = os.path.join(run_dir, "best_model.pth")
        metrics_save_path = os.path.join(run_dir, "test_metrics.txt")
        loss_plot_path = os.path.join(run_dir, "loss.png")
        confusion_matrix_plot_path = os.path.join(run_dir, "confusion_matrix.png")

        print(f"model_save_path = {model_save_path}")
        print(f"metrics_save_path = {metrics_save_path}")
        print(f"loss_plot_path = {loss_plot_path}")
        print(f"confusion_matrix_plot_path = {confusion_matrix_plot_path}\n")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            early_stopping_patience=self.hyperparams["early_stopping_patience"],
            num_epochs=self.hyperparams["num_epochs"],
            model_save_path=model_save_path,
            metrics_save_path=metrics_save_path,
            classes=self.classes,
            visualizer=Visualizer(
                loss_plot_path=loss_plot_path,
                confusion_matrix_plot_path=confusion_matrix_plot_path
            )
        )

        trainer.model_training()

    def run_all(self) -> None:
        for model_name in self.models:
            for dataset in self.datasets:
                self._train_single_model(model_name, dataset)