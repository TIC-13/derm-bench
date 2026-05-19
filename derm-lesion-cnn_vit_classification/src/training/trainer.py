from typing import List
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

from src.data.visualizer import Visualizer


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device,
        early_stopping_patience: int,
        num_epochs: int,
        model_save_path: str,
        metrics_save_path: str,
        classes: List[str],
        visualizer: Visualizer,
        save_model: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        self.metrics_save_path = metrics_save_path
        self.classes = classes
        self.visualizer = visualizer
        self.save_model = bool(save_model)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, targets in loop:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct / total)

        return running_loss / total, correct / total

    def validate_epoch(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)
            for inputs, targets in loop:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

                loop.set_postfix(loss=running_loss / total, acc=correct / total)

        return running_loss / total, correct / total

    def test(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Testing"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

        test_acc = correct / total if total else 0.0

        self.visualizer.save_confusion_matrix(all_targets, all_preds, classes=self.classes)

        precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

        class_report = classification_report(
            all_targets,
            all_preds,
            target_names=self.classes,
            digits=4,
            zero_division=0,
        )

        metrics_txt = (
            f"Detected classes: {self.classes}\n"
            f"Accuracy:  {test_acc:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall:    {recall:.4f}\n"
            f"F1-score:  {f1:.4f}\n\n"
            f"Classification Report:\n{class_report}"
        )
        with open(self.metrics_save_path, "w") as f:
            f.write(metrics_txt)

    def model_training(self):
        best_val_loss = float("inf")
        best_state_dict = None
        epochs_no_improve = 0

        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start_time

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                epochs_no_improve = 0

                best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

                if self.save_model:
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_val_loss": float(best_val_loss),
                            "model_state_dict": best_state_dict,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "classes": self.classes,
                        },
                        self.model_save_path,
                    )
            else:
                epochs_no_improve += 1

            print(
                f"Epoch {epoch}/{self.num_epochs} - "
                f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
                f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f} - "
                f"Epochs no improve: {epochs_no_improve}/{self.early_stopping_patience} - "
                f"Time: {elapsed:.1f}s"
            )

            if epochs_no_improve >= self.early_stopping_patience:
                print(f"[INFO] Early stopping after {epoch} epochs")
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            
        elif self.save_model and self.model_save_path and torch.path.exists(self.model_save_path):
            ckpt = torch.load(self.model_save_path, map_location=self.device)

            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])

        self.visualizer.plot_losses(self.train_losses, self.val_losses)
        self.test()
