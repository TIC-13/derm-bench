from typing import Union, Optional, Tuple, List

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)
from torch.utils.tensorboard import SummaryWriter


class MLPTrainer:
    """Trainer class handling training, validation, testing, and metrics."""
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: Optional[Union[str, torch.device]],
        lr: float = 1e-4,
        patience: int = 10,
        smoothing: float = 0.1,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        scheduler_mode: str = "min",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.patience = patience

        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=smoothing
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=scheduler_mode,
            patience=scheduler_patience,
            factor=scheduler_factor,
        )

        self.writer = SummaryWriter()


    def train(self):
        """Train the model until early stopping.

        The method runs training epochs, evaluates the validation loss after
        each epoch, logs losses to TensorBoard, updates the learning rate
        scheduler, and stops when validation loss does not improve for the
        configured patience.
        """
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        epoch = 0

        while True:
            self.model.train()
            total_train_loss = 0.0

            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_loader)
            val_loss = self.validation()

            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
            epoch += 1


    def validation(self) -> float:
        """Compute the average validation loss.

        Returns:
            Average loss over all validation batches.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)


    def test(
            self,
            label_encoder: Optional[LabelEncoder] = None,
        ) -> Tuple[
            float,
            float,
            float,
            float,
            str,
            np.ndarray[np.integer],
            np.ndarray[np.integer],
            np.ndarray[np.floating],
            float
        ]:
        """Evaluate the model on the test set.

        Args:
            label_encoder: Optional list of class names used in the
                classification report.

        Returns:
            A tuple containing accuracy, precision, recall, F1-score,
            classification report, ground-truth labels, predicted labels,
            predicted probabilities, and ROC AUC.
        """
        self.model.eval()
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_targets.extend(y_batch.numpy())
                all_probs.extend(probs)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        auc = roc_auc_score(all_targets, all_probs)

        report = classification_report(
            all_targets,
            all_preds,
            target_names=label_encoder.classes_ if label_encoder is not None else None,
        )
        
        print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

        return acc, prec, rec, f1, report, all_targets, all_preds, all_probs, auc
