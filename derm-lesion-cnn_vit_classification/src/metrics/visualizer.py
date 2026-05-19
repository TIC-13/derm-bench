import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Visualizer:
    def __init__(
            self,
            loss_plot_path: str,
            confusion_matrix_plot_path: str
        ) -> None:
        
        self.loss_plot_path = loss_plot_path
        self.confusion_matrix_plot_path = confusion_matrix_plot_path

    def plot_losses(
            self,
            train_losses,
            val_losses
        ) -> None:
        """Plot and save training and validation loss curves.

        Args:
            train_losses: Training loss values by epoch.
            val_losses: Validation loss values by epoch.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.loss_plot_path)
        plt.close()
        print(f"[INFO] Loss plot saved to {self.loss_plot_path}")

    def save_confusion_matrix(
            self,
            all_targets,
            all_preds,
            classes=None
        ) -> None:
        """Plot and save the confusion matrix.

        Args:
            all_targets: Ground-truth labels.
            all_preds: Predicted labels.
            classes: Class names used as axis labels.
        """
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(self.confusion_matrix_plot_path)
        plt.close()
        print(f"[INFO] Confusion matrix saved to {self.confusion_matrix_plot_path}")
