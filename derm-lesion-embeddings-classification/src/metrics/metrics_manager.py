import os
from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc

class MetricsManager:
    @staticmethod
    def save_confusion_matrix(
        y_true,
        y_pred,
        classes: List[str],
        output_path: str,
        file_name: str = "confusion_matrix.png"
    ):
        """Compute and save the confusion matrix for the test set.

        Args:
            y_true: Ground-truth class labels.
            y_pred: Model class labels.
            classes: List with the class names.
            output_path: Path where the image will be saved.
            file_name: Name of the output image file.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, file_name))
        plt.close()

    @staticmethod
    def save_report(
        output_path: str,
        model_name: str,
        dataset_name: str,
        classes: List[str],
        acc: float,
        prec: float,
        rec: float,
        f1: float,
        report: str
    ):
        """Save the classification metrics report to a text file.

        Args:
            output_path: Directory where the report file will be saved.
            model_name: Name of the evaluated model.
            dataset_name: Name of the evaluated dataset.
            classes: List with the detected class names.
            acc: Accuracy score.
            prec: Precision score.
            rec: Recall score.
            f1: F1-score.
            report: Full classification report as a string.
        """
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, f"{model_name}_{dataset_name}.txt")
        with open(file_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n\n")
            f.write(f"Detected classes: {list(classes)}\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1-score:  {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

    @staticmethod
    def save_roc_curve(
        y_true,
        y_score,
        output_path: str,
        title: str = "ROC Curve",
        file_name: str = "roc_curve.png",
        save_auc_txt: bool = True
    ) -> float:
        """Compute and save the ROC curve and optionally the AUC value.

        Args:
            y_true: Ground-truth binary labels.
            y_score: Predicted scores or probabilities for the positive class.
                If a 2D array is provided, the second column is used.
            output_path: Directory where the ROC curve image will be saved.
            title: Title of the ROC curve plot.
            file_name: Name of the output image file.
            save_auc_txt: Whether to save the AUC value in a text file.

        Returns:
            Computed area under the ROC curve.
        """
        os.makedirs(output_path, exist_ok=True)

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if y_score.ndim == 2 and y_score.shape[1] > 1:
            y_score = y_score[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, file_name))
        plt.close()

        if save_auc_txt:
            with open(os.path.join(output_path, "roc_auc.txt"), "w") as f:
                f.write(f"{roc_auc:.6f}\n")

        return float(roc_auc)
