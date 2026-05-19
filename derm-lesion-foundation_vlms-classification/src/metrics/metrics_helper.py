import os

import pandas as pd 

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


class MetricsHelper:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def save_csv_and_metrics(self, predictions: list[dict]) -> None:
        """Save predictions as CSV and compute metrics when possible.

        Args:
            predictions: List of prediction dictionaries.
        """
        self.data = pd.DataFrame(predictions)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        csv_path = os.path.splitext(self.output_path)[0] + ".csv"
        self.data.to_csv(csv_path, index=False)

        if self.data.empty:
            self.save_report(
                "No predictions were produced, so metrics could not be computed.\n"
                "Likely causes:\n"
                "- labels are missing/NaN in the selected split (e.g., test has no ground-truth)\n"
                "- image paths were not found (wrong folder/extension)\n"
                "- CSV columns differ from expected (img_id, benign_malignant)\n"
            )
            return

        missing = [c for c in ["label", "prediction"] if c not in self.data.columns]
        if missing:
            self.save_report(
                f"Missing columns in predictions DataFrame: {missing}\n"
                f"Available columns: {list(self.data.columns)}\n"
            )
            return

        self.evaluate()

    def evaluate(self) -> None:
        """Compute and save evaluation metrics."""
        self.load_data()
        self.detect_classes()
        report_text = self.compute_metrics()
        self.save_report(report_text)

    def load_data(self) -> None:
        """Prepare labels and predictions for evaluation."""
        self.data["label"] = self.data["label"].astype(str).str.lower()
        self.data["prediction"] = self.data["prediction"].astype(str).str.lower()
        self.y_true = self.data["label"]
        self.y_pred = self.data["prediction"]

    def detect_classes(self) -> None:
        """Detect all classes from labels and predictions."""
        unique_labels = set(self.y_true)
        unique_predictions = set(self.y_pred)
        self.labels = sorted(unique_labels.union(unique_predictions))
    
    def save_report(self, text: str) -> None:
        """Save the evaluation report as a text file.

        Args:
            text: Report content to save.
        """
        with open(self.output_path, "w") as f:
            f.write(text)
    
    def compute_metrics(self) -> str:
        """Compute classification metrics.

        Returns:
            Formatted classification report text.
        """
        accuracy = accuracy_score(
            self.y_true,
            self.y_pred,
        )

        precision = precision_score(
            self.y_true,
            self.y_pred,
            labels=self.labels,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            self.y_true,
            self.y_pred,
            labels=self.labels,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            self.y_true,
            self.y_pred,
            labels=self.labels,
            average="macro",
            zero_division=0,
        )

        report = classification_report(
            self.y_true,
            self.y_pred,
            labels=self.labels,
            zero_division=0,
        )
        output_text = (
            f"Detected classes: {self.labels}\n"
            f"Accuracy:  {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall:    {recall:.4f}\n"
            f"F1-score:  {f1:.4f}\n\n"
            f"Classification Report:\n{report}"
        )
        return output_text