import os
import re
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class EvaluationResult:
    model: str
    dataset: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def get_metric(self, metric: str) -> float:
        """Return a metric value by name.

        Args:
            metric: Metric attribute name.

        Returns:
            Metric value.
        """
        if metric == "f1":
            metric = "macro_f1"

        if not hasattr(self, metric):
            raise ValueError(
                f"Unknown metric: {metric}. "
                f"Valid examples: accuracy, macro_precision, macro_recall, macro_f1."
            )

        return getattr(self, metric)


class SklearnTxtMetricsParser:
    _FLOAT = r"([0-9]*\.?[0-9]+)"

    def parse(
            self,
            file_path: str,
            model: str,
            dataset: str
        ) -> Optional[EvaluationResult]:
        """Parse accuracy and macro metrics from a text report.

        Args:
            file_path: Path to the metrics text file.
            model: Model name.
            dataset: Dataset name.

        Returns:
            Parsed evaluation result or None if parsing fails.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return None

        accuracy = self._find_single_float(
            content,
            r"^Accuracy:\s*" + self._FLOAT,
        )

        macro_values = self._find_macro_avg(content)

        if accuracy is None or macro_values is None:
            return None

        macro_precision, macro_recall, macro_f1 = macro_values

        return EvaluationResult(
            model=model,
            dataset=dataset,
            accuracy=accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
        )

    @staticmethod
    def _find_single_float(
            text: str,
            pattern: str
        ) -> Optional[float]:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if not match:
            return None

        try:
            return float(match.group(1))
        except ValueError:
            return None

    @classmethod
    def _find_macro_avg(
            cls,
            text: str
        ) -> Optional[Tuple[float, float, float]]:
        match = re.search(
            r"^\s*macro avg\s+"
            + cls._FLOAT + r"\s+"
            + cls._FLOAT + r"\s+"
            + cls._FLOAT + r"\s+"
            + r"\d+\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if not match:
            return None

        try:
            return (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            )
        except ValueError:
            return None


class MetricsAnalyzer:
    def __init__(
            self,
            runs_root_path: str
        ):
        self.runs_root_path = runs_root_path
        self.results: List[EvaluationResult] = []
        self._parser = SklearnTxtMetricsParser()

    def collect_all_results(self) -> None:
        """Collect all evaluation results from the runs directory."""
        self.results.clear()

        if not os.path.isdir(self.runs_root_path):
            raise FileNotFoundError(f"Runs root path not found: {self.runs_root_path}")

        for model_name in sorted(os.listdir(self.runs_root_path)):
            model_path = os.path.join(self.runs_root_path, model_name)

            if not os.path.isdir(model_path):
                continue

            for dataset_name in sorted(os.listdir(model_path)):
                dataset_path = os.path.join(model_path, dataset_name)

                if not os.path.isdir(dataset_path):
                    continue

                txt_path = os.path.join(dataset_path, "test_metrics.txt")

                if not os.path.isfile(txt_path):
                    continue

                parsed = self._parser.parse(
                    txt_path,
                    model=model_name,
                    dataset=dataset_name,
                )

                if parsed is not None:
                    self.results.append(parsed)

    def export_metric_matrix_csv(
            self,
            output_csv_path: str,
            metric: str
        ) -> None:
        """Export a dataset-by-model metric matrix as CSV.

        Args:
            output_csv_path: Path where the CSV file will be saved.
            metric: Metric name to export.
        """
        datasets = sorted({result.dataset for result in self.results})
        models = sorted({result.model for result in self.results})

        matrix: Dict[Tuple[str, str], float] = {}

        for result in self.results:
            matrix[(result.dataset, result.model)] = result.get_metric(metric)

        out_dir = os.path.dirname(output_csv_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset"] + models)

            for dataset in datasets:
                row = [dataset]

                for model in models:
                    value = matrix.get((dataset, model))
                    row.append("" if value is None else f"{value:.6f}")

                writer.writerow(row)

    def save_metric_csv(
            self,
            output_csv_path: str,
            metric: str = "macro_f1"
        ) -> None:
        """Save a metric matrix CSV file.

        Args:
            output_csv_path: Path where the CSV file will be saved.
            metric: Metric name to export.
        """
        self.collect_all_results()
        self.export_metric_matrix_csv(output_csv_path, metric=metric)

        print(f"Metric matrix CSV saved to {output_csv_path} (metric={metric})")

    def save_metric_csvs(
            self,
            output_dir: str,
            base_name: str,
            f1_metric: str = "macro_f1"
        ) -> Dict[str, str]:
        """Save macro F1-score and accuracy metric matrix CSV files.

        Args:
            output_dir: Directory where the CSV files will be saved.
            base_name: Base name used for generated CSV files.
            f1_metric: F1-score metric name to export.

        Returns:
            Dictionary mapping metric names to CSV paths.
        """
        self.collect_all_results()

        os.makedirs(output_dir, exist_ok=True)

        metrics = [f1_metric, "accuracy"]
        out_paths: Dict[str, str] = {}

        for metric in metrics:
            csv_path = os.path.join(output_dir, f"{base_name}__{metric}.csv")
            self.export_metric_matrix_csv(csv_path, metric=metric)
            out_paths[metric] = csv_path

        return out_paths

    def save_two_metric_csvs(
            self,
            output_dir: str,
            f1_metric: str = "macro_f1"
        ) -> Tuple[str, str]:
        """Save macro F1-score and accuracy metric matrix CSV files.

        Args:
            output_dir: Directory where the CSV files will be saved.
            f1_metric: F1-score metric name to export.

        Returns:
            Paths to the generated macro F1 and accuracy CSV files.
        """
        out_paths = self.save_metric_csvs(
            output_dir=output_dir,
            base_name="metric_matrix",
            f1_metric=f1_metric,
        )

        return out_paths[f1_metric], out_paths["accuracy"]


def export_global_best_csv(
        output_csv_path: str,
        all_results: List[Tuple[str, EvaluationResult]],
        metric: str,
    ) -> None:
    """Export the best resource and model for each dataset.

    Args:
        output_csv_path: Path where the CSV file will be saved.
        all_results: List of resource names and evaluation results.
        metric: Metric used to select the best result.
    """
    datasets = sorted({result.dataset for _, result in all_results})

    out_dir = os.path.dirname(output_csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "dataset",
            "best_model",
            metric,
        ])

        for dataset in datasets:
            candidates = [
                (resource_name, result)
                for resource_name, result in all_results
                if result.dataset == dataset
            ]

            if not candidates:
                continue

            candidates = sorted(
                candidates,
                key=lambda item: (item[0], item[1].model),
            )

            _, best_result = max(
                candidates,
                key=lambda item: item[1].get_metric(metric),
            )

            writer.writerow([
                dataset,
                best_result.model,
                f"{best_result.get_metric(metric):.6f}",
            ])