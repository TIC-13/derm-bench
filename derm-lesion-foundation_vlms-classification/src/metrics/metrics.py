import re
import os
import csv
from typing import List, Dict, Tuple, Optional


class EvaluationResult:
    def __init__(
            self,
            model: str,
            dataset: str,
            f1_avg: float,
            accuracy: float
        ) -> None:
        self.model = model
        self.dataset = dataset
        self.f1_avg = f1_avg
        self.accuracy = accuracy

    def get_metric(self, metric: str) -> float:
        """Get a metric value by name.

        Args:
            metric: Metric name.

        Returns:
            Metric value as float.
        """
        if not hasattr(self, metric):
            raise ValueError("Unknown metric: {metric}. Valid metrics: f1_avg, accuracy.")
        
        value = getattr(self, metric)

        if value is None:
            raise ValueError(f"Metric '{metric}' is None for model={self.model}, dataset={self.dataset}.")
        
        return float(value)

class MetricsAnalyzer:
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path
        self.results: List[EvaluationResult] = []

    def parse_txt_file(
            self,
            file_path: str,
            model: str,
            dataset: str
        ) -> None:
        """Parse metrics from a text report.

        Args:
            file_path: Path to the report file.
            model: Model name.
            dataset: Dataset name.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        f1_header_match = re.search(
            r"^\s*F1-score:\s*([0-9]+(?:\.[0-9]+)?)",
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        acc_header_match = re.search(
            r"^\s*Accuracy:\s*([0-9]+(?:\.[0-9]+)?)",
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if f1_header_match and acc_header_match:
            self.results.append(
                EvaluationResult(
                    model=model,
                    dataset=dataset,
                    f1_avg=float(f1_header_match.group(1)),
                    accuracy=float(acc_header_match.group(1)),
                )
            )

    def collect_all_results(self, config_name: str) -> None:
        """Collect evaluation results only from one configuration folder."""
        self.results.clear()

        if not os.path.isdir(self.root_path):
            raise FileNotFoundError(f"Root path not found: {self.root_path}")

        config_path = os.path.normpath(self.root_path)
        config_folder = os.path.basename(config_path)

        if config_folder != config_name:
            raise ValueError(
                f"Wrong config folder. Expected '{config_name}', "
                f"but got '{config_folder}' from path: {self.root_path}"
            )

        for dirpath, _, filenames in os.walk(config_path):
            if "binary.txt" not in filenames:
                continue

            txt_path = os.path.join(dirpath, "binary.txt")

            rel_dir = os.path.relpath(dirpath, config_path)
            parts = rel_dir.split(os.sep)

            if len(parts) < 2:
                continue

            dataset = parts[-1]
            model = "/".join(parts[:-1])

            self.parse_txt_file(
                file_path=txt_path,
                model=model,
                dataset=dataset,
            )

    def export_metric_matrix_csv(self, output_csv_path: str, metric: str) -> None:
        """Export a dataset-by-model metric matrix as CSV.

        Args:
            output_csv_path: Path where the CSV file will be saved.
            metric: Metric name to export.
        """
        datasets = sorted({r.dataset for r in self.results})
        models = sorted({r.model for r in self.results})

        matrix: Dict[Tuple[str, str], Optional[float]] = {}

        for r in self.results:
            matrix[(r.dataset, r.model)] = r.get_metric(metric)

        out_dir = os.path.dirname(output_csv_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset"] + models)

            for ds in datasets:
                row = [ds]

                for m in models:
                    v = matrix.get((ds, m), None)
                    row.append("" if v is None else f"{v:.6f}")
                    
                writer.writerow(row)

    def save_metric_csvs(self, output_dir: str, base_name: str) -> Dict[str, str]:
        """Save metric matrix CSV files.

        Args:
            output_dir: Directory where CSV files will be saved.
            base_name: Base name used for generated CSV files.

        Returns:
            Dictionary mapping metric names to CSV paths.
        """
        self.collect_all_results(config_name=base_name)

        os.makedirs(output_dir, exist_ok=True)

        metrics = ["f1_avg", "accuracy"]
        out_paths: Dict[str, str] = {}

        for metric in metrics:
            csv_path = os.path.join(output_dir, f"{base_name}__{metric}.csv")
            self.export_metric_matrix_csv(output_csv_path=csv_path, metric=metric)
            out_paths[metric] = csv_path

        for metric, p in out_paths.items():
            print(f"Metric matrix CSV saved to {p} (metric={metric})")

        return out_paths


def export_global_best_csv(
    output_csv_path: str,
    all_results: List[Tuple[str, EvaluationResult]],
    metric: str,
) -> None:
    """Export the best model and prompt for each dataset.

    Args:
        output_csv_path: Path where the CSV file will be saved.
        all_results: List of prompt names and evaluation results.
        metric: Metric used to select the best result.
    """
    datasets = sorted({result.dataset for _, result in all_results})

    out_dir = os.path.dirname(output_csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "dataset",
            "best_prompt",
            "best_model",
            metric,
        ])

        for dataset in datasets:
            candidates = [
                (embedding_name, result)
                for embedding_name, result in all_results
                if result.dataset == dataset
            ]

            if not candidates:
                continue

            candidates = sorted(
                candidates,
                key=lambda item: (item[0], item[1].model),
            )

            best_embedding, best_result = max(
                candidates,
                key=lambda item: getattr(item[1], metric),
            )

            writer.writerow([
                dataset,
                best_embedding,
                best_result.model,
                f"{getattr(best_result, metric):.6f}",
            ])