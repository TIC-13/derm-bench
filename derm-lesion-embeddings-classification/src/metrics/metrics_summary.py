import os
import re
import csv
from os import path as osp
from typing import Dict, List, Tuple, Optional


class EvaluationResult:
    def __init__(self, model: str, dataset: str, f1_avg: float, accuracy: float):
        self.model = model
        self.dataset = dataset
        self.f1_avg = float(f1_avg)
        self.accuracy = float(accuracy)


class MetricsAnalyzer:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.results: List[EvaluationResult] = []

    def parse_txt_file(self, file_path: str) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return

        f1_avg_match = re.search(
            r"^\s*F1-score:\s*([0-9]*\.?[0-9]+)",
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        acc_match = re.search(
            r"^\s*Accuracy:\s*([0-9]*\.?[0-9]+)",
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if not (f1_avg_match and acc_match):
            return

        model = osp.basename(osp.dirname(file_path))
        dataset = osp.basename(osp.dirname(osp.dirname(file_path)))

        if not model or not dataset:
            return

        self.results.append(
            EvaluationResult(
                model=model,
                dataset=dataset,
                f1_avg=float(f1_avg_match.group(1)),
                accuracy=float(acc_match.group(1)),
            )
        )

    def collect_all_results(self) -> None:
        self.results.clear()

        if not osp.isdir(self.root_path):
            raise FileNotFoundError(f"Root path not found: {self.root_path}")

        for root, _, files in os.walk(self.root_path):
            for file in files:
                if file.endswith(".txt") and "_test_" in file:
                    self.parse_txt_file(osp.join(root, file))

    def export_metric_matrix_csv(self, output_csv_path: str, metric: str) -> None:
        datasets = sorted({r.dataset for r in self.results})
        models = sorted({r.model for r in self.results})

        matrix: Dict[Tuple[str, str], Optional[float]] = {}
        for r in self.results:
            matrix[(r.dataset, r.model)] = getattr(r, metric, None)

        out_dir = osp.dirname(output_csv_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset"] + models)

            for ds in datasets:
                row = [ds]
                for m in models:
                    v = matrix.get((ds, m))
                    row.append("" if v is None else f"{v:.6f}")
                writer.writerow(row)

    def save_metric_csvs(self, output_dir: str, base_name: str) -> Dict[str, str]:
        self.collect_all_results()

        os.makedirs(output_dir, exist_ok=True)

        metrics = ["f1_avg", "accuracy"]
        out_paths: Dict[str, str] = {}

        for metric in metrics:
            csv_path = osp.join(output_dir, f"{base_name}__{metric}.csv")
            self.export_metric_matrix_csv(csv_path, metric)
            out_paths[metric] = csv_path

        return out_paths


def export_global_best_csv(
    output_csv_path: str,
    all_results: List[Tuple[str, EvaluationResult]],
    metric: str,
) -> None:
    datasets = sorted({result.dataset for _, result in all_results})

    out_dir = osp.dirname(output_csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "dataset",
            "best_embedding",
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
