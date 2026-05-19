import os
import argparse
from typing import List, Tuple

import yaml

from src.metrics.metrics import MetricsAnalyzer, EvaluationResult, export_global_best_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export metric matrix CSVs for each report path.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    report_paths = cfg.get("report_paths", [])
    output_dir = cfg.get("csv_output_path", "./results/summaries")

    all_results: List[Tuple[str, EvaluationResult]] = []

    for report_path in report_paths:
        analyzer = MetricsAnalyzer(report_path)

        embedding_name = os.path.basename(os.path.normpath(report_path))

        analyzer.save_metric_csvs(
            output_dir=output_dir,
            base_name=embedding_name,
        )

        for result in analyzer.results:
            all_results.append((embedding_name, result))

    overall_f1_path = os.path.join(output_dir, "overall_best_by_f1_avg.csv")
    overall_accuracy_path = os.path.join(output_dir, "overall_best_by_accuracy.csv")

    export_global_best_csv(
        output_csv_path=overall_f1_path,
        all_results=all_results,
        metric="f1_avg",
    )

    export_global_best_csv(
        output_csv_path=overall_accuracy_path,
        all_results=all_results,
        metric="accuracy",
    )

    print(f"Overall best by f1_avg CSV saved to {overall_f1_path}")
    print(f"Overall best by accuracy CSV saved to {overall_accuracy_path}")