import os
import argparse
from typing import List, Tuple

import yaml

from src.metrics.summarizer import (
    MetricsAnalyzer,
    EvaluationResult,
    export_global_best_csv,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export metric matrix CSVs and overall best results from runs."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    default_csv_output_dir = config.get("csv_output_path", ".")
    default_csv_output_dir = default_csv_output_dir.rstrip("/\\") or "."

    f1_metric = config.get("csv_metric", "macro_f1")

    all_results: List[Tuple[str, EvaluationResult]] = []

    if "resources" in config:
        resources = config["resources"]
    else:
        resources = {
            "metric_matrix": {
                "runs_output_path": config["runs_output_path"],
                "csv_output_path": default_csv_output_dir,
                "summary_name": "metric_matrix",
            }
        }

    for name, resource in resources.items():
        runs_output_path = resource.get("runs_output_path") or resource.get("train_output_path")
        csv_output_dir = resource.get("csv_output_path", default_csv_output_dir)
        base_name = resource.get("summary_name", name)

        csv_output_dir = csv_output_dir.rstrip("/\\") or "."
        base_name = os.path.basename(os.path.normpath(base_name))

        analyzer = MetricsAnalyzer(runs_output_path)

        out_paths = analyzer.save_metric_csvs(
            output_dir=csv_output_dir,
            base_name=base_name,
            f1_metric=f1_metric,
        )

        for result in analyzer.results:
            all_results.append((base_name, result))

        for metric, path in out_paths.items():
            print(f"Saved {metric} CSV for {name}: {path}")

    overall_f1_path = os.path.join(
        default_csv_output_dir,
        f"overall_best_by_{f1_metric}.csv",
    )

    overall_accuracy_path = os.path.join(
        default_csv_output_dir,
        "overall_best_by_accuracy.csv",
    )

    export_global_best_csv(
        output_csv_path=overall_f1_path,
        all_results=all_results,
        metric=f1_metric,
    )

    export_global_best_csv(
        output_csv_path=overall_accuracy_path,
        all_results=all_results,
        metric="accuracy",
    )

    print(f"Saved overall best by {f1_metric} CSV: {overall_f1_path}")
    print(f"Saved overall best by accuracy CSV: {overall_accuracy_path}")