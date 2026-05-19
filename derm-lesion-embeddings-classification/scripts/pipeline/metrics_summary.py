from os import path as osp
import argparse
from typing import List, Tuple

import yaml

from src.metrics.metrics_summary import MetricsAnalyzer, EvaluationResult, export_global_best_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metrics analyzer runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    default_csv_output_dir = config.get("csv_output_path")

    all_results: List[Tuple[str, EvaluationResult]] = []

    for name, resource in config["resources"].items():
        train_output_path = resource["train_output_path"]
        csv_output_dir = resource.get("csv_output_path", default_csv_output_dir)

        analyzer = MetricsAnalyzer(train_output_path)

        base_name = resource.get("summary_name", name)
        base_name = osp.basename(osp.normpath(base_name))

        out_paths = analyzer.save_metric_csvs(output_dir=csv_output_dir, base_name=base_name)

        embedding_name = base_name

        for result in analyzer.results:
            all_results.append((embedding_name, result))

        for metric, p in out_paths.items():
            print(f"Saved {metric} CSV for {name}: {p}")

    overall_f1_path = osp.join(default_csv_output_dir, "overall_best_by_f1_avg.csv")
    overall_accuracy_path = osp.join(default_csv_output_dir, "overall_best_by_accuracy.csv")

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

    print(f"Saved overall best by f1_avg CSV: {overall_f1_path}")
    print(f"Saved overall best by accuracy CSV: {overall_accuracy_path}")
