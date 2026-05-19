import os
import argparse

import yaml

from src.metrics.plots import ModelPerformancePlotter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot best-per-dataset and per-dataset bars from metric CSVs."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    csv_root = config["csv_output_path"]
    plot_dir = config["plot_dir"]
    top_k = int(config.get("top_k", 0))

    os.makedirs(plot_dir, exist_ok=True)

    if os.path.isdir(csv_root):
        csv_files = [
            os.path.join(csv_root, filename)
            for filename in sorted(os.listdir(csv_root))
            if filename.lower().endswith(".csv")
        ]
    else:
        csv_files = [csv_root]

    for csv_path in csv_files:
        if "overall" not in csv_path:
            plotter = ModelPerformancePlotter(
                csv_path=csv_path,
                top_k=top_k,
            )

            plotter.plot_best_per_dataset(output_dir=plot_dir)
            plotter.plot_by_dataset(output_dir=plot_dir)