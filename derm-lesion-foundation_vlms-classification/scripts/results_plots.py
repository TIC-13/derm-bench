import os
import argparse

import yaml
from tqdm import tqdm

from src.metrics.plot import ModelPerformancePlotter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from summary CSVs.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    csv_dir = cfg["csv_output_path"]
    plot_root = cfg.get("plot_root", "./results/plots/")
    top_k = int(cfg.get("top_k", 0))

    csv_files = [fn for fn in os.listdir(csv_dir) if fn.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in: {csv_dir}")

    overall_files = [
        fn for fn in csv_files
        if fn.startswith("overall_best_by_")
    ]

    regular_files = [
        fn for fn in csv_files
        if not fn.startswith("overall_best_by_")
    ]

    for fn in tqdm(sorted(regular_files), desc="Generating regular plots", unit="csv"):
        csv_path = os.path.join(csv_dir, fn)

        base = os.path.splitext(fn)[0]
        summary_name = base.split("__", 1)[0] if "__" in base else "summary"
        out_dir = os.path.join(plot_root, summary_name)

        plotter = ModelPerformancePlotter(csv_path=csv_path, top_k=top_k)
        plotter.plot_best_per_dataset(output_dir=out_dir)
        plotter.plot_by_dataset(output_dir=out_dir)

    overall_out_dir = os.path.join(plot_root, "overall")

    for fn in tqdm(sorted(overall_files), desc="Generating overall plots", unit="csv"):
        csv_path = os.path.join(csv_dir, fn)

        plotter = ModelPerformancePlotter(csv_path=csv_path, top_k=top_k)

        plotter.plot_best_per_dataset(output_dir=overall_out_dir)

    print("Finished generating plots.")