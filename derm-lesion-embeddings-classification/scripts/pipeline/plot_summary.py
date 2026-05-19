from os import path as osp
import argparse

import yaml
from tqdm import tqdm

from src.metrics.plot_summary import ModelPerformancePlotter


def first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if osp.isfile(p):
            return p
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summary plot generator (CSV-based)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    csv_dir = config.get("csv_output_path")
    plot_root = config.get("plot_root")
    top_k = int(config.get("top_k", 0))

    items = list(config["resources"].items())
    skipped: list[str] = []

    for name, resource in tqdm(items, desc="Generating plots", unit="resource"):
        base_name = osp.basename(osp.normpath(resource.get("summary_name", name)))
        output_dir = resource.get("output_summary_plot", osp.join(plot_root, name))

        csv_paths = [
            first_existing([
                osp.join(csv_dir, f"{base_name}__f1_avg.csv"),
                osp.join(csv_dir, f"{base_name}_f1_avg.csv"),
            ]),
            first_existing([
                osp.join(csv_dir, f"{base_name}__accuracy.csv"),
                osp.join(csv_dir, f"{base_name}_accuracy.csv"),
            ]),
        ]

        found_any = False

        for csv_path in csv_paths:
            if not csv_path:
                continue

            plotter = ModelPerformancePlotter(csv_path=csv_path, top_k=top_k)
            plotter.plot_best_per_dataset(output_dir=output_dir)
            plotter.plot_by_dataset(output_dir=output_dir)

            found_any = True

        if not found_any:
            skipped.append(name)

    overall_output_dir = osp.join(plot_root, "overall")

    overall_csv_paths = [
        osp.join(csv_dir, "overall_best_by_f1_avg.csv"),
        osp.join(csv_dir, "overall_best_by_accuracy.csv"),
    ]

    found_overall = False

    for csv_path in tqdm(overall_csv_paths, desc="Generating overall plots", unit="csv"):
        if not osp.isfile(csv_path):
            continue

        plotter = ModelPerformancePlotter(csv_path=csv_path, top_k=top_k)

        plotter.plot_best_per_dataset(output_dir=overall_output_dir)

        found_overall = True

    if skipped:
        print(f"Finished generating plots (skipped {len(skipped)}: {', '.join(skipped)})")
    else:
        print("Finished generating regular plots.")

    if found_overall:
        print(f"Finished generating overall plots at: {overall_output_dir}")
    else:
        print("No overall CSV files found.")
