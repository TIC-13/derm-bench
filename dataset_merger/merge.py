import argparse

import yaml

from dataset_merger.merger import DatasetMerger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create merged datasets from a YAML configuration file."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    datasets_root_dir = config["datasets_root_dir"]
    merge_groups = config["merge_groups"]

    for group_name, group_config in merge_groups.items():
        output_dir = group_config.get("output_dir", group_name)
        included_datasets = group_config["included_datasets"]

        print(f"[INFO] Creating merged dataset: {group_name}")
        print(f"[INFO] Output directory: {output_dir}")
        print(f"[INFO] Included datasets: {included_datasets}")

        merger = DatasetMerger(
            root_dir=datasets_root_dir,
            output_dir=output_dir,
            included_datasets=included_datasets,
        )

        merger.merge()

        print(f"[INFO] Finished merged dataset: {group_name}\n")