import argparse

from src.metrics.false_case_exporter import export_false_cases_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export false positive and false negative cases from prediction CSVs"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()
    export_false_cases_from_config(args.config)


if __name__ == "__main__":
    main()