import argparse

import yaml

from src.metrics.alucination_plots import AlucinationPlotAnalyzer


def main() -> None:
    """Load configuration and export alucination/error plots."""
    parser = argparse.ArgumentParser(
        description="Export matplotlib plots for error predictions by prompt, model, dataset, and label."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    args = parser.parse_args()

    print(f"Loading config from: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    report_root_path = cfg.get("report_root_path", "./results/reports")
    output_dir = cfg.get("alucinations_output_path", "./results/alucinations")
    prediction_column = cfg.get("prediction_column", "prediction")
    label_column = cfg.get("label_column", "label")
    error_value = cfg.get("error_value", "error")
    ignored_alucination_prompts = cfg.get("ignored_alucination_prompts", [])

    print(f"Report root path: {report_root_path}")
    print(f"Output directory: {output_dir}")
    print(f"Prediction column: {prediction_column}")
    print(f"Label column: {label_column}")
    print(f"Error value: {error_value}")

    analyzer = AlucinationPlotAnalyzer(
        root_path=report_root_path,
        prediction_column=prediction_column,
        label_column=label_column,
        error_value=error_value,
        ignored_prompts=ignored_alucination_prompts,
    )

    analyzer.save_all_plots(output_dir=output_dir)


if __name__ == "__main__":
    main()