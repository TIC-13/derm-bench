import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class AlucinationRecord:
    prompt: str
    model: str
    dataset: str
    total_predictions: int
    error_count: int
    error_percentage: float


@dataclass
class LabelAlucinationRecord:
    prompt: str
    model: str
    dataset: str
    label: str
    total_predictions: int
    error_count: int
    error_percentage: float


class AlucinationPlotAnalyzer:
    def __init__(
        self,
        root_path: str,
        prediction_column: str = "prediction",
        label_column: str = "label",
        error_value: str = "error",
        ignored_prompts: List[str] | None = None,
    ) -> None:
        self.root_path = root_path
        self.prediction_column = prediction_column
        self.label_column = label_column
        self.error_value = error_value.lower()
        self.ignored_prompts = set(ignored_prompts or [])
        self.records: List[AlucinationRecord] = []
        self.label_records: List[LabelAlucinationRecord] = []

    def collect_all_results(self) -> None:
        """Collect error statistics from all prediction CSV files under the root path."""
        self.records.clear()
        self.label_records.clear()

        if not os.path.isdir(self.root_path):
            raise FileNotFoundError(f"Root path not found: {self.root_path}")

        csv_count = 0

        for dirpath, _, filenames in os.walk(self.root_path):
            rel_dir = os.path.relpath(dirpath, self.root_path)
            first_part = rel_dir.split(os.sep)[0]

            if first_part in self.ignored_prompts:
                continue

            for filename in filenames:
                if not filename.endswith(".csv"):
                    continue

                csv_count += 1
                csv_path = os.path.join(dirpath, filename)
                record, label_records = self._parse_prediction_csv(csv_path)

                if record is not None:
                    self.records.append(record)

                self.label_records.extend(label_records)

        print(f"Found CSV files: {csv_count}")
        print(f"Valid prediction CSV files: {len(self.records)}")
        print(f"Valid label-level records: {len(self.label_records)}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected records into a pandas DataFrame."""
        return pd.DataFrame(
            [
                {
                    "prompt": r.prompt,
                    "model": r.model,
                    "dataset": r.dataset,
                    "total_predictions": r.total_predictions,
                    "error_count": r.error_count,
                    "error_percentage": r.error_percentage,
                }
                for r in self.records
            ]
        )

    def to_label_dataframe(self) -> pd.DataFrame:
        """Convert collected label-level records into a pandas DataFrame."""
        return pd.DataFrame(
            [
                {
                    "prompt": r.prompt,
                    "model": r.model,
                    "dataset": r.dataset,
                    "label": r.label,
                    "total_predictions": r.total_predictions,
                    "error_count": r.error_count,
                    "error_percentage": r.error_percentage,
                }
                for r in self.label_records
            ]
        )

    def save_all_plots(self, output_dir: str) -> None:
        """Generate and save all alucination/error plots grouped by prompt, model, dataset, and label."""
        self.collect_all_results()
        data = self.to_dataframe()
        label_data = self.to_label_dataframe()

        print(f"Collected rows: {len(data)}")
        print(f"Collected label rows: {len(label_data)}")

        if data.empty:
            print("No valid prediction CSVs were found. No plots were generated.")
            return

        by_prompt_dir = os.path.join(output_dir, "by_prompt")
        by_model_dir = os.path.join(output_dir, "by_model")
        by_dataset_dir = os.path.join(output_dir, "by_dataset")
        by_label_dir = os.path.join(output_dir, "by_label")

        os.makedirs(by_prompt_dir, exist_ok=True)
        os.makedirs(by_model_dir, exist_ok=True)
        os.makedirs(by_dataset_dir, exist_ok=True)
        os.makedirs(by_label_dir, exist_ok=True)

        prompt_summary = self._aggregate(data, "prompt")
        model_summary = self._aggregate(data, "model")
        dataset_summary = self._aggregate(data, "dataset")
        label_summary = self._aggregate(label_data, "label")

        self._save_count_bar_plot(
            data=prompt_summary,
            category_col="prompt",
            title="Number of Error Predictions by Prompt",
            xlabel="Error Count",
            ylabel="Prompt",
            output_path=os.path.join(by_prompt_dir, "error_count_by_prompt.png"),
        )

        self._save_percentage_pie_plot(
            data=prompt_summary,
            category_col="prompt",
            title="Distribution of Error Predictions by Prompt",
            output_path=os.path.join(by_prompt_dir, "error_percentage_by_prompt.png"),
        )

        self._save_count_bar_plot(
            data=model_summary,
            category_col="model",
            title="Number of Error Predictions by Model",
            xlabel="Error Count",
            ylabel="Model",
            output_path=os.path.join(by_model_dir, "error_count_by_model.png"),
        )

        self._save_percentage_pie_plot(
            data=model_summary,
            category_col="model",
            title="Distribution of Error Predictions by Model",
            output_path=os.path.join(by_model_dir, "error_percentage_by_model.png"),
        )

        self._save_count_bar_plot(
            data=dataset_summary,
            category_col="dataset",
            title="Number of Error Predictions by Dataset",
            xlabel="Error Count",
            ylabel="Dataset",
            output_path=os.path.join(by_dataset_dir, "error_count_by_dataset.png"),
        )

        self._save_percentage_pie_plot(
            data=dataset_summary,
            category_col="dataset",
            title="Distribution of Error Predictions by Dataset",
            output_path=os.path.join(by_dataset_dir, "error_percentage_by_dataset.png"),
        )

        self._save_count_bar_plot(
            data=label_summary,
            category_col="label",
            title="Number of Error Predictions by Ground-Truth Label",
            xlabel="Error Count",
            ylabel="Ground-Truth Label",
            output_path=os.path.join(by_label_dir, "error_count_by_label.png"),
        )

        self._save_percentage_pie_plot(
            data=label_summary,
            category_col="label",
            title="Distribution of Error Predictions by Ground-Truth Label",
            output_path=os.path.join(by_label_dir, "error_percentage_by_label.png"),
        )

        print(f"Plots saved to: {output_dir}")

    def _parse_prediction_csv(
        self,
        csv_path: str,
    ) -> Tuple[Optional[AlucinationRecord], List[LabelAlucinationRecord]]:
        metadata = self._extract_metadata_from_path(csv_path)

        if metadata is None:
            return None, []

        prompt, model, dataset = metadata

        try:
            data = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path}: could not read CSV ({e})")
            return None, []

        required_columns = [self.prediction_column, self.label_column]
        missing_columns = [c for c in required_columns if c not in data.columns]

        if missing_columns:
            print(
                f"Skipping {csv_path}: missing columns {missing_columns}. "
                f"Available columns: {list(data.columns)}"
            )
            return None, []

        total_predictions = len(data)

        predictions = (
            data[self.prediction_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        labels = (
            data[self.label_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if total_predictions == 0:
            error_count = 0
            error_percentage = 0.0
        else:
            error_count = int((predictions == self.error_value).sum())
            error_percentage = 100.0 * error_count / total_predictions

        record = AlucinationRecord(
            prompt=prompt,
            model=model,
            dataset=dataset,
            total_predictions=total_predictions,
            error_count=error_count,
            error_percentage=error_percentage,
        )

        label_records = self._build_label_records(
            prompt=prompt,
            model=model,
            dataset=dataset,
            labels=labels,
            predictions=predictions,
        )

        return record, label_records

    def _build_label_records(
        self,
        prompt: str,
        model: str,
        dataset: str,
        labels: pd.Series,
        predictions: pd.Series,
    ) -> List[LabelAlucinationRecord]:
        label_records: List[LabelAlucinationRecord] = []

        unique_labels = sorted(labels[labels != ""].unique())

        for label in unique_labels:
            label_mask = labels == label
            label_total = int(label_mask.sum())
            label_error_count = int(((predictions == self.error_value) & label_mask).sum())

            if label_total == 0:
                label_error_percentage = 0.0
            else:
                label_error_percentage = 100.0 * label_error_count / label_total

            label_records.append(
                LabelAlucinationRecord(
                    prompt=prompt,
                    model=model,
                    dataset=dataset,
                    label=label,
                    total_predictions=label_total,
                    error_count=label_error_count,
                    error_percentage=label_error_percentage,
                )
            )

        return label_records

    def _extract_metadata_from_path(self, csv_path: str) -> Optional[Tuple[str, str, str]]:
        rel_path = os.path.relpath(csv_path, self.root_path)
        parts = rel_path.split(os.sep)

        if len(parts) < 4:
            return None

        prompt = parts[0]
        dataset = parts[-2]
        model = "/".join(parts[1:-2])

        if not prompt or not model or not dataset:
            return None

        return prompt, model, dataset

    def _aggregate(self, data: pd.DataFrame, group_col: str) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame(
                columns=[
                    group_col,
                    "total_predictions",
                    "error_count",
                    "error_percentage",
                ]
            )

        grouped = (
            data.groupby(group_col, dropna=False)[["total_predictions", "error_count"]]
            .sum()
            .reset_index()
        )

        grouped["error_percentage"] = grouped.apply(
            lambda row: (
                100.0 * row["error_count"] / row["total_predictions"]
                if row["total_predictions"] > 0
                else 0.0
            ),
            axis=1,
        )

        return grouped

    def _save_count_bar_plot(
        self,
        data: pd.DataFrame,
        category_col: str,
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: str,
    ) -> None:
        data = data.sort_values(by="error_count", ascending=False).reset_index(drop=True)

        figure_height = max(8, 0.45 * len(data))
        plt.figure(figsize=(14, figure_height))

        categories = data[category_col].astype(str).tolist()
        values = data["error_count"].astype(int).tolist()

        bars = plt.barh(categories, values)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().invert_yaxis()

        max_value = max(values) if values else 0
        text_offset = max_value * 0.01 if max_value > 0 else 0.1

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_width() + text_offset,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                va="center",
            )

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _save_percentage_pie_plot(
        self,
        data: pd.DataFrame,
        category_col: str,
        title: str,
        output_path: str,
    ) -> None:
        data = data.sort_values(by="error_percentage", ascending=False).reset_index(drop=True)
        data = data[data["error_count"] > 0].copy()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if data.empty:
            plt.figure(figsize=(10, 8))
            plt.title(title)
            plt.text(
                0.5,
                0.5,
                "No error predictions found",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return

        labels = data[category_col].astype(str).tolist()
        counts = data["error_count"].astype(int).tolist()
        percentages = data["error_percentage"].tolist()

        legend_labels = [
            f"{label} — {count} errors, {percentage:.2f}%"
            for label, count, percentage in zip(labels, counts, percentages)
        ]

        plt.figure(figsize=(12, 10))

        wedges, _, _ = plt.pie(
            counts,
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            startangle=90,
        )

        plt.title(title)
        plt.legend(
            wedges,
            legend_labels,
            title="Groups",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()