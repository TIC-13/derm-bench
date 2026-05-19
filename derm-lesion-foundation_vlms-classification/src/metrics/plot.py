import os
import csv

import matplotlib.pyplot as plt


class ModelPerformancePlotter:
    def __init__(
            self,
            csv_path: str,
            metric_name: str = "metric",
            top_k: int = 0
        ) -> None:
        self.csv_path = csv_path
        self.metric_name = metric_name
        self.top_k = int(top_k)

        self.models: list[str] = []
        self.values_by_dataset: dict[str, dict[str, float]] = {}
        self.best_by_dataset: dict[str, tuple[str, float]] = {}

        self._infer_metric_name_if_needed()
        self._parse_csv()

    def _infer_metric_name_if_needed(self) -> None:
        if self.metric_name != "metric":
            return
        
        stem = os.path.splitext(os.path.basename(self.csv_path))[0]
        self.metric_name = stem.split("__")[-1].strip() if "__" in stem else stem

    def _safe_name(self, name: str) -> str:
        name = (name or "").strip()

        return (
            name.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

    def _metric_dir(self, root_dir: str) -> str:
        return os.path.join(root_dir, self._safe_name(self.metric_name))

    def _short_model(self, model: str, max_len: int = 26) -> str:
        model = (model or "").strip()

        return model if len(model) <= max_len else (model[: max_len - 1] + "…")

    def _parse_csv(self) -> None:
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if not header or len(header) < 2:
                raise ValueError(f"Invalid CSV header in: {self.csv_path}")

            header = [h.strip() for h in header]

            is_overall_csv = (
                len(header) >= 4
                and header[0] == "dataset"
                and header[1] == "best_prompt"
                and header[2] == "best_model"
            )

            if is_overall_csv:
                metric_col = header[3]
                self.metric_name = metric_col

                values_by_dataset: dict[str, dict[str, float]] = {}

                for row in reader:
                    if not row or len(row) < 4:
                        continue

                    dataset = (row[0] or "").strip()
                    prompt = (row[1] or "").strip()
                    model = (row[2] or "").strip()
                    value = (row[3] or "").strip()

                    if not dataset or not prompt or not model or not value:
                        continue

                    try:
                        score = float(value)

                    except ValueError:
                        continue

                    label = f"{prompt} / {model}"
                    values_by_dataset[dataset] = {label: score}

                self.values_by_dataset = values_by_dataset
                self.models = sorted({
                    model
                    for row in values_by_dataset.values()
                    for model in row.keys()
                })

                self.best_by_dataset = {
                    ds: max(row.items(), key=lambda kv: kv[1])
                    for ds, row in self.values_by_dataset.items()
                }

                return

            self.models = [h.strip() for h in header[1:] if (h or "").strip()]
            values_by_dataset: dict[str, dict[str, float]] = {}

            for row in reader:
                if not row or len(row) < 2:
                    continue

                dataset = (row[0] or "").strip()
                if not dataset:
                    continue

                row_vals: dict[str, float] = {}
                for model, cell in zip(self.models, row[1:], strict=False):
                    cell = (cell or "").strip()
                    if not cell:
                        continue

                    try:
                        row_vals[model] = float(cell)
                    except ValueError:
                        continue

                if row_vals:
                    values_by_dataset[dataset] = row_vals

        self.values_by_dataset = values_by_dataset
        self.best_by_dataset = {
            ds: max(row.items(), key=lambda kv: kv[1])
            for ds, row in self.values_by_dataset.items()
        }

    def _plot_bar(self, labels: list[str], scores: list[float], title: str, ylabel: str, output_path: str) -> None:
        if not labels or not scores:
            return

        fig_w = max(10, min(24, 0.85 * len(labels)))
        plt.figure(figsize=(fig_w, 5))

        bars = plt.bar(labels, scores)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.ylim(0, 1.0)

        for bar, score in zip(bars, scores):
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.01,
                f"{score:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    def plot_best_per_dataset(self, output_dir: str) -> None:
        """Plot the best model score for each dataset.

        Args:
            output_dir: Directory where the plot will be saved.
        """
        if not self.best_by_dataset:
            print(f"No valid dataset scores found in {self.csv_path}")
            return

        items = sorted(self.best_by_dataset.items(), key=lambda kv: kv[1][1], reverse=True)

        labels = [f"{ds}\n{self._short_model(best_model)}" for ds, (best_model, _) in items]
        scores = [best_score for _, (_, best_score) in items]

        metric_dir = self._metric_dir(output_dir)
        output_path = os.path.join(metric_dir, "best_per_dataset.png")

        self._plot_bar(
            labels=labels,
            scores=scores,
            title=f"Best model per dataset ({self.metric_name})",
            ylabel=self.metric_name,
            output_path=output_path,
        )

    def plot_by_dataset(self, output_dir: str) -> None:
        """Plot model scores separately for each dataset.

        Args:
            output_dir: Directory where the plots will be saved.
        """
        metric_dir = self._metric_dir(output_dir)
        base_dir = os.path.join(metric_dir, "by_dataset")

        for dataset, row in self.values_by_dataset.items():
            pairs = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
            
            if self.top_k > 0:
                pairs = pairs[: self.top_k]

            models = [self._short_model(m) for m, _ in pairs]
            scores = [s for _, s in pairs]

            out_path = os.path.join(base_dir, self._safe_name(dataset), "models.png")
            self._plot_bar(
                labels=models,
                scores=scores,
                title=f"{dataset} ({self.metric_name})",
                ylabel=self.metric_name,
                output_path=out_path,
            )