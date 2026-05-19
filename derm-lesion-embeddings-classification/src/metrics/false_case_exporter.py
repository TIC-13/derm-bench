from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class ExportPaths:
    csv_path: str
    images_dir: str
    output_dir: str


class FalseCaseExporter:
    def run(
        self,
        csv_path: str = "./results/predictions.csv",
        images_dir: str = "./data/images",
        output_dir: str = "./results/false_cases",
    ) -> Tuple[int, int, int]:
        return self._export(ExportPaths(csv_path=csv_path, images_dir=images_dir, output_dir=output_dir))

    def _export(self, paths: ExportPaths) -> Tuple[int, int, int]:
        df = pd.read_csv(paths.csv_path)

        required = {"img_id", "true_label", "pred_label"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}. Found: {list(df.columns)}")

        out_fp = os.path.join(paths.output_dir, "false_malignant")
        out_fn = os.path.join(paths.output_dir, "false_benign")
        os.makedirs(out_fp, exist_ok=True)
        os.makedirs(out_fn, exist_ok=True)

        index = self._build_image_index(paths.images_dir)

        copied_fp = 0
        copied_fn = 0
        skipped = 0

        for _, row in df.iterrows():
            img_id = str(row.get("img_id", "")).strip()
            true_label = self._norm_label(row.get("true_label"))
            pred_label = self._norm_label(row.get("pred_label"))

            target_dir = None
            if true_label == "benign" and pred_label == "malignant":
                target_dir = out_fp
            elif true_label == "malignant" and pred_label == "benign":
                target_dir = out_fn
            else:
                continue

            src = self._resolve_image_path(img_id, paths.images_dir, index)
            if not src:
                skipped += 1
                continue

            self._safe_copy(src, target_dir, os.path.basename(img_id))

            if target_dir == out_fp:
                copied_fp += 1
            else:
                copied_fn += 1

        return copied_fp, copied_fn, skipped

    def _norm_label(self, x: object) -> str:
        if x is None:
            return ""
        return str(x).strip().lower()

    def _build_image_index(self, images_dir: str) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for root, _, files in os.walk(images_dir):
            for fn in files:
                key = fn.lower()
                if key not in index:
                    index[key] = os.path.join(root, fn)
        return index

    def _resolve_image_path(self, img_id: str, images_dir: str, index: Dict[str, str]) -> Optional[str]:
        if not img_id:
            return None

        direct = os.path.join(images_dir, img_id)
        if os.path.isfile(direct):
            return direct

        key = os.path.basename(img_id).lower()
        return index.get(key)

    def _safe_copy(self, src: str, dst_dir: str, base_name: str) -> str:
        os.makedirs(dst_dir, exist_ok=True)
        name, ext = os.path.splitext(base_name)
        dst = os.path.join(dst_dir, base_name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            return dst

        k = 2
        while True:
            candidate = os.path.join(dst_dir, f"{name}__{k}{ext}")
            if not os.path.exists(candidate):
                shutil.copy2(src, candidate)
                return candidate
            k += 1


def export_false_cases_from_config(
    config_path: str | Path,
    verbose: bool = True,
) -> dict[str, int]:
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    resources = config["resources"]
    datasets = config["datasets"]

    dataset_root_folder = Path(config["dataset_root_folder"])

    experiment_root = Path(config["csv_output_path"]).parent
    wrong_predictions_root = experiment_root / "wrong_predictions"

    exporter = FalseCaseExporter()

    total_fp = 0
    total_fn = 0
    total_skipped = 0
    total_runs = 0

    for resource_name, resource_cfg in resources.items():
        train_output_path = Path(resource_cfg["train_output_path"])

        if not train_output_path.exists():
            if verbose:
                print(f"[SKIP] Resource output path not found: {train_output_path}")
            continue

        for dataset_name in datasets:
            dataset_run_dir = train_output_path / dataset_name

            if not dataset_run_dir.exists():
                if verbose:
                    print(f"[SKIP] Dataset run dir not found: {dataset_run_dir}")
                continue

            images_dir = dataset_root_folder / dataset_name / "images"

            if not images_dir.exists():
                if verbose:
                    print(f"[SKIP] Images dir not found: {images_dir}")
                continue

            model_dirs = [
                path
                for path in dataset_run_dir.iterdir()
                if path.is_dir() and (path / "predictions.csv").exists()
            ]

            if not model_dirs:
                if verbose:
                    print(f"[SKIP] No predictions.csv found in: {dataset_run_dir}")
                continue

            for model_dir in sorted(model_dirs):
                model_name = model_dir.name
                csv_path = model_dir / "predictions.csv"

                output_dir = (
                    wrong_predictions_root
                    / dataset_name
                    / resource_name
                    / model_name
                )

                fp, fn, skipped = exporter.run(
                    csv_path=str(csv_path),
                    images_dir=str(images_dir),
                    output_dir=str(output_dir),
                )

                total_fp += fp
                total_fn += fn
                total_skipped += skipped
                total_runs += 1

                if verbose:
                    print()
                    print(f"[OK] {resource_name} | {dataset_name} | {model_name}")
                    print(f"CSV: {csv_path}")
                    print(f"Images: {images_dir}")
                    print(f"Output: {output_dir}")
                    print(f"Saved false_malignant (FP): {fp}")
                    print(f"Saved false_benign (FN): {fn}")
                    print(f"Skipped image not found: {skipped}")

    summary = {
        "runs": total_runs,
        "false_malignant": total_fp,
        "false_benign": total_fn,
        "skipped": total_skipped,
    }

    if verbose:
        print()
        print("Done.")
        print(f"Total runs: {summary['runs']}")
        print(f"Total false_malignant (FP): {summary['false_malignant']}")
        print(f"Total false_benign (FN): {summary['false_benign']}")
        print(f"Total skipped: {summary['skipped']}")

    return summary
