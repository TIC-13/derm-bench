from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.achitectures.dinov3_convnext_large import Dinov3ConvLargeModel
from src.data.data_loaders import DataLoaderFactory


@dataclass(frozen=True)
class Sample:
    img_id: str
    y_true: int
    y_pred: int
    p_pred: float
    idx: int


class WrongClassificationsRunner:
    def run(
            self,
            args: argparse.Namespace
        ) -> None:
        self._set_seed(int(args.seed))
        device = torch.device(str(args.device))

        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        classes = ckpt.get("classes", ["benign", "malignant"])
        if not isinstance(classes, list) or len(classes) != 2:
            raise ValueError(f"Expected 2 classes in checkpoint, got: {classes}")

        test_csv_path = self._resolve_test_csv_path(args.data_root, args.test_csv)
        img_ids = self._load_img_ids(test_csv_path, args.image_column)

        model = self._build_model(args, num_classes=len(classes))
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

        factory = DataLoaderFactory(
            root_dir=args.data_root,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            input_size=(int(args.input_h), int(args.input_w)),
            images_folder=args.images_folder,
            train_csv="train_metadata.csv",
            val_csv="validation_metadata.csv",
            test_csv=args.test_csv,
            label_column=args.label_column,
            image_column=args.image_column,
            label_list=classes,
        )
        _, _, test_loader = factory.get_dataloaders()

        out_root = str(args.out_root)
        out_dirs = {
            "false_malignant": os.path.join(out_root, "false_malignant"),
            "false_benign": os.path.join(out_root, "false_benign"),
        }
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

        max_per_bucket = int(getattr(args, "max_per_bucket", 0) or 0)
        buckets: Dict[str, List[Sample]] = {"false_malignant": [], "false_benign": []}

        sample_global_idx = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(inputs)
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                b = inputs.shape[0]
                for i in range(b):
                    yt = int(targets[i].item())
                    yp = int(preds[i].item())
                    p_pred = float(probs[i, yp].item())

                    img_id = self._img_id_for_index(img_ids, sample_global_idx)

                    if yt == 0 and yp == 1:
                        if max_per_bucket == 0 or len(buckets["false_malignant"]) < max_per_bucket:
                            buckets["false_malignant"].append(
                                Sample(img_id=img_id, y_true=yt, y_pred=yp, p_pred=p_pred, idx=sample_global_idx)
                            )
                    elif yt == 1 and yp == 0:
                        if max_per_bucket == 0 or len(buckets["false_benign"]) < max_per_bucket:
                            buckets["false_benign"].append(
                                Sample(img_id=img_id, y_true=yt, y_pred=yp, p_pred=p_pred, idx=sample_global_idx)
                            )

                    sample_global_idx += 1

                if max_per_bucket > 0 and all(len(v) >= max_per_bucket for v in buckets.values()):
                    break

        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Classes: {classes}")
        print(f"Output root: {out_root}")
        print(f"Saved false malignant: {len(buckets['false_malignant'])}")
        print(f"Saved false benign: {len(buckets['false_benign'])}")

        images_base = Path(args.data_root) / str(args.images_folder)

        for tag, samples in buckets.items():
            self._copy_samples(
                samples=samples,
                out_dir=Path(out_dirs[tag]),
                images_base=images_base,
                classes=classes,
            )

    def _build_model(self, args: argparse.Namespace, num_classes: int) -> nn.Module:
        model = Dinov3ConvLargeModel(
            pretrained=True,
            model_name=args.model_name,
            local_weights=args.local_weights,
            hubconf_folder_path=args.hubconf_folder_path,
            in_features=int(args.in_features),
        )
        model.replace_classifier(num_classes)
        return model

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _resolve_test_csv_path(self, data_root: str, test_csv: str) -> Path:
        p = Path(test_csv)
        if p.is_file():
            return p.resolve()
        return (Path(data_root) / p).resolve()

    def _detect_delimiter(self, header_line: str) -> str:
        c = header_line.count(",")
        s = header_line.count(";")
        return ";" if s > c else ","

    def _load_img_ids(self, csv_path: Path, image_column: str) -> List[str]:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            first = f.readline()
            if not first:
                raise ValueError(f"Empty CSV: {csv_path}")
            delim = self._detect_delimiter(first)
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delim)
            if reader.fieldnames is None or image_column not in reader.fieldnames:
                raise ValueError(f"CSV missing column '{image_column}'. Found: {reader.fieldnames}")
            ids: List[str] = []
            for row in reader:
                v = (row.get(image_column) or "").strip()
                ids.append(v)
        if not ids:
            raise ValueError(f"No rows found in CSV: {csv_path}")
        return ids

    def _img_id_for_index(self, img_ids: List[str], idx: int) -> str:
        if idx < 0 or idx >= len(img_ids):
            raise IndexError(f"Sample idx {idx} out of range for img_ids (len={len(img_ids)})")
        return img_ids[idx]

    def _safe_name(self, s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_\-]", "", s)
        return s or "class"

    def _pick_best_candidate(self, paths: List[Path]) -> Optional[Path]:
        if not paths:
            return None
        pref = {".jpg": 0, ".jpeg": 1, ".png": 2, ".webp": 3}
        paths = sorted(paths, key=lambda p: (pref.get(p.suffix.lower(), 99), len(str(p))))
        return paths[0]

    def _find_image_file(self, images_base: Path, img_id: str) -> Optional[Path]:
        if not img_id:
            return None

        p = images_base / img_id
        if p.is_file():
            return p

        candidates = list(images_base.glob(f"{img_id}.*"))
        best = self._pick_best_candidate(candidates)
        if best is not None:
            return best

        candidates = list(images_base.rglob(f"{img_id}.*"))
        best = self._pick_best_candidate(candidates)
        if best is not None:
            return best

        return None

    def _copy_samples(self, samples: List[Sample], out_dir: Path, images_base: Path, classes: List[str]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        for s in samples:
            src = self._find_image_file(images_base, s.img_id)
            if src is None:
                print(f"[WARN] Could not find file for img_id='{s.img_id}' under '{images_base}'")
                continue

            true_name = self._safe_name(classes[s.y_true])
            pred_name = self._safe_name(classes[s.y_pred])

            dst_name = f"idx{s.idx}_true{true_name}_pred{pred_name}_p{s.p_pred:.3f}{src.suffix.lower()}"
            dst = out_dir / dst_name

            shutil.copy2(src, dst)
