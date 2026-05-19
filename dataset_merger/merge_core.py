import os
import shutil
from typing import List, Optional

import pandas as pd
from tqdm import tqdm


class DatasetMerger:
    def __init__(
            self,
            root_dir: str,
            output_dir: str = "merged",
            included_datasets: Optional[List[str]] = None
        ) -> None:
        self.root_dir = root_dir
        self.output_dir = os.path.join(root_dir, output_dir)
        self.output_images_dir = os.path.join(self.output_dir, "images")
        self.included_datasets = included_datasets
        self.partitions = ["train_metadata.csv", "validation_metadata.csv", "test_metadata.csv"]
        os.makedirs(self.output_images_dir, exist_ok=True)

    def _get_dataset_dirs(self) -> List[str]:
        if self.included_datasets:
            return [os.path.join(self.root_dir, d)
                    for d in self.included_datasets
                    if os.path.isdir(os.path.join(self.root_dir, d))]
        else:
            return [os.path.join(self.root_dir, d)
                    for d in os.listdir(self.root_dir)
                    if os.path.isdir(os.path.join(self.root_dir, d)) and d != os.path.basename(self.output_dir)]

    def _copy_image(
            self,
            src_path: str,
            dst_filename: str
        ) -> None:
        dst_path = os.path.join(self.output_images_dir, dst_filename)
        shutil.copy(src_path, dst_path)

    def merge(self):
        """Merge selected datasets into a single dataset directory.

        Copies images, resolves duplicated image names, and saves merged
        metadata CSV files for each partition.
        """
        dataframes = {partition: [] for partition in self.partitions}
        image_names_seen = set()

        for dataset_dir in self._get_dataset_dirs():
            dataset_name = os.path.basename(dataset_dir)
            images_dir = os.path.join(dataset_dir, "images")

            for partition in self.partitions:
                csv_path = os.path.join(dataset_dir, partition)
                if not os.path.exists(csv_path):
                    print(f"Warning: {csv_path} not found. Skipping.")
                    continue

                df = pd.read_csv(csv_path)

                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Merging {partition} from {dataset_name}"):
                    img_id = row["img_id"]

                    valid_extensions = [".jpg", ".jpeg", ".png"]
                    if not any(img_id.lower().endswith(ext) for ext in valid_extensions):
                        img_id += ".jpg"

                    src_img_path = os.path.join(images_dir, img_id)

                    if not os.path.exists(src_img_path):
                        print(f"Image not found: {src_img_path}. Skipping.")
                        continue

                    new_img_id = img_id
                    if new_img_id in image_names_seen:
                        name, ext = os.path.splitext(img_id)
                        count = 1
                        while new_img_id in image_names_seen:
                            new_img_id = f"{name}_{count}{ext}"
                            count += 1
                        row["img_id"] = new_img_id

                    self._copy_image(src_img_path, new_img_id)
                    image_names_seen.add(new_img_id)
                    dataframes[partition].append(row)

        for partition, rows in dataframes.items():
            out_csv_path = os.path.join(self.output_dir, partition)
            merged_df = pd.DataFrame(rows)
            merged_df.to_csv(out_csv_path, index=False)
            print(f"Saved merged CSV: {out_csv_path}")

