import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional
from torch import Tensor


class CustomDataset(Dataset):
    def __init__(
            self,
            csv_path: str,
            images_dir: str,
            transform: Optional[Callable[[Image.Image], Tensor]] = None,
            image_column: str = "img_id",
            label_column: str = "benign_malignant",
            label_list: list[str] = ["benign", "malignant"],
            verbose: bool = True
        ):

        self.image_column = image_column
        self.label_column = label_column
        self.images_dir = images_dir
        self.transform = transform

        data = pd.read_csv(csv_path)
        data = data.dropna(subset=[label_column, image_column])
        self.data = self.filter_existing_images(data)

        self.label_list = label_list

        if verbose:
            class_counts = self.data[self.label_column].str.lower().value_counts()
            print(f"[INFO] Loaded {len(self.data)} samples from '{csv_path}'")
            print("[INFO] Class distribution:")
            for cls, count in class_counts.items():
                pct = count / len(self.data) * 100
                print(f"    {cls}: {count} samples ({pct:.1f}%)")
            print()

    def filter_existing_images(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows with existing image files.

        Args:
            dataframe: Metadata dataframe to filter.

        Returns:
            Filtered dataframe with valid image paths.
        """
        valid_rows = []
        for _, row in dataframe.iterrows():
            img_name = row[self.image_column]
            base_path = os.path.join(self.images_dir, img_name)

            if os.path.exists(base_path):
                valid_rows.append(row)
            else:
                for ext in ['.jpg', '.jpeg', '.png']:
                    if os.path.exists(base_path + ext):
                        row_copy = row.copy()
                        row_copy[self.image_column] = img_name + ext
                        valid_rows.append(row_copy)
                        break

        return pd.DataFrame(valid_rows)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_name = self.data.iloc[idx][self.image_column]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_str = self.data.iloc[idx][self.label_column].lower()
        label = self.label_list.index(label_str)

        return image, label
