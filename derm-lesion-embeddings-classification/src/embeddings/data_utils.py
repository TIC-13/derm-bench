from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.embeddings.h5_io import h5_path_from_stem, read_partition_h5, read_embeddings_matrix


class DataUtils:
    def __init__() -> None:
        pass

    @staticmethod
    def load_and_clean_h5(
        base_path: str,
        dataset: str,
        split: str,
    ) -> pd.DataFrame:
        """Load an H5 partition metadata file and remove invalid rows.

        Args:
            base_path: Root directory containing the datasets.
            dataset: Dataset folder name.
            split: Dataset split name, such as train, validation, or test.

        Returns:
            DataFrame containing valid rows with img_id and benign_malignant.

        Raises:
            FileNotFoundError: If the expected H5 file does not exist.
        """
        file_path = h5_path_from_stem(Path(base_path) / dataset / f"{split}_metadata.h5")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = read_partition_h5(file_path)

        df["img_id"] = df["img_id"].astype(str).str.strip()

        df["benign_malignant"] = (
            df["benign_malignant"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["benign_malignant"] = df["benign_malignant"].replace(
            {
                "": np.nan,
                "nan": np.nan,
                "none": np.nan,
                "null": np.nan,
            }
        )

        df = df.dropna(subset=["img_id", "benign_malignant"])

        df = df[df["benign_malignant"].isin(["benign", "malignant"])]

        return df.reset_index(drop=True)

    @staticmethod
    def load_embeddings_and_labels(
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings and labels from a DataFrame.

        Args:
            df: DataFrame containing metadata, labels, and embedding columns.

        Returns:
            A tuple containing the embedding matrix and label array.
        """
        emb_cols = [c for c in df.columns if c.startswith("embedding_")]
        if emb_cols:
            emb_cols = sorted(emb_cols, key=lambda x: int(x.split("_")[-1]))
            X = df[emb_cols].to_numpy(dtype=np.float32)
        else:
            X = df.iloc[:, 2:].to_numpy(dtype=np.float32)
        y = df["benign_malignant"].values
        return X, y

    @staticmethod
    def load_embeddings_and_labels_from_h5(
        base_path: str,
        dataset: str,
        split: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load embeddings and labels directly from an H5 file.

        Args:
            base_path: Root directory containing the datasets.
            dataset: Dataset folder name.
            split: Dataset split name, such as train, validation, or test.

        Returns:
            A tuple containing the embedding matrix and label array.
        """
        file_path = h5_path_from_stem(Path(base_path) / dataset / f"{split}_metadata.h5")
        return read_embeddings_matrix(file_path)

    @staticmethod
    def encode_labels(
        label_encoder: LabelEncoder,
        *dfs: pd.DataFrame,
    ) -> None:
        """Encode class labels in one or more DataFrames.

        The label encoder is fitted on the first DataFrame and then reused to
        transform the remaining DataFrames.

        Args:
            label_encoder: Scikit-learn label encoder used to encode labels.
            *dfs: DataFrames containing the benign_malignant label column.
        """
        if not dfs:
            return

        dfs[0]["benign_malignant"] = label_encoder.fit_transform(
            dfs[0]["benign_malignant"]
        )

        for df in dfs[1:]:
            df["benign_malignant"] = label_encoder.transform(
                df["benign_malignant"]
            )

