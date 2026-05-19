from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, df):
        self.X = df.iloc[:, 2:].to_numpy(dtype=np.float32)
        self.y = df["benign_malignant"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class MLPDatasetLoader:
    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        batch_size=512,
        use_smote=False,
        smote_ratio=1.0,
        seed=42,
    ):
        self.train_dataset = TabularDataset(train_df)
        if use_smote:
            X_aug, y_aug = self._smote_lite_offline(
                self.train_dataset.X,
                self.train_dataset.y,
                ratio=smote_ratio,
                seed=seed,
            )
            self.train_dataset.X = X_aug.astype(np.float32)
            self.train_dataset.y = y_aug.astype(np.int64)

        self.val_dataset = TabularDataset(val_df)
        self.test_dataset = TabularDataset(test_df)
        self.batch_size = batch_size

    def _smote_lite_offline(self, X, y, ratio=1.0, seed=42):
        rng = np.random.default_rng(seed)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("Implementação assume classificação binária (2 classes).")

        min_class = classes[np.argmin(counts)]
        maj_count = int(np.max(counts))
        min_count = int(np.min(counts))

        gap = max(0, maj_count - min_count)
        n_new = int(round(ratio * gap))
        if n_new <= 0:
            return X, y

        Xm = X[y == min_class]
        M = Xm.shape[0]
        if M < 2:
            return X, y

        i = rng.integers(0, M, size=n_new)
        j = rng.integers(0, M, size=n_new)
        j = np.where(j == i, (j + 1) % M, j)

        lam = rng.random(size=(n_new, 1), dtype=np.float32)
        X_new = Xm[i] + lam * (Xm[j] - Xm[i])
        y_new = np.full((n_new,), min_class, dtype=y.dtype)

        X_aug = np.concatenate([X, X_new], axis=0)
        y_aug = np.concatenate([y, y_new], axis=0)
        return X_aug, y_aug

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create and return DataLoaders for training, validation and testing.

        Returns:
        Tuple[DataLoader, DataLoader, DataLoader]
            (train_loader, val_loader, test_loader). Train loader is shuffled;
            validation and test loaders are not. All use self.batch_size.
        """
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
