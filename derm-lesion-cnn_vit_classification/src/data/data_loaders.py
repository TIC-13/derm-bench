import os
import csv
from collections import Counter

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.data.dataset import CustomDataset


class DataLoaderFactory:
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        input_size: tuple[int, int] = (224, 224),
        images_folder: str = "images",
        train_csv: str = "train_metadata.csv",
        val_csv: str = "validation_metadata.csv",
        test_csv: str = "test_metadata.csv",
        label_column: str = "benign_malignant",
        image_column: str = "img_id",
        label_list: list[str] | None = None,
        augmentation_params: dict | None = None,
        normalization_mean: list[float] = [0.485, 0.456, 0.406],
        normalization_std: list[float] = [0.229, 0.224, 0.225],
        balance_minority_with_augmentation: bool = True,
    ):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.images_folder = images_folder
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.label_column = label_column
        self.image_column = image_column
        self.label_list = label_list
        self.balance_minority_with_augmentation = balance_minority_with_augmentation

        aug = augmentation_params or {
            "scale": (0.8, 1.0),
            "ratio": (0.75, 1.33),
            "rotation": 15,
            "perspective_distortion": 0.5,
            "color_jitter": (0.8, 1.2),
            "affine_degrees": 10,
            "affine_translate": (0.05, 0.05),
            "affine_scale": (0.95, 1.05),
            "affine_shear": 5,
            "blur_kernel": (3, 3),
            "blur_sigma": (0.1, 1.0),
        }

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.input_size, scale=aug["scale"], ratio=aug["ratio"]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(aug["rotation"]),
                transforms.RandomPerspective(distortion_scale=aug["perspective_distortion"], p=0.5),
                transforms.ColorJitter(brightness=aug["color_jitter"]),
                transforms.RandomAffine(
                    degrees=aug["affine_degrees"],
                    translate=aug["affine_translate"],
                    scale=aug["affine_scale"],
                    shear=aug["affine_shear"],
                ),
                transforms.GaussianBlur(kernel_size=aug["blur_kernel"], sigma=aug["blur_sigma"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalization_mean, std=normalization_std),
            ]
        )

        self.val_test_transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalization_mean, std=normalization_std),
            ]
        )

    def _read_labels_from_csv(self, csv_path: str) -> list:
        labels = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.label_column not in row:
                    continue
                label = row[self.label_column]
                if self.label_list is not None and label not in self.label_list:
                    continue
                labels.append(label)
        return labels

    def _get_balanced_sampler(self, train_csv_path: str, dataset_len: int) -> WeightedRandomSampler | None:
        labels = self._read_labels_from_csv(train_csv_path)
        if not labels or len(labels) != dataset_len:
            return None

        counts = Counter(labels)
        if len(counts) < 2:
            return None

        max_count = max(counts.values())
        num_classes = len(counts)
        num_samples = max_count * num_classes

        sample_weights = [1.0 / counts[l] for l in labels]
        return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

    def get_dataloaders(self):
        """Create train, validation, and test dataloaders.

        Returns:
            Train, validation, and test dataloaders.
        """
        images_dir = os.path.join(self.root_dir, self.images_folder)
        train_csv_path = os.path.join(self.root_dir, self.train_csv)
        val_csv_path = os.path.join(self.root_dir, self.val_csv)
        test_csv_path = os.path.join(self.root_dir, self.test_csv)

        train_dataset = CustomDataset(
            train_csv_path,
            images_dir,
            transform=self.train_transform,
            label_column=self.label_column,
            image_column=self.image_column,
            label_list=self.label_list,
        )

        val_dataset = CustomDataset(
            val_csv_path,
            images_dir,
            transform=self.val_test_transform,
            label_column=self.label_column,
            image_column=self.image_column,
            label_list=self.label_list,
        )

        test_dataset = CustomDataset(
            test_csv_path,
            images_dir,
            transform=self.val_test_transform,
            label_column=self.label_column,
            image_column=self.image_column,
            label_list=self.label_list,
        )

        sampler = None
        if self.balance_minority_with_augmentation:
            sampler = self._get_balanced_sampler(train_csv_path, len(train_dataset))

        if sampler is None:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=self.num_workers,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader
