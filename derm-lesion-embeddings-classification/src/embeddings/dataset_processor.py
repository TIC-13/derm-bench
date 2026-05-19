import os
from typing import List

from PIL import Image
from tqdm import tqdm
import torch

from src.embeddings.augmentation import SafeAugmentor
from src.embeddings.extractor import DermEmbeddingExtractor
from src.embeddings.h5_io import h5_path_from_stem, open_embeddings_h5_writer, read_partition_h5


class EmbeddingDatasetProcessor:
    def __init__(
        self,
        model_name: str,
        dataset_root: str,
        batch_size: int,
        augmentor: SafeAugmentor,
        cuda_empty_cache_every_batch: bool = True,
    ):
        self.dataset_root = dataset_root
        self.batch_size = max(1, int(batch_size))
        self.extractor = DermEmbeddingExtractor(model_name)
        self.augmentor = augmentor
        self.cuda_empty_cache_every_batch = bool(cuda_empty_cache_every_batch)

    @staticmethod
    def _is_train_partition(partition_filename: str) -> bool:
        return "train" in partition_filename.lower()

    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def _process_one_image_tf(
        self,
        img: Image.Image,
        image_id: str,
        label: str,
        writer,
    ) -> None:
        emb = self.extractor.extract_embedding(img)
        writer.append_row(image_id, label, emb)

    def _process_batch_and_write(
        self,
        ids: List[str],
        labels: List[str],
        images: List[Image.Image],
        writer,
    ) -> None:
        """Runs a batch through the model and writes each row immediately."""
        if not images:
            return
        embs = self.extractor.extract_batch_embeddings(images)
        writer.append_batch(ids, labels, embs)
        if self.cuda_empty_cache_every_batch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_dataset(self, dataset_name: str, partition_files: List[str], output_root: str):
        dataset_path = os.path.join(self.dataset_root, dataset_name)
        output_folder = os.path.join(output_root, dataset_name)
        self._ensure_dir(output_folder)

        for partition_file in partition_files:
            input_h5 = h5_path_from_stem(os.path.join(dataset_path, partition_file))
            if not input_h5.exists():
                print(f"Metadata not found: {input_h5}")
                continue

            is_train = self._is_train_partition(partition_file)
            df = read_partition_h5(input_h5, expand_embeddings=False)

            print(f"Processing {dataset_name}/{partition_file} (train={is_train})...")

            output_h5 = h5_path_from_stem(os.path.join(output_folder, partition_file))

            batch_images: List[Image.Image] = []
            batch_labels: List[str] = []
            batch_ids: List[str] = []

            per_image_mode = self.extractor.backend == "tf"

            with open_embeddings_h5_writer(output_h5) as writer:
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset_name}-{partition_file}"):
                    label = row["benign_malignant"]
                    image_id = row["img_id"]
                    if "isic" in str(image_id).lower():
                        image_id = f"{image_id}.jpg"

                    image_path = os.path.join(dataset_path, "images", image_id)
                    if not os.path.exists(image_path):
                        print(f"Image not found: {image_path}")
                        continue

                    try:
                        img = Image.open(image_path).convert("RGB")

                        if per_image_mode:
                            self._process_one_image_tf(img, image_id, label, writer)
                        else:
                            batch_images.append(img)
                            batch_labels.append(label)
                            batch_ids.append(image_id)
                            if len(batch_images) == self.batch_size:
                                self._process_batch_and_write(
                                    batch_ids, batch_labels, batch_images, writer
                                )
                                batch_images, batch_labels, batch_ids = [], [], []

                        if is_train and self.augmentor.is_enabled():
                            aug_images = self.augmentor.generate(img)

                            if per_image_mode:
                                for k, aimg in enumerate(aug_images, start=1):
                                    aug_id = f"{image_id}_aug_{k}"
                                    self._process_one_image_tf(aimg, aug_id, label, writer)
                            else:
                                for s in range(0, len(aug_images), self.batch_size):
                                    chunk_imgs = aug_images[s : s + self.batch_size]
                                    chunk_ids = [
                                        f"{image_id}_aug_{k}"
                                        for k in range(s + 1, s + 1 + len(chunk_imgs))
                                    ]
                                    chunk_labels = [label] * len(chunk_imgs)
                                    self._process_batch_and_write(
                                        chunk_ids, chunk_labels, chunk_imgs, writer
                                    )

                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

                if (not per_image_mode) and batch_images:
                    self._process_batch_and_write(batch_ids, batch_labels, batch_images, writer)

            print(f"Saved (streamed): {output_h5}")
