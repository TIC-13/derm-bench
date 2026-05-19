import argparse

import yaml

from src.embeddings.dataset_processor import EmbeddingDatasetProcessor, SafeAugmentor
from src.config.seed_setting import set_global_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding extraction runner (train-only augmentations) — streamed H5 writes per image")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_global_seed()

    dataset_root_folder = config["dataset_root_folder"]
    datasets = config["datasets"]
    partitions = config["partitions"]
    
    batch_size = config.get("extraction_batch_size")

    aug_cfg = config.get("augmentation", {})
    augmentor = SafeAugmentor(aug_cfg)

    cuda_empty_cache_every_batch = bool(config.get("cuda_empty_cache_every_batch"))

    for name, resource in config["resources"].items():
        print(f"\n=== Processing resource: {name} ===")
        model_name = resource["model_name"]
        output_root = resource["dataset_path"]

        processor = EmbeddingDatasetProcessor(
            model_name=model_name,
            dataset_root=dataset_root_folder,
            batch_size=batch_size,
            augmentor=augmentor,
            cuda_empty_cache_every_batch=cuda_empty_cache_every_batch
        )

        for dataset in datasets:
            processor.process_dataset(dataset, partitions, output_root)
