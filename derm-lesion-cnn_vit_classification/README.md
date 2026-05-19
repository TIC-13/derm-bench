# derm-lesion-cnn_vit_classification

End-to-end fine-tuning benchmark for **benign vs. malignant** dermatology lesion classification using CNNs, Vision Transformers, and DINOv3-ConvNeXt backbones.

## Overview

This project trains image-level classifiers directly on RGB lesion images. A configuration-driven grid runs every combination of **model × dataset**, fine-tunes pretrained architectures, evaluates on held-out test splits, and aggregates metrics into summary CSVs and comparison plots.

Part of the [derm-bench](../README.md) monorepo. For shared dataset preparation and merged corpora, see the [root README](../README.md).

## Features

- Grid training over multiple architectures and datasets from a single YAML config.
- Strong training augmentation and optional minority-class oversampling via `WeightedRandomSampler`.
- Early stopping on validation loss with AdamW and cosine learning-rate schedule.
- Per-run artifacts: best checkpoint, test metrics, loss curves, confusion matrices.
- Post-hoc aggregation: model×dataset metric matrices and global best-model CSVs.
- Optional misclassification export for error analysis.

## Project structure

```
derm-lesion-cnn_vit_classification/
├── Makefile
├── requirements.txt
├── configuration/
│   └── config.yaml                 # Models, datasets, hyperparameters, paths
├── scripts/
│   ├── model_train.py              # Training entrypoint
│   ├── results_summary.py          # Aggregate test_metrics.txt → CSVs
│   ├── results_plots.py            # Bar charts from summary CSVs
│   └── wrong_classifications.py    # Misclassification analysis
├── src/
│   ├── architectures/              # Model loaders (torchvision + DINOv3)
│   ├── data/                       # Dataset, transforms, DataLoaders
│   ├── training/                   # Trainer and orchestrator
│   └── metrics/                    # Summarizer, plots, visualizer
├── models/                         # Local DINOv3 weights (gitignored)
└── results/                        # Experiment outputs (runs, summary, plots)
```

## Prerequisites

- Python 3.10+
- CUDA GPU recommended for training large models (ViT-L, ConvNeXt-L, DINOv3-L)
- Prepared datasets under `../datasets/` (see [Dataset layout](#dataset-layout))
- DINOv3 checkpoint files in `./models/` when using DINOv3 architectures (see [Installation](#installation))

## Dataset layout

Datasets are read from `dataset_path` in config (default `../datasets`). Each dataset folder must contain:

```
../datasets/<dataset_name>/
├── images/
├── train_metadata.csv
├── validation_metadata.csv
└── test_metadata.csv
```

### CSV columns

| Column | Required | Description |
|--------|----------|-------------|
| `img_id` | Yes | Filename or stem under `images/` |
| `benign_malignant` | Yes | `benign` or `malignant` |

Only rows with resolvable image paths and valid labels are kept.

### Configured datasets (default)

HAM10000, ISIC18, ISIC24, HAM10000_SEGMENTED, ISIC18_SEGMENTED, ISIC24_SEGMENTED, PAD, HC, ddi, sd-198, merged_clinic, merged_dermatoscopic, merged.

## Installation

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### DINOv3 weights

For `dinov3_convnext_tiny` and `dinov3_convnext_large` (and frozen-backbone variants), place pretrained weights in `./models/`:

- `models/dinov3_convnext_tiny.pth`
- `models/dinov3_convnext_large.pth`

Torchvision models download ImageNet weights automatically on first use.

## Configuration

Primary file: [`configuration/config.yaml`](configuration/config.yaml).

| Key | Description |
|-----|-------------|
| `models` | List of architecture names to train |
| `datasets` | Dataset folder names under `dataset_path` |
| `classes` | Label names (default: `benign`, `malignant`) |
| `hyperparams` | Training knobs (batch size, epochs, LR, early stopping, etc.) |
| `dataset_path` | Root directory for all datasets (default: `../datasets`) |
| `runs_output_path` | Per-run output directory |
| `csv_output_path` | Summary CSV output directory |
| `plot_dir` | Generated comparison plots |

### Default hyperparameters

| Parameter | Default |
|-----------|---------|
| `batch_size` | 128 |
| `num_epochs` | 100 |
| `learning_rate` | 1e-5 |
| `weight_decay` | 1e-4 |
| `early_stopping_patience` | 10 |
| `num_workers` | 4 |
| `input_size` | `[224, 224]` |
| `seed` | 42 |
| `t_max` | 100 |
| `eta_min` | 1e-6 |

Override config path: `make CONFIG=./path/to/config.yaml train`.

## Usage

```mermaid
flowchart LR
  config[config.yaml] --> train[make train]
  train --> runs[runs/model/dataset/]
  runs --> summary[make summary]
  summary --> csv[summary/*.csv]
  csv --> plots[make plots]
  plots --> figs[results_plots/]
```

### Makefile targets

| Target | Description |
|--------|-------------|
| `make train` | Run full model×dataset training grid |
| `make summary` | Parse `test_metrics.txt` files into summary CSVs |
| `make plots` | Generate bar charts from summary CSVs |
| `make run` | `train` + `summary` + `plots` |
| `make wrong_classifications` | Export misclassified test images (CLI args) |
| `make clean` | Remove `__pycache__` and `.pyc` files |
| `make help` | Show available commands |

### Examples

```bash
# Full pipeline
make run

# Training only with custom config
make CONFIG=./configuration/config.yaml train

# Different Python interpreter
make PY=python3.11 train
```

### Direct script invocation

```bash
python3 -m scripts.model_train --config ./configuration/config.yaml
python3 -m scripts.results_summary --config ./configuration/config.yaml
python3 -m scripts.results_plots --config ./configuration/config.yaml
```

## Models

Architectures are loaded via `ClassifierLoader` in `src/architectures/torch_models.py`.

| Model name | Family | Notes |
|------------|--------|-------|
| `resnet18`, `resnet152` | ResNet | torchvision, replace `fc` |
| `efficientnet_b0`, `efficientnet_b7` | EfficientNet | Replace classifier head |
| `densenet121`, `densenet161` | DenseNet | Replace classifier |
| `vit_l_16`, `vit_b_32` | Vision Transformer | Replace classification head |
| `convnext_tiny`, `convnext_base`, `convnext_large` | ConvNeXt | Replace `classifier[2]` |
| `dinov3_convnext_tiny` | DINOv3 ConvNeXt | Full fine-tune, local weights |
| `dinov3_convnext_tiny_backbone_freezed` | DINOv3 ConvNeXt | Frozen backbone |
| `dinov3_convnext_large` | DINOv3 ConvNeXt | Full fine-tune |
| `dinov3_convnext_large_backbone_freezed` | DINOv3 ConvNeXt | Frozen backbone |

## Outputs

Per training run (`{runs_output_path}/{model}/{dataset}/`):

| File | Description |
|------|-------------|
| `best_model.pth` | Best checkpoint by validation loss |
| `test_metrics.txt` | Accuracy, weighted P/R/F1, sklearn classification report |
| `loss.png` | Training / validation loss curve |
| `confusion_matrix.png` | Test-set confusion matrix |

After `make summary` (`{csv_output_path}/`):

| File | Description |
|------|-------------|
| `metric_matrix__accuracy.csv` | Models × datasets accuracy matrix |
| `metric_matrix__macro_f1.csv` | Models × datasets macro F1 matrix |
| `overall_best_by_accuracy.csv` | Best model per dataset by accuracy |
| `overall_best_by_macro_f1.csv` | Best model per dataset by macro F1 |

After `make plots`: bar charts under `{plot_dir}/`.

Example experiment folders in the repo: `results/unbalanced/`, `results/balanced_aug/`.

## Training details

- **Optimizer:** AdamW with cosine annealing (`CosineAnnealingLR`).
- **Loss:** `CrossEntropyLoss`.
- **Augmentation (train):** RandomResizedCrop, flips, rotation, perspective, color jitter, affine, Gaussian blur, ImageNet normalization.
- **Val / test:** Resize to `input_size`, normalize.
- **Class balancing:** `WeightedRandomSampler` on the training set (minority oversampling) when enabled in the data loader factory.

## Related components

- [derm-bench root README](../README.md) — Shared dataset contract and merged datasets.
- [dataset_merger](../dataset_merger/) — Build `merged_*` CSV datasets.
- [notebooks](../notebooks/) — Per-source dataset preprocessing.
- [derm-lesion-foundation_vlms-classification](../derm-lesion-foundation_vlms-classification/) — VLM benchmark on the same task.
- [derm-lesion-embeddings-classification](../derm-lesion-embeddings-classification/) — Frozen-embedding + head classifiers.

## License

Licensed under the [Apache License 2.0](../LICENSE).
