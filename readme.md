# S2C Training Codebase

This repository packages the S2C experimental training scripts and provides a unified, configuration-driven entry point for different datasets. The goal is to keep the workflow easy to reproduce and release on GitHub.

## Directory Layout

- `configs/`: YAML configs per dataset, for example `clcd.yaml`, `second.yaml`.
- `train.py`: the canonical CLI entry point that accepts a config plus optional overrides.
- `s2c/`: runtime modules that host the config loader and the training engine.
- `datasets/`, `models/`, `utils/`: original dataset/model/utility implementations kept unchanged unless otherwise noted.

## Installation

### 1. Create a Python environment

You can use either Conda or plain `venv`. A typical Conda setup:

```bash
conda create -n s2c python=3.10 -y
conda activate s2c
```

### 2. Install dependencies

Install PyTorch (with the right CUDA version for your machine), then the remaining packages:

```bash
# Example: CUDA 11.8 build (adjust to your environment)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Core Python dependencies
pip install tensorboardX albumentations scikit-image scipy pyyaml
```

If you plan to use the optional DINOv3 PCA visualization tools under `models/dinov3`, also install:

```bash
pip install matplotlib scikit-learn opencv-python pillow tqdm
```

> DINOv3 pretrained weights are **not** included in this repository because of size limits.  
> To enable DINOv3 features (including the PCA visualizer), download the official DINOv3 checkpoints manually and place them under `models/dinov3/pretrained_weights/` (for example, `dinov3_vitl16_pretrain_*.pth`, `dinov3_vit7b16_pretrain_*.pth`). The helper code will automatically pick up matching files in that directory.

## Quick Start

```bash
python train.py --config configs/clcd.yaml
```

Add `--options key=value` to override any field on the fly, e.g.:

```bash
python train.py --config configs/clcd.yaml --options training.lr=0.01 training.epochs=20
```

## Custom Configs

Each config generally contains:

- `experiment`: experiment name and dataset identifier.
- `model`: module path, class name, and optional init args.
- `dataset`: dataset module/class plus separate train/val/test build arguments.
- `dataset.root`: dataset root directory (relative paths are resolved against the repo root by default).
- `training`: batch sizes, learning rate, epochs, GPU / multi-GPU flags, enabled loss terms, etc.
- `paths`: relative output directories for logs, predictions, and checkpoints.

Copy `configs/clcd.yaml`, adjust the fields you need, and launch a new experiment.

### Dataset configuration

For each dataset, the `dataset` section controls how images and labels are loaded:

- `module`: Python module that implements the dataset, e.g. `datasets.CLCD_aug`.
- `class_name`: dataset class name inside that module (default is `RS`).
- `root`: dataset root directory; this **must** point to the folder that contains the `train/`, `val/`, and `test/` subfolders used by the dataset code.
- `train/val/test`: each split has:
  - `split`: passed as the `mode` argument when constructing the dataset.
  - `kwargs`: extra keyword arguments, such as `random_crop`, `crop_nums`, `crop_size`, or `sliding_crop`.

During training, the engine will override the `root` variable in the dataset module using `dataset.root`, so you only need to change paths in the YAML files.

### Training configuration

Key fields under `training`:

- `train_batch_size`, `val_batch_size`, `num_workers`: standard PyTorch `DataLoader` settings.
- `lr`, `epochs`: learning rate and total training epochs.
- `gpu`, `dev_id`: whether to use GPU and which CUDA device index.
- `multi_gpu`: optional list/string of GPU indices (e.g. `"0,1"`); when set, the model runs under `DataParallel` or `BalancedDataParallel` depending on `use_balanced_dp`.
- `loss_terms`: which losses to include in the total loss; any subset of `triplet`, `infoNCE`, `sparse`.
- `tta`: enable test-time augmentation in validation/test.
- `load_path`: optional path to a pretrained checkpoint to load before training.

The `paths` section controls where logs, checkpoints, and prediction visualizations are written. All paths are resolved relative to the repo root.

## Running Experiments

### Basic runs

Train on the CLCD dataset with the default configuration:

```bash
python train.py --config configs/clcd.yaml
```

Train on the SECOND dataset:

```bash
python train.py --config configs/second.yaml
```

### Overriding config options from the CLI

You can override any nested field using `--options key=value` pairs. For example:

```bash
python train.py \
  --config configs/clcd.yaml \
  --options training.lr=0.01 training.epochs=20 training.multi_gpu="0,1"
```

This keeps the YAML files as the single source of truth, while still allowing quick ad‑hoc changes from the command line.

## Notes

- Core model definitions, loss functions, and the training pipeline remain identical to the original implementation.
- Logging uses `tensorboardX` by default; visualize with `tensorboard --logdir runs`.

