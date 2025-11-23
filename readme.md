# S2C Training Codebase

This repository packages the S2C experimental training scripts and provides a unified, configuration‑driven entry point for different datasets. The goal is to keep the workflow easy to reproduce and release on GitHub.

## Paper and Citation

If you use this codebase in your research, please cite the corresponding S2C paper: 

```bibtex
@article{ding2025s2c,
  title={S2C: Learning Noise-Resistant Differences for Unsupervised Change Detection in Multimodal Remote Sensing Images},
  author={Ding, Lei and Zuo, Xibing and Hong, Danfeng and Guo, Haitao and Lu, Jun and Gong, Zhihui and Bruzzone, Lorenzo},
  journal={arXiv preprint arXiv:2502.12604},
  year={2025}
}
```

## Directory Layout

- `configs/`: YAML configs per dataset, for example `clcd.yaml`, `second.yaml`.
- `train.py`: the main CLI entry point that accepts a config plus optional overrides.
- `s2c/`: runtime modules that host the config loader and the training engine.
- `datasets/`, `models/`, `utils/`: dataset/model/utility implementations.

## Installation (Short Version)

1. Create a Python 3.10+ environment (Conda or `venv`).
2. Install PyTorch (matching your CUDA version), then:

```bash
pip install tensorboardX albumentations scikit-image scipy pyyaml
```

For the optional DINOv3 PCA tools under `models/dinov3`, see `models/dinov3/README_RS_PCA.md` for details on extra dependencies and checkpoints.

## Quick Start

Train on the CLCD dataset:

```bash
python train.py --config configs/clcd.yaml
```

Train on the SECOND dataset:

```bash
python train.py --config configs/second.yaml
```

Override any field from the command line with `--options key=value`, for example:

```bash
python train.py --config configs/clcd.yaml --options training.lr=0.01 training.epochs=20
```

## Config Overview

Each config generally contains:

- `experiment`: experiment name and dataset identifier.
- `model`: module path, class name, and optional init args.
- `dataset`: dataset module/class plus separate train/val/test build arguments.
- `dataset.root`: dataset root directory (relative paths are resolved against the repo root by default).
- `training`: batch sizes, learning rate, epochs, GPU / multi‑GPU flags, enabled loss terms, etc.
- `paths`: relative output directories for logs, predictions, and checkpoints.

Copy `configs/clcd.yaml`, adjust what you need, and launch a new experiment.

### Dataset section

- `module`: Python module that implements the dataset, e.g. `datasets.CLCD_aug`.
- `class_name`: dataset class name inside that module (default is `RS`).
- `root`: dataset root directory containing `train/`, `val/`, and `test` subfolders; this is the only place you need to edit data paths.
- `train/val/test`: each split provides:
  - `split`: passed as the `mode` argument when constructing the dataset.
  - `kwargs`: extra keyword arguments such as `random_crop`, `crop_nums`, `crop_size`, or `sliding_crop`.

During training, the engine overwrites the `root` variable in the dataset module using `dataset.root`.

### Training section

Typical fields under `training`:

- `train_batch_size`, `val_batch_size`, `num_workers`.
- `lr`, `epochs`.
- `gpu`, `dev_id`, `multi_gpu`, `use_balanced_dp`.
- `loss_terms`: subset of `triplet`, `infoNCE`, `sparse`.
- `tta`, `load_path`.

All output paths under `paths` are resolved relative to the repo root.

## Notes

- Core model definitions, loss functions, and the training pipeline remain identical to the original implementation.
- Logging uses `tensorboardX` by default; visualize with `tensorboard --logdir runs`.



