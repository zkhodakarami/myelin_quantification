# nnU-Net White Matter Segmentation on Histology Images

Automated white-matter (WM) mask generation from low-resolution H&E-stained histology brain slides using [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), the state-of-the-art self-configuring framework for biomedical image segmentation.

## Overview

This project trains a **2-D nnU-Net** to produce binary white-matter masks from RGB histology images. A custom trainer (`nnUNetTrainerHistoAug`) extends nnU-Net's already extensive augmentation pipeline with histology-specific colour perturbations to improve generalisation across staining variability.

### Pipeline

```
Raw TIFF images + masks
        |
        v
  prepare_dataset.py       Convert to nnU-Net format (PNG, 80/20 split)
        |
        v
  run_training.sh          Preprocess -> 5-fold cross-validation training
        |
        v
  run_inference.sh          Predict on held-out test set (ensemble of 5 folds)
        |
        v
  evaluate.py              Compute Dice, IoU, Precision, Recall, HD95
```

## Dataset

| Property | Value |
|---|---|
| Input images | Low-resolution H&E histology slides (RGB TIFF) |
| Masks | Binary white-matter segmentation masks (TIFF) |
| Matched pairs | ~340 |
| Training set | ~272 (80%) |
| Test set | ~68 (20%) |
| Split seed | 42 (reproducible) |

**Naming conventions:**
- Images: `slide_{ID}_lowres.tiff` (in `HistoImages/LowRes_all/`)
- Masks: `WM{ID}.tif` (in `HistoImages/WMMask_all/`)

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU with at least 8 GB VRAM
- NVIDIA drivers and CUDA toolkit

### Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# 2. Install PyTorch with CUDA (adjust for your CUDA version)
#    See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare the dataset

Convert raw TIFF images and masks into the nnU-Net v2 dataset format:

```bash
python prepare_dataset.py \
    --image_dir ../HistoImages/LowRes_all \
    --mask_dir  ../HistoImages/WMMask_all \
    --output_dir ./nnUNet_raw/Dataset001_WMHistoSeg \
    --test_gt_dir ./test_ground_truth \
    --test_ratio 0.2 \
    --seed 42
```

This creates:
```
nnUNet_raw/Dataset001_WMHistoSeg/
    imagesTr/          272 cases x 3 channels (R, G, B as separate PNGs)
    labelsTr/          272 binary label PNGs
    imagesTs/          68 test cases x 3 channels
    dataset.json       nnU-Net dataset descriptor
    split_info.json    Train/test case IDs for reproducibility

test_ground_truth/     68 ground-truth masks for evaluation
```

### Step 2: Train

Run the full training pipeline (installs custom trainer, preprocesses data, trains 5-fold CV):

```bash
bash run_training.sh        # all 5 folds
bash run_training.sh 0      # single fold (for quick testing)
```

Training a single fold on a modern GPU typically takes 12--24 hours (1000 epochs, nnU-Net default). All five folds run sequentially.

### Step 3: Predict and evaluate

```bash
bash run_inference.sh
```

This runs the 5-fold ensemble on the test set and computes evaluation metrics. Results are saved to `evaluation_results.csv` and optional visualisations to `visualisations/`.

### Step 4: Predict on new images

To segment new images, place them in a directory with the nnU-Net channel convention (`{case_id}_0000.png`, `{case_id}_0001.png`, `{case_id}_0002.png` for R, G, B), then:

```bash
bash run_inference.sh /path/to/new/images
```

## Model Architecture

nnU-Net v2 automatically configures the network architecture based on the dataset properties. For 2-D histology data the framework typically selects:

| Property | Value |
|---|---|
| Architecture | U-Net (PlainConvUNet) |
| Spatial dims | 2D |
| Encoder/Decoder | Convolutional blocks with instance normalisation |
| Loss function | Dice + Cross-Entropy |
| Optimiser | SGD with Nesterov momentum (0.99) |
| Learning rate | 0.01 with polynomial decay |
| Training epochs | 1000 |
| Cross-validation | 5-fold |
| Inference | Ensemble of all 5 folds, test-time mirroring |

The exact patch size, number of pooling operations, and feature map sizes are determined automatically by nnU-Net's experiment planner based on image dimensions and available GPU memory.

## Augmentation Strategy

The custom `nnUNetTrainerHistoAug` trainer builds on `nnUNetTrainerDA5` (the most aggressive built-in augmentation variant) and adds histology-specific perturbations:

### Built-in augmentations (from nnU-Net DA5)

| Transform | Parameters |
|---|---|
| Rotation | Full 180 degrees (always, for histology) |
| Scaling | 0.5 -- 1.6 (wider than default 0.7 -- 1.43) |
| Mirroring | Horizontal + vertical |
| 90-degree rotation | Random 0/90/180/270 |
| Axis transposition | Random axis swaps |
| Gaussian noise | p=0.15 |
| Gaussian blur | sigma 0.3 -- 1.5 |
| Median filter | kernel 2 -- 8 |
| Brightness | Additive, per-channel |
| Contrast | Multiplicative, range 0.5 -- 2.0 |
| Gamma | Range 0.7 -- 1.5, with and without inversion |
| Simulate low resolution | Zoom 0.25 -- 1.0 |
| Blank rectangles (cutout) | 1 -- 5 rectangles |
| Brightness gradient | Local additive gradient |
| Local gamma | Spatially varying gamma |
| Sharpening | Strength 0.1 -- 1.0 |

### Additional histology augmentations

| Transform | Parameters | Rationale |
|---|---|---|
| Colour jitter | Additive (-0.1, 0.1) + multiplicative (0.85, 1.15), per-channel | Simulate H&E stain intensity variability |
| Higher rotation probability | p=0.6 (vs 0.4 default) | Histology slides have no canonical orientation |
| Higher scale probability | p=0.3 (vs 0.2 default) | Slides vary in tissue magnification |

## Evaluation Metrics

The `evaluate.py` script computes the following metrics for each test case:

| Metric | Description |
|---|---|
| **Dice (DSC)** | Overlap measure, 2*TP / (2*TP + FP + FN) |
| **IoU (Jaccard)** | Intersection / Union |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **HD95** | 95th percentile Hausdorff distance (boundary accuracy) |

Results are saved per-case and as mean +/- std in `evaluation_results.csv`.

## File Reference

| File | Description |
|---|---|
| `prepare_dataset.py` | Convert TIFF data to nnU-Net v2 format |
| `custom_trainer.py` | `nnUNetTrainerHistoAug` with histology augmentations |
| `install_trainer.py` | Install the custom trainer into nnunetv2 package |
| `run_training.sh` | Full training pipeline (preprocess + 5-fold training) |
| `run_inference.sh` | Predict on test set + run evaluation |
| `evaluate.py` | Compute metrics and generate visualisations |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignore data and results directories |

## Citation

If you use this code, please cite the nnU-Net paper:

```bibtex
@article{isensee2021nnunet,
  title     = {nnU-Net: a self-configuring method for deep learning-based
               biomedical image segmentation},
  author    = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A.
               and Petersen, Jens and Maier-Hein, Klaus H.},
  journal   = {Nature Methods},
  volume    = {18},
  number    = {2},
  pages     = {203--211},
  year      = {2021},
  publisher = {Nature Publishing Group}
}
```

## License

This project is provided for research purposes. The nnU-Net framework is distributed under the [Apache License 2.0](https://github.com/MIC-DKFZ/nnUNet/blob/master/LICENSE).
