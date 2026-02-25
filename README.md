# CLAM-Based Reference Region Identification

Automated identification of intact (least-damaged) white matter tissue in LFB+CV-stained histology slides using attention-based deep learning (CLAM). This pipeline serves as the reference region selection step for white matter hyperintensity (WMH) optical density analysis.

## Overview

The pipeline processes whole-slide TIFF images through three stages:

1. **CLAM Inference** (`ClamResumeSuperpixels.py`): Tiles the slide into 256x256 patches, extracts ResNet50 features, runs CLAM attention-based classification, aggregates attention into SLIC superpixels.
2. **Intact Tissue Analysis** (`intact_tissue_analyzer.py`): Applies a white matter mask, identifies superpixels with the lowest attention scores (least pathology), and generates binary masks and visualizations.
3. **Batch Orchestrator** (`batch_process_heatmap_tissue_analyser.py`): Runs both stages on a list of slides, handles resume/skip logic, and generates JSON reports.

## Prerequisites

- CUDA-capable GPU (required for CLAM inference)
- Conda environment `clam_latest`

### Environment Setup

```bash
conda activate clam_latest
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
```

Key packages in the environment:
- PyTorch (with CUDA)
- torchvision
- numpy, pandas, Pillow, matplotlib
- scikit-image (for SLIC superpixels)
- tqdm
- tifffile (for reading large TIFFs)

## Pre-trained Model Weights

The `weights/` directory contains two model files required to run the pipeline:

### `resnet50-11ad3fa6.pth` (~98 MB)

Standard ResNet50 weights from torchvision. Used as the feature extraction backbone. The pipeline truncates ResNet50 after `layer3` and adds adaptive average pooling to produce 1024-dimensional feature vectors per tile.

### `s_2_checkpoint.pt` (~3.1 MB)

Fine-tuned CLAM single-branch (CLAM_SB) model checkpoint trained to classify tissue patches as intact vs. damaged in LFB+CV-stained white matter. This model was trained using split 2 of the dataset (see `splits_2.csv` in the original training directory for the train/val/test partition). The attention weights from this model indicate how much each patch contributes to the "damaged" classification -- lower attention = more intact tissue.

## Input Format

### `cases.txt`

A text file with one slide ID per line. Lines starting with `#` are ignored. Two naming formats are supported:

```
# Format 1: hyphen-separated (original)
DI_123456_ABC-01-LFB+CV

# Format 2: underscore-separated
DI_000003R_04S_07_LFBCV
```

The script automatically resolves slide IDs to TIFF paths under the archive:
```
<ARCHIVE_ROOT>/INDD{number}/histo_raw/{slide_id}.tif
```

### White Matter (WM) Masks

For intact tissue analysis, each slide needs a white matter mask file (`WM*.tif`) placed in the slide's CLAM output directory (e.g., `phas_clam_outputs/slide_DI_.../L0_T256_S256/WM*.tif`). These are binary masks where white pixels indicate white matter regions.

## Usage

### Step 1: Run CLAM analysis (get attention scores)

```bash
python batch_process_heatmap_tissue_analyser.py --input cases.txt --clam_only
```

This generates per-slide output directories under `phas_clam_outputs/` containing features, attention maps, superpixel segmentations, and heatmap visualizations.

### Step 2: Add WM masks

Copy the white matter mask (`WM*.tif`) for each slide into its output directory:
```
phas_clam_outputs/slide_DI_123456_.../L0_T256_S256/WM_slide_name.tif
```

### Step 3: Run intact tissue analysis

Using bottom percentile selection (e.g., bottom 2% of attention):
```bash
python batch_process_heatmap_tissue_analyser.py --input cases.txt --intact_only --percentile 2
```

Or select an exact number of lowest-attention superpixels:
```bash
python batch_process_heatmap_tissue_analyser.py --input cases.txt --intact_only --n_tiles 5
```

### Full pipeline (CLAM + intact analysis in one run)

```bash
python batch_process_heatmap_tissue_analyser.py --input cases.txt --percentile 2
```

### Dry run (validate files without processing)

```bash
python batch_process_heatmap_tissue_analyser.py --input cases.txt --dry_run
```

### Custom output directory

```bash
python batch_process_heatmap_tissue_analyser.py --input cases.txt --out_dir phas_clam_outputs/evaluation2/ --clam_only
```

## Output Structure

For each slide, the pipeline creates:

```
phas_clam_outputs/
  slide_DI_123456_.../
    L0_T256_S256/
      # CLAM outputs
      grid_all_tiles_level.csv        # All tile coordinates
      grid_kept_tiles_level.csv       # Tissue-filtered tile coordinates
      features_fp16.pt                # ResNet50 features (fp16)
      attention_raw.npy               # Raw attention weights
      attention_smoothed.npy          # Gaussian-smoothed attention
      tissue_mask.npy                 # Binary tissue mask
      attention_tiles.csv             # Per-tile attention scores
      superpixels_labels_thumb.npy    # SLIC superpixel labels
      superpixels_mask_thumb.npy      # Tissue mask at thumbnail scale
      superpixels_stats.json          # Per-superpixel mean attention
      heatmap_superpixels.png         # Superpixel attention heatmap
      overlay_superpixels.png         # Heatmap overlaid on tissue
      tiling_info.json                # Slide metadata and parameters
      predictions.json                # CLAM classification result

      # Intact analysis outputs (after Step 3)
      intact_analysis/
        mask_intact_tissue.npy        # Binary mask of intact WM
        mask_intact_tissue.png
        mask_white_matter.npy         # WM mask at thumbnail scale
        mask_borders.npy              # Borders of intact regions
        wm_superpixel_stats.csv       # All WM superpixels with attention + is_intact flag
        intact_superpixels.json       # Selected intact superpixel IDs and stats
        analysis_summary.json         # Summary statistics
        overlay_borders_on_tissue.png # Visualization
        masks_analysis.png            # Multi-panel mask visualization
        attention_distribution.png    # Attention histogram
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--percentile` | 10 | Bottom X% attention superpixels considered intact |
| `--n_tiles` | None | Select exactly N lowest-attention superpixels (overrides percentile) |
| `--sp_n` | 2000 | Number of SLIC superpixels |
| `--resume` | off | Skip already-processed slides |
| `--clam_only` | off | Only run CLAM (skip intact analysis) |
| `--intact_only` | off | Only run intact analysis (assumes CLAM done) |

## Scanning Parameters

- Resolution: 0.4 um/pixel (20x magnification)
- Tile size: 256x256 pixels (102.4 um x 102.4 um physical area)
- Stride: 256 (non-overlapping tiles)
- Based on: McKenzie et al. 2022 (PMC9490907)
