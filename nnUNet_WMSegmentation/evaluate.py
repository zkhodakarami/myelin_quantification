#!/usr/bin/env python3
"""
evaluate.py
===========
Evaluate nnU-Net predictions against ground-truth white-matter masks.

Metrics
-------
* Dice Similarity Coefficient (DSC)
* Intersection over Union (IoU / Jaccard)
* Precision
* Recall (Sensitivity)
* Hausdorff Distance 95th percentile (HD95)

Outputs
-------
* Per-case CSV (``evaluation_results.csv``)
* Summary statistics printed to stdout
* Optional side-by-side visualisations (input | GT | prediction)

Usage
-----
    python evaluate.py \
        --pred_dir ./predictions \
        --gt_dir   ./test_ground_truth \
        --output_csv evaluation_results.csv \
        --vis_dir  visualisations
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import distance_transform_edt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    intersection = np.sum(pred * gt)
    return float((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth))


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return float((intersection + smooth) / (union + smooth))


def precision_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    tp = np.sum(pred * gt)
    return float((tp + smooth) / (np.sum(pred) + smooth))


def recall_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    tp = np.sum(pred * gt)
    return float((tp + smooth) / (np.sum(gt) + smooth))


def hausdorff_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute 95th-percentile Hausdorff distance between two binary masks."""
    if not HAS_SCIPY:
        return float("nan")
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float("inf")

    pred_border = pred.astype(bool) & ~_erode(pred.astype(bool))
    gt_border = gt.astype(bool) & ~_erode(gt.astype(bool))

    dt_pred = distance_transform_edt(~pred_border)
    dt_gt = distance_transform_edt(~gt_border)

    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0:
        return float("inf")

    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Simple binary erosion by 1 pixel (4-connected)."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=1)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_mask(path: str) -> np.ndarray:
    """Load a single-channel mask and binarise to {0, 1}."""
    with Image.open(path) as img:
        arr = np.array(img.convert("L"), dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def find_case_id(filename: str) -> str | None:
    """Extract the case_id (e.g. ``slide_12345``) from a filename."""
    m = re.match(r"(slide_\d+)", filename)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_visualisation(
    case_id: str,
    pred: np.ndarray,
    gt: np.ndarray,
    dsc: float,
    vis_dir: str,
    input_image_dir: str | None = None,
):
    """Save a side-by-side comparison figure."""
    if not HAS_MPL:
        return

    ncols = 3 if input_image_dir else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    col = 0
    if input_image_dir:
        img_path = None
        for ext in (".png", ".tiff", ".tif"):
            candidate = os.path.join(input_image_dir, f"{case_id}_0000{ext}")
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path:
            with Image.open(img_path) as im:
                axes[col].imshow(np.array(im.convert("L")), cmap="gray")
        axes[col].set_title("Input (ch 0)")
        axes[col].axis("off")
        col += 1

    axes[col].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[col].set_title("Ground Truth")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[col].set_title(f"Prediction (DSC={dsc:.3f})")
    axes[col].axis("off")

    plt.suptitle(case_id, fontsize=14)
    plt.tight_layout()
    os.makedirs(vis_dir, exist_ok=True)
    fig.savefig(os.path.join(vis_dir, f"{case_id}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net WM segmentation predictions.")
    parser.add_argument("--pred_dir", required=True, help="Directory with predicted mask PNGs.")
    parser.add_argument("--gt_dir", required=True, help="Directory with ground-truth mask PNGs.")
    parser.add_argument("--output_csv", default="evaluation_results.csv", help="Output CSV path.")
    parser.add_argument("--vis_dir", default=None, help="Directory for visualisation images (optional).")
    parser.add_argument(
        "--input_image_dir",
        default=None,
        help="Optional directory with input channel-0 images for visualisation.",
    )
    args = parser.parse_args()

    pred_dir = os.path.abspath(args.pred_dir)
    gt_dir = os.path.abspath(args.gt_dir)

    gt_files = {
        find_case_id(f): os.path.join(gt_dir, f)
        for f in os.listdir(gt_dir)
        if f.endswith(".png") and find_case_id(f) is not None
    }

    pred_files = {
        find_case_id(f): os.path.join(pred_dir, f)
        for f in os.listdir(pred_dir)
        if f.endswith(".png") and find_case_id(f) is not None
    }

    common_ids = sorted(set(gt_files.keys()) & set(pred_files.keys()))
    if len(common_ids) == 0:
        print("ERROR: No matching prediction/ground-truth pairs found.")
        sys.exit(1)

    print(f"Evaluating {len(common_ids)} cases ...")
    print("-" * 72)
    print(f"{'Case':>20s}  {'DSC':>7s}  {'IoU':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'HD95':>8s}")
    print("-" * 72)

    rows = []
    for cid in common_ids:
        pred = load_mask(pred_files[cid])
        gt = load_mask(gt_files[cid])

        # Resize prediction to GT size if they differ
        if pred.shape != gt.shape:
            pred_img = Image.fromarray(pred * 255, mode="L").resize(
                (gt.shape[1], gt.shape[0]), Image.Resampling.NEAREST
            )
            pred = (np.array(pred_img) > 127).astype(np.uint8)

        dsc = dice_score(pred, gt)
        iou = iou_score(pred, gt)
        prec = precision_score(pred, gt)
        rec = recall_score(pred, gt)
        hd95 = hausdorff_95(pred, gt)

        rows.append({
            "case_id": cid,
            "dice": dsc,
            "iou": iou,
            "precision": prec,
            "recall": rec,
            "hd95": hd95,
        })

        print(f"{cid:>20s}  {dsc:7.4f}  {iou:7.4f}  {prec:7.4f}  {rec:7.4f}  {hd95:8.2f}")

        if args.vis_dir:
            save_visualisation(
                cid,
                pred,
                gt,
                dsc,
                args.vis_dir,
                input_image_dir=args.input_image_dir,
            )

    # Summary
    print("-" * 72)
    metrics = ["dice", "iou", "precision", "recall", "hd95"]
    means = {m: np.mean([r[m] for r in rows if np.isfinite(r[m])]) for m in metrics}
    stds = {m: np.std([r[m] for r in rows if np.isfinite(r[m])]) for m in metrics}

    print(f"{'MEAN':>20s}  {means['dice']:7.4f}  {means['iou']:7.4f}  "
          f"{means['precision']:7.4f}  {means['recall']:7.4f}  {means['hd95']:8.2f}")
    print(f"{'STD':>20s}  {stds['dice']:7.4f}  {stds['iou']:7.4f}  "
          f"{stds['precision']:7.4f}  {stds['recall']:7.4f}  {stds['hd95']:8.2f}")

    # Write CSV
    csv_path = os.path.abspath(args.output_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id"] + metrics)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow({"case_id": "MEAN", **{m: f"{means[m]:.4f}" for m in metrics}})
        writer.writerow({"case_id": "STD", **{m: f"{stds[m]:.4f}" for m in metrics}})

    print(f"\nResults saved to {csv_path}")
    if args.vis_dir:
        print(f"Visualisations saved to {os.path.abspath(args.vis_dir)}")


if __name__ == "__main__":
    main()
