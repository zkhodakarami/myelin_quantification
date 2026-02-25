#!/usr/bin/env python3
"""
prepare_dataset.py
==================
Convert raw TIFF histology images and white-matter masks into nnU-Net v2
dataset format (2-D, PNG).

Workflow
--------
1. Scan LowRes_all/ and WMMask_all/ directories.
2. Match images to masks by numeric slide ID.
3. Split matched pairs into train (80 %) / test (20 %).
4. Convert each image to three single-channel PNGs (R, G, B) and each mask
   to a single-channel label PNG (0 = background, 1 = white matter).
5. Write ``dataset.json`` required by nnU-Net v2.

Usage
-----
    python prepare_dataset.py \
        --image_dir ../HistoImages/LowRes_all \
        --mask_dir  ../HistoImages/WMMask_all \
        --output_dir ./nnUNet_raw/Dataset001_WMHistoSeg \
        --test_gt_dir ./test_ground_truth \
        --test_ratio 0.2 \
        --seed 42
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_RE = re.compile(r"slide_(\d+)_lowres\.tiff?$", re.IGNORECASE)
MASK_RE = re.compile(r"[Ww][Mm](\d+)\.tiff?$", re.IGNORECASE)


def extract_image_id(filename: str) -> str | None:
    m = IMAGE_RE.search(filename)
    return m.group(1) if m else None


def extract_mask_id(filename: str) -> str | None:
    m = MASK_RE.search(filename)
    return m.group(1) if m else None


def match_images_masks(image_dir: str, mask_dir: str):
    """Return list of (slide_id, image_path, mask_path) tuples."""
    images = {}
    for f in os.listdir(image_dir):
        sid = extract_image_id(f)
        if sid is not None:
            images[sid] = os.path.join(image_dir, f)

    masks = {}
    for f in os.listdir(mask_dir):
        sid = extract_mask_id(f)
        if sid is not None:
            masks[sid] = os.path.join(mask_dir, f)

    matched = []
    for sid in sorted(images.keys()):
        if sid in masks:
            matched.append((sid, images[sid], masks[sid]))

    return matched


def convert_and_save(
    pairs: list,
    images_out: str,
    labels_out: str,
    gt_out: str | None = None,
    is_test: bool = False,
):
    """Convert matched (sid, img_path, mask_path) pairs to nnU-Net PNGs.

    Parameters
    ----------
    pairs : list of (sid, image_path, mask_path)
    images_out : directory to write channel PNGs
    labels_out : directory to write label PNGs (training) or None
    gt_out : directory to write ground-truth masks for evaluation (test)
    is_test : if True, skip writing into labelsTr and write GT separately
    """
    os.makedirs(images_out, exist_ok=True)
    if labels_out and not is_test:
        os.makedirs(labels_out, exist_ok=True)
    if gt_out and is_test:
        os.makedirs(gt_out, exist_ok=True)

    for sid, img_path, mask_path in pairs:
        case_id = f"slide_{sid}"

        # --- image (RGB -> 3 single-channel PNGs) ---
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            arr = np.array(img_rgb, dtype=np.uint8)  # (H, W, 3)

        for ch_idx in range(3):
            ch_img = Image.fromarray(arr[:, :, ch_idx], mode="L")
            ch_img.save(os.path.join(images_out, f"{case_id}_{ch_idx:04d}.png"))

        # --- mask (resize to match image, binarise -> label PNG) ---
        img_h, img_w = arr.shape[:2]  # height, width from the image array
        with Image.open(mask_path) as msk:
            msk_l = msk.convert("L")
            # Resize mask to match image dimensions (NEAREST to preserve labels)
            msk_resized = msk_l.resize((img_w, img_h), Image.Resampling.NEAREST)
            msk_arr = np.array(msk_resized, dtype=np.uint8)

        label_arr = (msk_arr > 0).astype(np.uint8)

        if is_test and gt_out:
            label_img = Image.fromarray(label_arr, mode="L")
            label_img.save(os.path.join(gt_out, f"{case_id}.png"))
        elif labels_out:
            label_img = Image.fromarray(label_arr, mode="L")
            label_img.save(os.path.join(labels_out, f"{case_id}.png"))


def write_dataset_json(output_dir: str, num_training: int):
    """Write the dataset.json required by nnU-Net v2."""
    ds = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B",
        },
        "labels": {
            "background": 0,
            "white_matter": 1,
        },
        "numTraining": num_training,
        "file_ending": ".png",
    }
    out_path = os.path.join(output_dir, "dataset.json")
    with open(out_path, "w") as f:
        json.dump(ds, f, indent=4)
    print(f"  dataset.json written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert histology TIFF data to nnU-Net v2 dataset format."
    )
    parser.add_argument(
        "--image_dir",
        default=os.path.join(os.path.dirname(__file__), "LowRes_all"),
        help="Path to LowRes_all directory.",
    )
    parser.add_argument(
        "--mask_dir",
        default=os.path.join(os.path.dirname(__file__), "WMMask_all"),
        help="Path to WMMask_all directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "nnUNet_raw", "Dataset001_WMHistoSeg"),
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--test_gt_dir",
        default=os.path.join(os.path.dirname(__file__), "test_ground_truth"),
        help="Directory to save ground-truth masks for the held-out test set.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of data for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Resolve paths
    image_dir = os.path.abspath(args.image_dir)
    mask_dir = os.path.abspath(args.mask_dir)
    output_dir = os.path.abspath(args.output_dir)
    test_gt_dir = os.path.abspath(args.test_gt_dir)

    print("=" * 60)
    print("nnU-Net v2 Dataset Preparation")
    print("=" * 60)
    print(f"  Image dir : {image_dir}")
    print(f"  Mask dir  : {mask_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Test GT   : {test_gt_dir}")
    print(f"  Test ratio: {args.test_ratio}")
    print(f"  Seed      : {args.seed}")
    print()

    # 1. Match images to masks
    matched = match_images_masks(image_dir, mask_dir)
    print(f"Found {len(matched)} matched image-mask pairs.")
    if len(matched) == 0:
        print("ERROR: No matched pairs found. Check directory paths and naming.")
        sys.exit(1)

    # 2. Train / test split
    train_pairs, test_pairs = train_test_split(
        matched, test_size=args.test_ratio, random_state=args.seed
    )
    print(f"  Training : {len(train_pairs)}")
    print(f"  Test     : {len(test_pairs)}")
    print()

    # 3. Convert training data
    print("Converting training data ...")
    convert_and_save(
        train_pairs,
        images_out=os.path.join(output_dir, "imagesTr"),
        labels_out=os.path.join(output_dir, "labelsTr"),
    )
    print(f"  Wrote {len(train_pairs)} training cases.")

    # 4. Convert test data
    print("Converting test data ...")
    convert_and_save(
        test_pairs,
        images_out=os.path.join(output_dir, "imagesTs"),
        labels_out=None,
        gt_out=test_gt_dir,
        is_test=True,
    )
    print(f"  Wrote {len(test_pairs)} test cases.")

    # 5. Write dataset.json
    print("Writing dataset.json ...")
    write_dataset_json(output_dir, num_training=len(train_pairs))

    # 6. Save split info for reproducibility
    split_info = {
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "train_ids": [sid for sid, _, _ in train_pairs],
        "test_ids": [sid for sid, _, _ in test_pairs],
    }
    split_path = os.path.join(output_dir, "split_info.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Split info saved to {split_path}")

    print()
    print("Done! Dataset is ready for nnU-Net v2.")
    print(f"  Dataset root: {output_dir}")
    print("  Next step  : set environment variables and run nnUNetv2_plan_and_preprocess")


if __name__ == "__main__":
    main()
