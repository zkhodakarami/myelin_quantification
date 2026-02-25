#!/usr/bin/env python3
"""
Analyze superpixel heatmaps to find most intact white matter tissue
- Apply white matter mask (exclude EVERYTHING except WM - both GM and background)
- Identify X% least attention superpixels within WM only (most intact tissue)
- Save binary masks with borders
- Overlay on downsampled original TIFF
- Save CSV with attention information for each WM superpixel alongside other outputs
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

# -------------------------
# Loading utilities
# -------------------------
def load_superpixel_results(run_dir):
    """Load all superpixel analysis results stored in run_dir"""
    seg = np.load(os.path.join(run_dir, "superpixels_labels_thumb.npy"))
    mask_tissue = np.load(os.path.join(run_dir, "superpixels_mask_thumb.npy"))
    with open(os.path.join(run_dir, "superpixels_stats.json"), "r") as f:
        sp_stats = json.load(f)
    heat_sp = plt.imread(os.path.join(run_dir, "heatmap_superpixels.png"))
    if heat_sp.ndim == 3:
        heat_sp = heat_sp[:, :, 0]
    with open(os.path.join(run_dir, "superpixels_params.json"), "r") as f:
        params = json.load(f)
    return seg, mask_tissue, sp_stats, heat_sp, params

# -------------------------
# WM mask handling
# -------------------------
def apply_wm_mask(wm_mask_path, target_shape, run_dir):
    """
    Load and resize white matter mask to match superpixel thumbnail size.
    Returns boolean mask (True = WM only).
    """
    wm_mask = np.array(Image.open(wm_mask_path).convert('L'))
    with open(os.path.join(run_dir, "tiling_info.json"), "r") as f:
        tiling_info = json.load(f)
    orig_w, orig_h = tiling_info.get("level_dimensions_wh", (wm_mask.shape[1], wm_mask.shape[0]))

    print(f"[WM mask] Original mask shape: {wm_mask.shape}")
    print(f"[WM mask] Target thumb shape: {target_shape}")
    print(f"[WM mask] Original slide dims (from tiling_info): {orig_w}x{orig_h}")

    th, tw = target_shape
    wm_mask_resized = Image.fromarray(wm_mask).resize((tw, th), Image.NEAREST)
    wm_mask_resized = np.array(wm_mask_resized)
    mask_wm = wm_mask_resized > 127

    print(f"[WM mask] WM pixels: {np.sum(mask_wm)} ({100*np.sum(mask_wm)/mask_wm.size:.1f}%)")
    return mask_wm

# -------------------------
# Core analysis
# -------------------------
def identify_intact_superpixels(
        seg,
        mask_wm,
        sp_stats,
        percentile=None,
        n_tiles=None,
        save_csv_path=None
    ):
    """
    Identify superpixels in white matter with lowest attention (most intact).
    Only superpixels with >50% area inside WM mask are considered.

    Args:
        seg: [H, W] segmentation (integer superpixel ids)
        mask_wm: [H, W] boolean WM mask (True = WM only)
        sp_stats: list/dict of superpixel statistics (must contain 'superpixel_id' and 'mean_attention')
        percentile: bottom X% to select (default 10 if both None)
        n_tiles: exact number of tiles to select (overrides percentile)
        save_csv_path: if provided, write a CSV of all WM superpixels with attention + is_intact flag

    Returns:
        intact_sp_ids: list of superpixel ids selected as intact
        wm_sp_stats: list of stats for all WM superpixels (sorted ascending by mean_attention)
    """
    # collect WM superpixel ids (majority in WM)
    wm_sp_ids = set()
    for sp_id in np.unique(seg):
        if sp_id == 0:
            continue
        sp_mask = (seg == sp_id)
        total = np.sum(sp_mask)
        if total == 0:
            continue
        overlap = np.sum(sp_mask & mask_wm)
        if overlap / total > 0.5:
            wm_sp_ids.add(int(sp_id))

    print(f"[Analysis] Total superpixels in seg: {len(np.unique(seg))}")
    print(f"[Analysis] Superpixels considered WM (>50% overlap): {len(wm_sp_ids)}")

    # filter sp_stats to WM only
    wm_sp_stats = [s for s in sp_stats if int(s.get("superpixel_id")) in wm_sp_ids]
    if len(wm_sp_stats) == 0:
        print("[WARNING] No WM superpixel stats found - empty result.")
        return [], []

    # ensure numeric attention exists
    for s in wm_sp_stats:
        if "mean_attention" not in s:
            s["mean_attention"] = float(s.get("attention", 0.0))

    wm_sp_stats.sort(key=lambda x: float(x["mean_attention"]))

    # determine selection count
    if n_tiles is not None:
        n_intact = min(n_tiles, len(wm_sp_stats))
        selection_method = f"n_tiles={n_tiles}"
    else:
        if percentile is None:
            percentile = 10.0
        n_intact = max(1, int(len(wm_sp_stats) * float(percentile) / 100.0))
        selection_method = f"percentile={percentile}"

    intact_sp_stats = wm_sp_stats[:n_intact]
    intact_sp_ids = [int(s["superpixel_id"]) for s in intact_sp_stats]

    if intact_sp_stats:
        print(f"[Analysis] Selected {len(intact_sp_ids)} intact superpixels ({selection_method})")
        print(f"[Analysis] Attention range for selected: {intact_sp_stats[0]['mean_attention']:.6f} - {intact_sp_stats[-1]['mean_attention']:.6f}")

    # Save CSV if path provided
    if save_csv_path is not None:
        try:
            df = pd.DataFrame(wm_sp_stats)
            df["superpixel_id"] = df["superpixel_id"].astype(int)
            df["mean_attention"] = df["mean_attention"].astype(float)
            df["is_intact"] = df["superpixel_id"].isin(intact_sp_ids)
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            df.to_csv(save_csv_path, index=False)
            print(f"[Output] WM superpixel CSV saved → {save_csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed saving CSV ({save_csv_path}): {e}")

    return intact_sp_ids, wm_sp_stats

# -------------------------
# Mask creation & TIFF handling
# -------------------------
def create_binary_masks(seg, intact_sp_ids, mask_wm):
    """
    Create binary masks for intact regions and their borders (only within WM).
    """
    mask_intact_sp = np.zeros_like(seg, dtype=bool)
    for sp_id in intact_sp_ids:
        mask_intact_sp |= (seg == sp_id)

    mask_intact = mask_intact_sp & mask_wm
    mask_borders = find_boundaries(mask_intact, mode='thick')

    print(f"[Masks] Intact sp pixels before WM filter: {np.sum(mask_intact_sp)}")
    print(f"[Masks] Intact pixels after WM filter: {np.sum(mask_intact)}")
    print(f"[Masks] Border pixels: {np.sum(mask_borders)}")
    return mask_intact, mask_borders

def load_and_downsample_tiff(tiff_path, target_shape):
    """
    Load original TIFF and downsample to target_shape (height, width).
    """
    print(f"[TIFF] Loading: {tiff_path}")
    try:
        import tifffile as tiff
        arr = tiff.imread(tiff_path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        rgb_full = arr.astype(np.uint8)
    except Exception:
        rgb_full = np.array(Image.open(tiff_path).convert("RGB"))
    th, tw = target_shape
    rgb_downsampled = np.array(Image.fromarray(rgb_full).resize((tw, th), Image.BILINEAR))
    print(f"[TIFF] Downsampled to {tw}x{th}")
    return rgb_downsampled

# -------------------------
# Visualization
# -------------------------
def create_overlay_visualization(rgb_img, mask_wm, mask_intact, mask_borders, output_path):
    """
    Overlay borders and intact regions on downsampled RGB image and save.
    """
    overlay = rgb_img.copy()
    overlay[~mask_wm] = overlay[~mask_wm] // 3
    overlay[mask_intact] = (0.7 * overlay[mask_intact] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[mask_borders] = [255, 0, 0]
    Image.fromarray(overlay).save(output_path)
    print(f"[Overlay] Saved to {output_path}")
    return overlay

def visualize_results(seg, mask_tissue, mask_wm, intact_sp_ids, wm_sp_stats,
                      mask_intact, mask_borders, rgb_img, output_dir):
    """
    Create and save a set of visualization figures summarizing the analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    th, tw = seg.shape

    # Masks grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(mask_tissue, cmap='gray'); axes[0, 0].set_title('Original Tissue Mask'); axes[0, 0].axis('off')
    axes[0, 1].imshow(mask_wm, cmap='gray'); axes[0, 1].set_title('White Matter Mask (WM ONLY)'); axes[0, 1].axis('off')
    axes[0, 2].imshow(mask_wm, cmap='Reds', alpha=0.5); axes[0, 2].set_title(f'WM Area ({100*np.sum(mask_wm)/mask_wm.size:.1f}%)'); axes[0, 2].axis('off')
    axes[1, 0].imshow(mask_intact, cmap='gray'); axes[1, 0].set_title(f'Intact WM Regions ({np.sum(mask_intact)} px)'); axes[1, 0].axis('off')
    axes[1, 1].imshow(mask_borders, cmap='gray'); axes[1, 1].set_title(f'Region Borders ({np.sum(mask_borders)} px)'); axes[1, 1].axis('off')

    combined_view = np.zeros((th, tw, 3), dtype=np.uint8)
    combined_view[mask_wm] = [100, 100, 100]
    combined_view[mask_intact] = [0, 255, 0]
    combined_view[mask_borders] = [255, 0, 0]
    axes[1, 2].imshow(combined_view); axes[1, 2].set_title('WM (Gray) + Intact (Green) + Borders (Red)'); axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "masks_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Overlays
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb_img); axes[0].set_title('Original Tissue (Downsampled)'); axes[0].axis('off')
    overlay_wm = rgb_img.copy(); overlay_wm[~mask_wm] = overlay_wm[~mask_wm] // 3
    axes[1].imshow(overlay_wm); axes[1].set_title('WM Region Highlighted'); axes[1].axis('off')
    overlay2 = rgb_img.copy(); overlay2[~mask_wm] = overlay2[~mask_wm] // 3
    overlay2[mask_intact] = (0.7 * overlay2[mask_intact] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
    overlay2[mask_borders] = [255, 0, 0]
    axes[2].imshow(overlay2); axes[2].set_title('Intact WM with Borders'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlay_visualization.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Attention distribution (WM only)
    if len(wm_sp_stats) > 0:
        all_attentions = [float(s["mean_attention"]) for s in wm_sp_stats]
        intact_attentions = [float(s["mean_attention"]) for s in wm_sp_stats if int(s["superpixel_id"]) in intact_sp_ids]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(all_attentions, bins=50, alpha=0.7, edgecolor='black'); axes[0].hist(intact_attentions, bins=20, alpha=0.7, edgecolor='black', label='Intact (bottom)')
        axes[0].axvline(np.median(all_attentions), color='blue', linestyle='--', label=f'Median: {np.median(all_attentions):.3f}')
        if intact_attentions:
            axes[0].axvline(max(intact_attentions), color='green', linestyle='--', label=f'Cutoff: {max(intact_attentions):.3f}')
        axes[0].set_xlabel('Mean Attention'); axes[0].set_ylabel('Count'); axes[0].set_title('Attention Distribution (WM)'); axes[0].legend(); axes[0].grid(alpha=0.3)

        sorted_att = sorted(all_attentions)
        cumulative = np.arange(1, len(sorted_att)+1) / len(sorted_att) * 100
        axes[1].plot(sorted_att, cumulative, linewidth=2)
        if intact_attentions:
            axes[1].axvline(max(intact_attentions), color='green', linestyle='--', label='Intact cutoff')
        axes[1].set_xlabel('Mean Attention'); axes[1].set_ylabel('Cumulative %'); axes[1].set_title('Cumulative Distribution (WM)'); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attention_distribution.png"), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[Visualizations] Saved figures to {output_dir}")

# -------------------------
# Save everything
# -------------------------
def save_results(intact_sp_ids, wm_sp_stats, seg, mask_tissue, mask_wm,
                mask_intact, mask_borders, output_dir, percentile=None, n_tiles=None,
                csv_path=None):
    """
    Save binary masks, overlays, JSON summaries and ensure CSV is saved (either written here or was written earlier).
    Returns the effective CSV path (where WM stats CSV resides).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save binary masks (.npy)
    np.save(os.path.join(output_dir, "mask_white_matter.npy"), mask_wm)
    np.save(os.path.join(output_dir, "mask_intact_tissue.npy"), mask_intact)
    np.save(os.path.join(output_dir, "mask_borders.npy"), mask_borders)

    # Save binary masks (.png)
    Image.fromarray((mask_intact * 255).astype(np.uint8)).save(os.path.join(output_dir, "mask_intact_tissue.png"))
    Image.fromarray((mask_borders * 255).astype(np.uint8)).save(os.path.join(output_dir, "mask_borders.png"))
    Image.fromarray((mask_wm * 255).astype(np.uint8)).save(os.path.join(output_dir, "mask_white_matter.png"))

    print(f"[Saved] Binary masks (.npy and .png) to {output_dir}")

    # Intact superpixel stats and JSON
    intact_stats = [s for s in wm_sp_stats if int(s["superpixel_id"]) in intact_sp_ids]
    selection_info = {}
    if n_tiles is not None:
        selection_info["selection_method"] = "n_tiles"
        selection_info["n_tiles"] = int(n_tiles)
    else:
        selection_info["selection_method"] = "percentile"
        selection_info["percentile"] = float(percentile) if percentile is not None else 10.0

    with open(os.path.join(output_dir, "intact_superpixels.json"), "w") as f:
        json.dump({
            **selection_info,
            "n_intact": len(intact_sp_ids),
            "n_total_wm": len(wm_sp_stats),
            "intact_superpixel_ids": intact_sp_ids,
            "intact_superpixel_stats": intact_stats
        }, f, indent=2)

    # Summary JSON
    all_attentions = [float(s["mean_attention"]) for s in wm_sp_stats] if wm_sp_stats else []
    intact_attentions = [float(s["mean_attention"]) for s in intact_stats] if intact_stats else []
    summary = {
        "analysis_type": "intact_white_matter",
        **selection_info,
        "total_superpixels_in_wm": len(wm_sp_stats),
        "intact_superpixels": len(intact_sp_ids),
        "tissue_pixels": int(np.sum(mask_tissue)),
        "wm_pixels": int(np.sum(mask_wm)),
        "non_wm_pixels_excluded": int(np.sum(~mask_wm)),
        "intact_pixels": int(np.sum(mask_intact)),
        "border_pixels": int(np.sum(mask_borders)),
        "attention_stats": {
            "all_wm_mean": float(np.mean(all_attentions)) if all_attentions else None,
            "all_wm_median": float(np.median(all_attentions)) if all_attentions else None,
            "all_wm_std": float(np.std(all_attentions)) if all_attentions else None,
            "intact_mean": float(np.mean(intact_attentions)) if intact_attentions else None,
            "intact_max": float(max(intact_attentions)) if intact_attentions else None,
            "intact_min": float(min(intact_attentions)) if intact_attentions else None,
        }
    }
    with open(os.path.join(output_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # CSV: if csv_path was provided and file exists, assume identify_intact_superpixels saved it.
    # Otherwise, write CSV here.
    effective_csv_path = csv_path or os.path.join(output_dir, "wm_superpixel_stats.csv")
    if os.path.exists(effective_csv_path):
        print(f"[CSV] Found existing CSV at {effective_csv_path}")
    else:
        try:
            df = pd.DataFrame(wm_sp_stats)
            df["superpixel_id"] = df["superpixel_id"].astype(int)
            df["mean_attention"] = df["mean_attention"].astype(float)
            df["is_intact"] = df["superpixel_id"].isin(intact_sp_ids)
            df.to_csv(effective_csv_path, index=False)
            print(f"[CSV] Written WM superpixel CSV → {effective_csv_path}")
        except Exception as e:
            print(f"[ERROR] Could not write CSV at {effective_csv_path}: {e}")

    print(f"[Results] Saved results to {output_dir}")
    return effective_csv_path

# -------------------------
# CLI / main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze superpixel heatmaps for intact white matter")
    parser.add_argument("--run_dir", required=True, help="Directory with superpixel results")
    parser.add_argument("--wm_mask", required=True, help="Path to white matter mask image (full res)")
    parser.add_argument("--tiff_image", required=True, help="Path to original TIFF image")
    parser.add_argument("--percentile", type=float, default=None, help="Bottom X%% attention within WM to consider as intact (default: 10 if not specified)")
    parser.add_argument("--n_tiles", type=int, default=None, help="Exact number of tiles with lowest attention to select (overrides --percentile)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: run_dir/intact_analysis)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_dir, "intact_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Loading] Superpixel results from {args.run_dir}")
    seg, mask_tissue, sp_stats, heat_sp, params = load_superpixel_results(args.run_dir)

    print(f"[Loading] White matter mask from {args.wm_mask}")
    mask_wm = apply_wm_mask(args.wm_mask, seg.shape, args.run_dir)

    # Determine selection method display
    if args.n_tiles is not None:
        print(f"[Analyzing] Selecting {args.n_tiles} tiles with lowest attention within WM ONLY")
    else:
        percentile = args.percentile if args.percentile is not None else 10.0
        print(f"[Analyzing] Identifying bottom {percentile}% attention within WM ONLY")

    # Prepare CSV path (we prefer to pass a target path so identify_intact_superpixels can save early)
    csv_target = os.path.join(args.output_dir, "wm_superpixel_stats.csv")

    intact_sp_ids, wm_sp_stats = identify_intact_superpixels(
        seg=seg,
        mask_wm=mask_wm,
        sp_stats=sp_stats,
        percentile=args.percentile,
        n_tiles=args.n_tiles,
        save_csv_path=csv_target
    )

    if len(intact_sp_ids) == 0:
        print("[ERROR] No intact superpixels found. Exiting.")
        return

    print("[Creating] Binary masks for intact regions and borders")
    mask_intact, mask_borders = create_binary_masks(seg, intact_sp_ids, mask_wm)

    print("[Loading] Original TIFF and downsampling")
    rgb_downsampled = load_and_downsample_tiff(args.tiff_image, seg.shape)

    print("[Creating] Overlay visualization")
    create_overlay_visualization(
        rgb_downsampled, mask_wm, mask_intact, mask_borders,
        os.path.join(args.output_dir, "overlay_borders_on_tissue.png")
    )

    print("[Visualizing] Creating analysis plots")
    visualize_results(seg, mask_tissue, mask_wm, intact_sp_ids, wm_sp_stats,
                      mask_intact, mask_borders, rgb_downsampled, args.output_dir)

    print("[Saving] Writing results and binary masks")
    effective_csv = save_results(intact_sp_ids, wm_sp_stats, seg, mask_tissue, mask_wm,
                mask_intact, mask_borders, args.output_dir, percentile=args.percentile,
                n_tiles=args.n_tiles, csv_path=csv_target)

    print("\n[Done] Analysis complete!")
    print(f"       Outputs written to: {args.output_dir}")
    print(f"       WM superpixel CSV: {effective_csv}")
    print("       Binary masks (.npy/.png), overlays and JSON summaries saved.")

if __name__ == "__main__":
    main()