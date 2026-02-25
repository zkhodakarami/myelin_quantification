#!/usr/bin/env python3
# Resume-able CLAM inference on local TIFFs
# - Resumes from existing CSVs / features / attention if present
# - OOM-safe: features on CPU(fp16) + chunked attention/pooling on GPU
# - Safe heatmaps on tile-grid (not full-res), capped overlay size
# - Vicinity-aware attention (Gaussian smoothing on tile grid)
# - SLIC-like superpixels on thumbnail, aggregates (smoothed) attention
# - FIXED: Mask-based filtering to exclude holes/background from superpixels
#
# SCANNING PARAMETERS:
# - Resolution: 0.4 μm/pixel (20x magnification)
# - Tile size: 256×256 pixels = 102.4 μm × 102.4 μm physical area
# - Matching paper: McKenzie et al. 2022 (PMC9490907) which used 0.5 μm/pixel, 256×256 tiles

import os, json, csv, time, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50
from torchvision.transforms import InterpolationMode

# Optional superpixels
try:
    from skimage.segmentation import slic, mark_boundaries
    from skimage.color import rgb2lab
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

# ======================= Defaults =======================
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH      = os.path.join(_SCRIPT_DIR, "weights", "s_2_checkpoint.pt")
LOCAL_RESNET50 = os.path.join(_SCRIPT_DIR, "weights", "resnet50-11ad3fa6.pth")
MODEL_TYPE     = "clam_sb"
OUT_DIR        = "./phas_clam_outputs"

LEVEL          = 0         # logical level tag for outputs
TILE_SIZE      = 256       # Match paper: 256x256 pixels (102.4 μm at 0.4 μm/pixel)
STRIDE         = 256       # Non-overlapping tiles
TISSUE_S_THRESH= 0.05

BATCH_FEATS    = 256
MAX_WORKERS    = max(8, os.cpu_count() or 8)

# MIL chunking
ATTN_CHUNK     = 4096
POOL_CHUNK     = 4096

# Heatmap / overlay
HEATMAP_SCALE  = 1         # 1 = one cell per tile; >1 thickens cells via np.kron
OVERLAY_LONG   = 4000      # cap longest side of overlay image

# SLIC superpixels (thumbnail)
SP_N_SEGMENTS  = 2000
SP_COMPACTNESS = 10.0
SP_SIGMA       = 1.0
SP_LONG        = 4000

# Vicinity smoothing on tile grid
SMOOTH_SIGMA_TILES = 1.5   # Gaussian sigma in tile units (e.g., 1.5 tiles)
# ========================================================

# Keep CPU BLAS reasonable
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
try:
    torch.set_num_threads(8)
except Exception:
    pass

from models.model_clam import CLAM_SB, CLAM_MB

# ======================= Utilities =======================
def file_stem(path):
    b = os.path.basename(path)
    return os.path.splitext(b)[0]

def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

def gpu_mem_str():
    if not torch.cuda.is_available(): return "n/a"
    return f"{torch.cuda.memory_allocated()/1e9:.2f} GB alloc / {torch.cuda.memory_reserved()/1e9:.2f} GB reserved"

def build_feature_extractor_1024(device, resnet50_path):
    if not os.path.isfile(resnet50_path):
        raise FileNotFoundError(resnet50_path)
    rn = resnet50(weights=None)
    sd = torch.load(resnet50_path, map_location="cpu")
    rn.load_state_dict(sd, strict=False)
    trunk = nn.Sequential(
        rn.conv1, rn.bn1, rn.relu, rn.maxpool,
        rn.layer1, rn.layer2, rn.layer3,
        nn.AdaptiveAvgPool2d((1, 1)),
    ).eval().to(device)
    if device.type == "cuda":
        trunk = trunk.to(memory_format=torch.channels_last)
    return trunk

def load_clam(ckpt_path, model_type="clam_sb", device=torch.device("cpu")):
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")
    sd = None
    if isinstance(obj, dict):
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
        else:
            for k in ("model_state_dict","state_dict","net","model","module","params"):
                if isinstance(obj.get(k), dict):
                    sd = obj[k]; break
    if sd is None and hasattr(obj, "state_dict"):
        try: sd = obj.state_dict()
        except Exception: pass
    if sd is None:
        raise ValueError("No state_dict in checkpoint")

    def find_classifier_weight(sdict):
        for k, v in sdict.items():
            if isinstance(v, torch.Tensor) and v.ndim==2 and (
                k.endswith("classifiers.weight") or
                k.endswith("classifier.weight") or
                k.endswith("fc.weight") or "classif" in k
            ):
                return k, v
        return None, None

    _, W = find_classifier_weight(sd)
    n_classes = int(W.shape[0]) if W is not None else 2
    print(f"[load_clam] n_classes={n_classes}")

    Model = CLAM_SB if model_type.lower()=="clam_sb" else CLAM_MB
    model = Model(n_classes=n_classes, dropout=True)
    try:
        model.load_state_dict(sd, strict=False)
    except RuntimeError:
        from collections import OrderedDict
        sd = OrderedDict((k.replace("module.",""), v) for k, v in sd.items())
        model.load_state_dict(sd, strict=False)
    return model.eval().to(device)

def infer_attn_in(model):
    for m in model.attention_net.modules():
        if isinstance(m, nn.Linear): return m.in_features
    return None

def infer_cls_in(model):
    for m in model.classifiers.modules():
        if isinstance(m, nn.Linear): return m.in_features
    return None

def ensure_dim(X, target_dim):
    N, D = X.shape
    if D == target_dim: return X
    if D > target_dim:  return X[:, :target_dim]
    out = torch.zeros((N, target_dim), dtype=X.dtype, device=X.device)
    out[:, :D] = X
    return out

def build_centers_grid(W, H, tile, stride):
    xs = np.arange(tile//2, W, stride, dtype=np.int32)
    ys = np.arange(tile//2, H, stride, dtype=np.int32)
    XX, YY = np.meshgrid(xs, ys)
    centers = np.stack([XX.ravel(), YY.ravel()], axis=1)  # [N,2]
    return centers, xs, ys

def coarse_tissue_mask(rgb_np, s_thresh, down=8):
    H, W, _ = rgb_np.shape
    h2, w2 = max(1, H//down), max(1, W//down)
    small = Image.fromarray(rgb_np, "RGB").resize((w2, h2), Image.BILINEAR).convert("HSV")
    S = np.asarray(small, dtype=np.uint8)[...,1]
    keep = S > int(s_thresh*255)
    mask = np.asarray(Image.fromarray((keep.astype(np.uint8)*255), "L").resize((W, H), Image.NEAREST)) > 0
    return mask

def mask_centers(centers, mask):
    H, W = mask.shape
    cx = np.clip(centers[:,0], 0, W-1)
    cy = np.clip(centers[:,1], 0, H-1)
    keep = mask[cy, cx]
    return centers[keep]

def gaussian_kernel1d(sigma, radius=None):
    if sigma <= 0: return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = max(1, int(3.0 * sigma + 0.5))
    x = np.arange(-radius, radius+1, dtype=np.float32)
    k = np.exp(-0.5 * (x/sigma)**2)
    k /= k.sum()
    return k

def separable_gaussian_2d(arr, sigma):
    if sigma <= 0: return arr.copy()
    k = gaussian_kernel1d(sigma)
    # horizontal
    pad = len(k)//2
    tmp = np.pad(arr, ((0,0),(pad,pad)), mode="reflect")
    tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, tmp)
    # vertical
    tmp = np.pad(tmp, ((pad,pad),(0,0)), mode="reflect")
    tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, tmp)
    return tmp

def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def load_csv_int_coords(path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    if arr.ndim == 1 and arr.size == 0:
        return np.empty((0,2), dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

# ======================= Main =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt", default=CKPT_PATH)
    parser.add_argument("--resnet50", default=LOCAL_RESNET50)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--model_type", default=MODEL_TYPE, choices=["clam_sb","clam_mb"])

    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--tile", type=int, default=TILE_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--tissue_s_thresh", type=float, default=TISSUE_S_THRESH)
    parser.add_argument("--down_for_mask", type=int, default=8)

    parser.add_argument("--batch_feats", type=int, default=BATCH_FEATS)
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--attn_chunk", type=int, default=ATTN_CHUNK)
    parser.add_argument("--pool_chunk", type=int, default=POOL_CHUNK)

    parser.add_argument("--heatmap_scale", type=int, default=HEATMAP_SCALE)
    parser.add_argument("--overlay_long_side", type=int, default=OVERLAY_LONG)

    # Superpixels
    parser.add_argument("--sp_enable", action="store_true", help="Enable SLIC-like superpixels")
    parser.add_argument("--sp_n", type=int, default=SP_N_SEGMENTS)
    parser.add_argument("--sp_compact", type=float, default=SP_COMPACTNESS)
    parser.add_argument("--sp_sigma", type=float, default=SP_SIGMA)
    parser.add_argument("--sp_long_side", type=int, default=SP_LONG)

    # Vicinity smoothing on tile grid
    parser.add_argument("--smooth_sigma_tiles", type=float, default=SMOOTH_SIGMA_TILES)

    # Resume flags
    parser.add_argument("--resume_tiles", action="store_true", help="Reuse existing grid CSVs if found")
    parser.add_argument("--resume_feats", action="store_true", help="Reuse features if features_fp16.pt exists")
    parser.add_argument("--resume_attn", action="store_true", help="Reuse attention if attention_raw.npy exists")

    args = parser.parse_args()

    # ---- GPU setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA GPU required.")
    print(f"[device] {device} :: {gpu_name()}")
    print(f"[cuda] CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    use_amp = (device.type == "cuda")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ---- Load image ----
    img_path = os.path.expanduser(args.image)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(img_path)

    # TIFF path → RGB array
    rgb_np = None
    if img_path.lower().endswith((".tif",".tiff")):
        try:
            import tifffile as tiff
            arr = tiff.imread(img_path)
            if arr.ndim == 2:
                arr = np.stack([arr,arr,arr], axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[...,:3]
            rgb_np = arr.astype(np.uint8)
        except Exception:
            pass
    if rgb_np is None:
        rgb_np = np.asarray(Image.open(img_path).convert("RGB"))
    H_lvl, W_lvl = rgb_np.shape[0], rgb_np.shape[1]
    print(f"[image] {img_path}  size={W_lvl}x{H_lvl}")

    # ---- Output dirs ----
    slide_name = file_stem(img_path)
    run_dir = os.path.join(args.out, f"slide_{slide_name}", f"L{args.level}_T{args.tile}_S{args.stride}")
    os.makedirs(run_dir, exist_ok=True)

    # Paths for resume artifacts
    p_all   = os.path.join(run_dir, "grid_all_tiles_level.csv")
    p_keptL = os.path.join(run_dir, "grid_kept_tiles_level.csv")
    p_keptF = os.path.join(run_dir, "grid_kept_tiles_full.csv")
    p_feats = os.path.join(run_dir, "features_fp16.pt")
    p_attn  = os.path.join(run_dir, "attention_raw.npy")
    p_attn_sm = os.path.join(run_dir, "attention_smoothed.npy")
    p_mask  = os.path.join(run_dir, "tissue_mask.npy")

    # ---- Tile grid + (optional) resume ----
    if args.resume_tiles and os.path.isfile(p_all) and os.path.isfile(p_keptL) and os.path.isfile(p_mask):
        print("[resume] Loading existing tile CSVs and tissue mask")
        centers_all = load_csv_int_coords(p_all)
        centers_lvl = load_csv_int_coords(p_keptL)
        mask = np.load(p_mask)
        xs = np.arange(args.tile//2, W_lvl, args.stride, dtype=np.int32)
        ys = np.arange(args.tile//2, H_lvl, args.stride, dtype=np.int32)
    else:
        centers_all, xs, ys = build_centers_grid(W_lvl, H_lvl, args.tile, args.stride)
        mask = coarse_tissue_mask(rgb_np, args.tissue_s_thresh, down=args.down_for_mask)
        centers_lvl = mask_centers(centers_all, mask)
        print(f"[grid] tiles before mask: {centers_all.shape[0]}")
        print(f"[grid] tiles after mask:  {centers_lvl.shape[0]}")
        save_csv(p_all,   ["x_lvl","y_lvl"], centers_all.tolist())
        save_csv(p_keptL, ["x_lvl","y_lvl"], centers_lvl.tolist())
        save_csv(p_keptF, ["x_full","y_full"], centers_lvl.tolist())
        np.save(p_mask, mask)

    if centers_lvl.shape[0] == 0:
        raise RuntimeError("No tiles to process.")

    # ---- Preprocessing ----
    # Match paper: 256x256 tiles -> center crop to 224x224 (no upsampling)
    tfm = T.Compose([
        T.ToPILImage(mode="RGB"),
        T.CenterCrop(224),  # Directly crop 256 -> 224 (no resize to avoid upsampling)
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # crop helper (zero-pad borders)
    half = args.tile // 2
    def crop_np(center_xy):
        x, y = int(center_xy[0]), int(center_xy[1])
        x0, x1 = max(0, x-half), min(W_lvl, x+half)
        y0, y1 = max(0, y-half), min(H_lvl, y+half)
        patch = np.zeros((args.tile, args.tile, 3), dtype=np.uint8)
        px0, py0 = half - (x - x0), half - (y - y0)
        patch[py0:py0 + (y1-y0), px0:px0 + (x1-x0)] = rgb_np[y0:y1, x0:x1]
        return patch

    # ---- Models ----
    trunk = build_feature_extractor_1024(device, args.resnet50)
    clam  = load_clam(args.ckpt, args.model_type, device)
    attn_in = infer_attn_in(clam); cls_in = infer_cls_in(clam)
    if attn_in is None or cls_in is None:
        raise RuntimeError("Could not infer attn/classifier input dims from model.")
    print(f"[dims] attn_in={attn_in}, cls_in={cls_in}")

    # Optional compile
    try:
        trunk = torch.compile(trunk, mode="max-autotune")
        clam  = torch.compile(clam,  mode="max-autotune")
        print("[compile] torch.compile enabled")
    except Exception as e:
        print(f"[compile] skipped: {e}")

    # ---- Feature extraction (resume-aware) ----
    if args.resume_feats and os.path.isfile(p_feats):
        bag_cpu = torch.load(p_feats, map_location="cpu")
        if bag_cpu.dtype != torch.float16: bag_cpu = bag_cpu.half()
        kept_coords = centers_lvl.tolist()
        print(f"[resume] Loaded features: {list(bag_cpu.shape)}")
    else:
        feats_cpu, kept_coords = [], []
        def stream_crops():
            centers_list = centers_lvl.tolist()
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                for patch in ex.map(crop_np, centers_list, chunksize=256):
                    yield patch
        t0 = time.perf_counter()
        batch_imgs, batch_coords = [], []
        for (c, patch) in zip(centers_lvl.tolist(), tqdm(stream_crops(), total=centers_lvl.shape[0], desc="Crops+Feats")):
            x = tfm(patch)
            batch_imgs.append(x); batch_coords.append(tuple(c))
            if len(batch_imgs) >= args.batch_feats:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    X = torch.stack(batch_imgs, 0)
                    if use_amp: X = X.contiguous(memory_format=torch.channels_last).pin_memory()
                    X = X.to(device, non_blocking=True)
                    F = trunk(X).flatten(1)
                feats_cpu.append(F.to("cpu", dtype=torch.float16))
                kept_coords.extend(batch_coords)
                batch_imgs.clear(); batch_coords.clear()
        if batch_imgs:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                X = torch.stack(batch_imgs, 0)
                if use_amp: X = X.contiguous(memory_format=torch.channels_last).pin_memory()
                X = X.to(device, non_blocking=True)
                F = trunk(X).flatten(1)
            feats_cpu.append(F.to("cpu", dtype=torch.float16))
            kept_coords.extend(batch_coords)
        bag_cpu = torch.cat(feats_cpu, dim=0)
        torch.save(bag_cpu, p_feats)
        print(f"[timing] feature extraction for {bag_cpu.shape[0]} tiles: {time.perf_counter()-t0:.2f}s")
        print("[mem after feats]", gpu_mem_str())

    N = bag_cpu.shape[0]
    assert N == len(kept_coords), "Features and coords length mismatch."

    # ---- Attention & pooling (resume-aware) ----
    if args.resume_attn and os.path.isfile(p_attn):
        att = np.load(p_attn)
        print(f"[resume] Loaded attention: {att.shape}")
    else:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            dummy = ensure_dim(bag_cpu[:1].to(device, non_blocking=True), attn_in)
            out = clam.attention_net(dummy)
            K = (out[0] if isinstance(out, tuple) else out).shape[-1]
        del dummy, out
        print(f"[MIL] attention heads K={K}")

        A_logits_cpu = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            for i in tqdm(range(0, N, args.attn_chunk), desc="Attention logits (chunked)"):
                x = bag_cpu[i:i+args.attn_chunk].to(device, non_blocking=True)
                h_attn = ensure_dim(x, attn_in)
                att_out = clam.attention_net(h_attn)
                a_logits = att_out[0] if isinstance(att_out, tuple) else att_out
                A_logits_cpu.append(a_logits.float().cpu())
        A_logits = torch.cat(A_logits_cpu, dim=0)
        A = torch.softmax(A_logits.T, dim=1)
        
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            K = A.shape[0]
            cls_in = infer_cls_in(clam)
            M = torch.zeros((K, cls_in), device=device, dtype=torch.float32)
            for i in tqdm(range(0, N, args.pool_chunk), desc="Pooling (chunked)"):
                x = bag_cpu[i:i+args.pool_chunk].to(device, non_blocking=True)
                h_cls = ensure_dim(x, cls_in)
                a_chunk = A[:, i:i+args.pool_chunk].to(device, non_blocking=True)
                M = M + (a_chunk @ h_cls.float())
            logits = clam.classifiers(M)
            Y_prob = torch.softmax(logits, dim=1)
        slide_prob = float(Y_prob.max().float().cpu())
        slide_pred = int(Y_prob.argmax(dim=-1).item())
        k_idx = 0 if K == 1 else min(slide_pred, K-1)
        att = A[k_idx].cpu().numpy()
        np.save(p_attn, att)
        with open(os.path.join(run_dir, "predictions.json"), "w") as f:
            json.dump({"slide_name": slide_name, "pred": slide_pred, "max_prob": slide_prob}, f, indent=2)
        print(f"[pred] class={slide_pred} prob={slide_prob:.4f}")
        print("[mem end]", gpu_mem_str())

    # ---- Build tile-grid heatmap (and smoothed attention) ----
    xs = np.arange(args.tile//2, W_lvl, args.stride, dtype=np.int32)
    ys = np.arange(args.tile//2, H_lvl, args.stride, dtype=np.int32)
    gx, gy = len(xs), len(ys)

    def to_idx(x, y):
        ix = (x - (args.tile//2)) // args.stride
        iy = (y - (args.tile//2)) // args.stride
        return int(ix), int(iy)

    heat_grid  = np.zeros((gy, gx), dtype=np.float32)
    count_grid = np.zeros((gy, gx), dtype=np.uint16)
    for (x,y), a in zip(kept_coords, att):
        ix, iy = to_idx(x, y)
        if 0 <= ix < gx and 0 <= iy < gy:
            heat_grid[iy, ix] += float(a)
            count_grid[iy, ix] += 1
    m = count_grid > 0
    if m.any():
        heat_grid[m] /= count_grid[m]
        vmin, vmax = float(heat_grid[m].min()), float(heat_grid[m].max())
        if vmax > vmin:
            heat_grid = (heat_grid - vmin) / (vmax - vmin)

    smooth_sigma = max(0.0, float(args.smooth_sigma_tiles))
    heat_grid_sm = separable_gaussian_2d(heat_grid, smooth_sigma) if smooth_sigma > 0 else heat_grid.copy()

    att_sm = []
    for (x,y) in kept_coords:
        ix, iy = to_idx(x, y)
        ix = np.clip(ix, 0, gx-1); iy = np.clip(iy, 0, gy-1)
        att_sm.append(float(heat_grid_sm[iy, ix]))
    att_sm = np.array(att_sm, dtype=np.float32)
    np.save(p_attn_sm, att_sm)

    with open(os.path.join(run_dir, "attention_tiles.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x_lvl","y_lvl","x_full","y_full","attention_raw","attention_smooth"])
        for (x,y), a, s in zip(kept_coords, att, att_sm):
            w.writerow([x, y, x, y, float(a), float(s)])

    # Save heatmaps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = max(1, int(args.heatmap_scale))
    def save_grid_png(name, grid):
        grid_vis = np.kron(grid, np.ones((scale, scale), dtype=np.float32)) if scale>1 else grid
        plt.imsave(os.path.join(run_dir, name), grid_vis, vmin=0.0, vmax=1.0, cmap="jet")

    save_grid_png("heatmap_grid_raw.png", heat_grid)
    save_grid_png("heatmap_grid_smooth.png", heat_grid_sm)

    # Overlay on downsampled RGB
    try:
        long_side = max(W_lvl, H_lvl)
        s = min(1.0, float(args.overlay_long_side) / float(long_side))
        ow, oh = max(1, int(W_lvl*s)), max(1, int(H_lvl*s))
        rgb_small = Image.fromarray(rgb_np, "RGB").resize((ow, oh), Image.BILINEAR)

        heat_img_raw = Image.fromarray((heat_grid * 255).astype(np.uint8), "L").resize((ow, oh), Image.BILINEAR)
        heat_img_sm  = Image.fromarray((heat_grid_sm * 255).astype(np.uint8), "L").resize((ow, oh), Image.BILINEAR)

        cm = plt.cm.get_cmap("jet")
        def to_color(imL):
            arr = np.array(imL)/255.0
            return Image.fromarray((cm(arr)[..., :3]*255).astype(np.uint8), "RGB")

        overlay_raw = Image.blend(rgb_small, to_color(heat_img_raw), alpha=0.45)
        overlay_sm  = Image.blend(rgb_small, to_color(heat_img_sm),  alpha=0.45)
        overlay_raw.save(os.path.join(run_dir, "overlay_grid_raw.png"))
        overlay_sm.save(os.path.join(run_dir, "overlay_grid_smooth.png"))
    except Exception as e:
        with open(os.path.join(run_dir, "overlay_warning.txt"), "w") as f:
            f.write(f"Overlay skipped: {e}\n")

    # ---- Superpixels (optional, on smoothed attention with mask filtering) ----
    if args.sp_enable:
        if not HAVE_SKIMAGE:
            print("[superpixels] scikit-image not installed; skipping.")
        else:
            try:
                # Build thumbnail
                long_side = max(W_lvl, H_lvl)
                ssp = min(1.0, float(args.sp_long_side) / float(long_side))
                tw, th = max(1, int(W_lvl*ssp)), max(1, int(H_lvl*ssp))
                thumb = Image.fromarray(rgb_np, "RGB").resize((tw, th), Image.BILINEAR)
                thumb_np = np.asarray(thumb)
                
                # Resize tissue mask to thumbnail size for filtering
                mask_thumb = Image.fromarray(mask.astype(np.uint8)*255, "L").resize((tw, th), Image.NEAREST)
                mask_thumb = np.asarray(mask_thumb) > 0  # [th, tw] bool
                
                # Run SLIC segmentation with mask to exclude background/holes
                seg = slic(rgb2lab(thumb_np), n_segments=args.sp_n,
                           compactness=args.sp_compact, sigma=args.sp_sigma,
                           mask=mask_thumb,  # CRITICAL: Only segment tissue regions
                           start_label=0, enforce_connectivity=True)

                # Map smoothed per-tile attention to superpixels via tile centers
                sp_sum, sp_cnt = {}, {}
                sx, sy = tw / float(W_lvl), th / float(H_lvl)
                
                for (x,y), a in zip(kept_coords, att_sm):
                    ix = int(x * sx); iy = int(y * sy)
                    # Double-check we're in tissue region
                    if 0 <= ix < tw and 0 <= iy < th and mask_thumb[iy, ix]:
                        sp = int(seg[iy, ix])
                        sp_sum[sp] = sp_sum.get(sp, 0.0) + float(a)
                        sp_cnt[sp] = sp_cnt.get(sp, 0) + 1

                # Build superpixel heatmap (only for tissue superpixels)
                heat_sp = np.zeros((th, tw), dtype=np.float32)
                for sp, ssum in sp_sum.items():
                    heat_sp[seg == sp] = ssum / max(1, sp_cnt.get(sp, 1))

                # Normalize only the tissue regions
                mask_sp = (heat_sp > 0) & mask_thumb
                if mask_sp.any():
                    vmin, vmax = float(heat_sp[mask_sp].min()), float(heat_sp[mask_sp].max())
                    if vmax > vmin:
                        heat_sp_norm = np.zeros_like(heat_sp)
                        heat_sp_norm[mask_sp] = (heat_sp[mask_sp] - vmin) / (vmax - vmin)
                        heat_sp = heat_sp_norm

                # Save superpixel heatmap + overlay with boundaries
                plt.imsave(os.path.join(run_dir, "heatmap_superpixels.png"), heat_sp, vmin=0.0, vmax=1.0, cmap="jet")
                cm = plt.cm.get_cmap("jet")
                heat_rgb = (cm(heat_sp)[..., :3] * 255).astype(np.uint8)
                overlay_sp = Image.blend(thumb, Image.fromarray(heat_rgb, "RGB"), alpha=0.45)
                ov_np = np.array(overlay_sp)
                ov_bound = (mark_boundaries(ov_np, seg, color=(1,1,0), mode="thick") * 255).astype(np.uint8)
                Image.fromarray(ov_bound).save(os.path.join(run_dir, "overlay_superpixels.png"))

                # Save raw artifacts
                np.save(os.path.join(run_dir, "superpixels_labels_thumb.npy"), seg)
                np.save(os.path.join(run_dir, "superpixels_mask_thumb.npy"), mask_thumb)
                
                # Save superpixel statistics
                sp_stats = []
                for sp in sorted(sp_sum.keys()):
                    sp_stats.append({
                        "superpixel_id": int(sp),
                        "mean_attention": float(sp_sum[sp] / max(1, sp_cnt[sp])),
                        "n_tiles": int(sp_cnt[sp]),
                        "area_pixels": int(np.sum(seg == sp))
                    })
                with open(os.path.join(run_dir, "superpixels_stats.json"), "w") as f:
                    json.dump(sp_stats, f, indent=2)
                
                with open(os.path.join(run_dir, "superpixels_params.json"), "w") as f:
                    json.dump({
                        "n_segments": int(args.sp_n),
                        "compactness": float(args.sp_compact),
                        "sigma": float(args.sp_sigma),
                        "thumb_size_wh": [int(tw), int(th)],
                        "scale_thumb": float(ssp),
                        "mask_filtering": True,
                        "tissue_pixels": int(np.sum(mask_thumb)),
                        "total_pixels": int(th * tw),
                        "tissue_fraction": float(np.sum(mask_thumb) / (th * tw))
                    }, f, indent=2)
                    
                print(f"[superpixels] Generated {len(sp_sum)} tissue superpixels (excluded background/holes)")
                
            except Exception as e:
                with open(os.path.join(run_dir, "superpixels_warning.txt"), "w") as f:
                    f.write(str(e)+"\n")
                print(f"[superpixels] Error: {e}")

    # ---- Metadata ----
    meta = {
        "slide_name": slide_name,
        "level": int(args.level),
        "tile_size": int(args.tile),
        "stride": int(args.stride),
        "tile_physical_size_um": float(args.tile * 0.4),  # Document physical size
        "pixel_size_um": 0.4,
        "magnification": "20x",
        "tissue_s_thresh": float(args.tissue_s_thresh),
        "grid_tiles_total": int(len(xs) * len(ys)),
        "tiles_kept_after_filters": int(centers_lvl.shape[0]),
        "level_dimensions_wh": [int(W_lvl), int(H_lvl)],
        "batch_feats": int(args.batch_feats),
        "max_workers": int(args.max_workers),
        "amp": True,
        "device": str(device),
        "attn_chunk": int(args.attn_chunk),
        "pool_chunk": int(args.pool_chunk),
        "heatmap_scale": int(args.heatmap_scale),
        "overlay_long_side": int(args.overlay_long_side),
        "smooth_sigma_tiles": float(args.smooth_sigma_tiles),
        "sp_enable": bool(args.sp_enable),
        "sp_n": int(args.sp_n),
        "sp_compact": float(args.sp_compact),
        "sp_sigma": float(args.sp_sigma),
        "sp_long_side": int(args.sp_long_side),
        "ckpt_path": os.path.abspath(args.ckpt),
        "resnet50_path": os.path.abspath(args.resnet50),
        "image_path": os.path.abspath(img_path),
    }
    with open(os.path.join(run_dir, "tiling_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done → {os.path.abspath(run_dir)}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    main()