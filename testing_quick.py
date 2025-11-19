# File: testing_quick.py
"""
Quick evaluation on the UDC-SIT subset (data/UDC-SIT_subset).

- Loads full 4-ch .npy images from subset (train/val).
- Tiles into 256x256 patches, runs model, reconstructs full image.
- Computes PSNR, SSIM (Bayer domain) and LPIPS on simple RGB.
- Saves metrics locally and copies CSV + summary to Google Drive.

Assumptions:
- Subset structure:
    data/UDC-SIT_subset/
        train/input/*.npy
        train/GT/*.npy
        val/input/*.npy
        val/GT/*.npy
- Each .npy is (H, W, 4) with values in [0, 1023].
"""

import os
import sys
import csv
import numpy as np
from glob import glob

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast  # OK, just gives a FutureWarning for now

import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import imageio.v2 as imageio  # only needed if you add debugging PNGs

# Add external MambaIR path if needed
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)

from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent

# ---------------- CONFIG ----------------
SUBSET_ROOT = "data/UDC-SIT_subset"
SPLIT = "val"           # "train" or "val"
PATCH_SIZE = 256
BATCH_SIZE = 8          # patches per forward pass
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Which models to evaluate on subset
MODEL_CONFIGS = [
    {
        "name": "teacher_quick",
        "model_type": "teacher",
        "weights": "teacher_quick_1epoch.pth",
    },
    {
        "name": "student_quick",
        "model_type": "student",
        "weights": "student_quick_1epoch.pth",
    },
]

RESULTS_ROOT = "results_quick"
DRIVE_RESULTS_ROOT = "/content/drive/MyDrive/Computational Imaging Project/results_quick"

SAVE_NPY = True     # whether to save predicted 4-ch .npy full images
NUM_EXAMPLES = None # None = all, or set an int for debug
# ---------------------------------------


def load_full_npy_pair(input_path, gt_path):
    """
    Load 4-ch full .npy (H, W, 4) and return normalized CHW tensors in [0,1].

    Returns:
        udc: (4, H, W) float32, CPU
        gt:  (4, H, W) float32, CPU
    """
    udc = np.load(input_path)  # (H, W, 4)
    gt  = np.load(gt_path)

    assert udc.shape == gt.shape, f"Shape mismatch: {udc.shape} vs {gt.shape}"

    udc = torch.from_numpy(udc).permute(2, 0, 1).float() / 1023.0
    gt  = torch.from_numpy(gt).permute(2, 0, 1).float() / 1023.0
    return udc, gt


def pad_to_multiple(tensor, patch_size):
    """
    Pad CHW tensor so H,W are multiples of patch_size using reflect padding.

    Args:
        tensor: (C, H, W) torch.Tensor
        patch_size: int

    Returns:
        padded: (C, H_pad, W_pad) same device as input
        H_orig, W_orig: original spatial dims
    """
    _, H, W = tensor.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h == 0 and pad_w == 0:
        return tensor, H, W

    # F.pad pads last two dims as (left, right, top, bottom)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, H, W


def run_model_on_full_image(model, udc_full_4ch, patch_size, batch_size):
    """
    Run model on full 4-ch image by tiling into non-overlapping patches.

    Args:
        model: nn.Module on some device
        udc_full_4ch: (4, H, W) tensor in [0,1], CPU or GPU
        patch_size: int
        batch_size: int (# patches per forward pass)

    Returns:
        pred_full_4ch: (4, H_orig, W_orig) tensor in [0,1] on CPU
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        # Ensure input is on same device as model
        udc_full_4ch = udc_full_4ch.to(device)

        x, H_orig, W_orig = pad_to_multiple(udc_full_4ch, patch_size)  # (4, H_pad, W_pad) on device
        C, H_pad, W_pad = x.shape

        # Add batch dim
        x = x.unsqueeze(0)  # (1, C, H_pad, W_pad)

        # Prepare patch coordinates
        ys = list(range(0, H_pad, patch_size))
        xs = list(range(0, W_pad, patch_size))
        coords = [(y, x_) for y in ys for x_ in xs]

        # Canvas to write predictions into (same device as x/model)
        pred_full = torch.zeros_like(x)

        # Process patches in batches
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = []
            for (y, x_) in batch_coords:
                patch = x[:, :, y:y + patch_size, x_:x_ + patch_size]  # (1, 4, ps, ps)
                patches.append(patch)
            patches = torch.cat(patches, dim=0)  # (B, 4, ps, ps) on device

            # Mixed precision on CUDA only
            if device.type == "cuda":
                with autocast():
                    out, *_ = model(patches)  # (B, 4, ps, ps)
            else:
                out, *_ = model(patches)

            # Write predictions back into the full canvas
            for b, (y, x_) in enumerate(batch_coords):
                pred_full[:, :, y:y + patch_size, x_:x_ + patch_size] = out[b:b + 1]

        # Crop back to original size
        pred_full = pred_full[:, :, :H_orig, :W_orig]  # (1, 4, H_orig, W_orig)
        return pred_full.squeeze(0).cpu()  # (4, H_orig, W_orig) on CPU


def compute_metrics_bayer_and_lpips(pred_4ch, gt_4ch, lpips_model):
    """
    Compute PSNR/SSIM in Bayer domain and LPIPS on pseudo-RGB.

    Args:
        pred_4ch: (4, H, W) tensor in [0,1], CPU
        gt_4ch:   (4, H, W) tensor in [0,1], CPU
        lpips_model: LPIPS model (possibly on GPU)

    Returns:
        psnr (float), ssim (float), lpips_rgb (float)
    """
    # --- Bayer-domain PSNR / SSIM (CPU / numpy) ---
    pred_np = pred_4ch.permute(1, 2, 0).cpu().numpy()
    gt_np   = gt_4ch.permute(1, 2, 0).cpu().numpy()

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=-1)

    # --- LPIPS on pseudo-RGB (first 3 channels) ---
    pred_rgb = pred_4ch[:3, :, :].unsqueeze(0)  # (1, 3, H, W)
    gt_rgb   = gt_4ch[:3, :, :].unsqueeze(0)    # (1, 3, H, W)

    # Move inputs to same device as lpips_model
    lpips_device = next(lpips_model.parameters()).device
    pred_rgb = pred_rgb.to(lpips_device)
    gt_rgb   = gt_rgb.to(lpips_device)

    # Scale [0,1] → [-1,1]
    pred_rgb_lp = pred_rgb * 2 - 1
    gt_rgb_lp   = gt_rgb   * 2 - 1

    with torch.no_grad():
        lp = lpips_model(pred_rgb_lp, gt_rgb_lp).item()

    return psnr, ssim, lp


def main():
    input_dir = os.path.join(SUBSET_ROOT, SPLIT, "input")
    gt_dir    = os.path.join(SUBSET_ROOT, SPLIT, "GT")

    assert os.path.isdir(input_dir), f"Missing input dir: {input_dir}"
    assert os.path.isdir(gt_dir),    f"Missing GT dir: {gt_dir}"

    input_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    if NUM_EXAMPLES is not None:
        input_files = input_files[:NUM_EXAMPLES]

    print(f"--- [testing_quick] Split: {SPLIT}, Num images: {len(input_files)}")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(DRIVE_RESULTS_ROOT, exist_ok=True)

    # LPIPS model (VGG backbone)
    lpips_model = lpips.LPIPS(net='vgg').to(DEVICE)

    for cfg in MODEL_CONFIGS:
        model_name = cfg["name"]
        model_type = cfg["model_type"]
        weights    = cfg["weights"]

        if not os.path.exists(weights):
            print(f"--- [testing_quick] Skipping {model_name}, weights not found: {weights}")
            continue

        print(f"\n=== Evaluating {model_name} ({model_type}) on subset {SPLIT} ===")

        # Build model
        if model_type == "teacher":
            model = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(DEVICE)
        elif model_type == "student":
            model = UNetStudent(in_channels=4, out_channels=4).to(DEVICE)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.load_state_dict(torch.load(weights, map_location=DEVICE))
        model.eval()

        # Output dirs for this model
        model_result_dir = os.path.join(RESULTS_ROOT, f"{model_name}_{SPLIT}")
        os.makedirs(model_result_dir, exist_ok=True)

        if SAVE_NPY:
            npy_out_dir = os.path.join(model_result_dir, "npy")
            os.makedirs(npy_out_dir, exist_ok=True)
        else:
            npy_out_dir = None

        metrics_csv_path = os.path.join(model_result_dir, "metrics_patch.csv")
        summary_path     = os.path.join(model_result_dir, "metrics_summary.txt")

        # Metrics accumulators
        rows = []
        psnr_list = []
        ssim_list = []
        lpips_list = []

        for inp_path in input_files:
            base = os.path.basename(inp_path)
            gt_path = os.path.join(gt_dir, base)
            if not os.path.exists(gt_path):
                print(f"WARNING: GT not found for {base}, skipping.")
                continue

            # Load full image pair (CPU)
            udc_full, gt_full = load_full_npy_pair(inp_path, gt_path)

            # Run model (internally moves to DEVICE)
            pred_full = run_model_on_full_image(
                model, udc_full, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE
            )  # (4, H, W) CPU

            # Save predicted 4-ch .npy
            if SAVE_NPY and npy_out_dir is not None:
                out_npy_path = os.path.join(npy_out_dir, base)
                np.save(out_npy_path, pred_full.permute(1, 2, 0).numpy())  # (H, W, 4)

            # Metrics
            psnr, ssim, lpv = compute_metrics_bayer_and_lpips(pred_full, gt_full, lpips_model)

            rows.append({
                "filename": base,
                "psnr_bayer": psnr,
                "ssim_bayer": ssim,
                "lpips_rgb": lpv
            })
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpv)

        # Write CSV with per-image metrics
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "psnr_bayer", "ssim_bayer", "lpips_rgb"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        # Summary stats
        mean_psnr  = float(np.mean(psnr_list)) if psnr_list else float("nan")
        mean_ssim  = float(np.mean(ssim_list)) if ssim_list else float("nan")
        mean_lpips = float(np.mean(lpips_list)) if lpips_list else float("nan")

        with open(summary_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Split: {SPLIT}\n")
            f.write(f"Num images: {len(psnr_list)}\n\n")
            f.write(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB\n")
            f.write(f"Mean SSIM (Bayer): {mean_ssim:.4f}\n")
            f.write(f"Mean LPIPS (RGB):  {mean_lpips:.4f}\n")

        print(f"--- [testing_quick] {model_name} results:")
        print(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB")
        print(f"Mean SSIM (Bayer): {mean_ssim:.4f}")
        print(f"Mean LPIPS (RGB):  {mean_lpips:.4f}")
        print(f"Saved CSV to {metrics_csv_path}")
        print(f"Saved summary to {summary_path}")

        # Copy CSV + summary to Drive
        import shutil
        drive_model_dir = os.path.join(DRIVE_RESULTS_ROOT, f"{model_name}_{SPLIT}")
        os.makedirs(drive_model_dir, exist_ok=True)
        shutil.copy(metrics_csv_path, os.path.join(drive_model_dir, "metrics_patch.csv"))
        shutil.copy(summary_path,     os.path.join(drive_model_dir, "metrics_summary.txt"))
        print(f"Copied metrics to Drive: {drive_model_dir}")


if __name__ == "__main__":
    main()
