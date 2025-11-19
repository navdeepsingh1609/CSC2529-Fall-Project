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
        train/gt/*.npy
        val/input/*.npy
        val/gt/*.npy
- Each .npy is (H, W, 4) with values in [0, 1023].
"""

import os
import sys
import csv
import numpy as np
from glob import glob

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

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
    """Load 4-ch full .npy (H, W, 4) → normalized torch tensors (4, H, W) in [0,1]."""
    udc = np.load(input_path)  # (H, W, 4)
    gt  = np.load(gt_path)

    assert udc.shape == gt.shape, f"Shape mismatch: {udc.shape} vs {gt.shape}"

    udc = torch.from_numpy(udc).permute(2, 0, 1).float() / 1023.0
    gt  = torch.from_numpy(gt).permute(2, 0, 1).float() / 1023.0
    return udc, gt  # (4, H, W), (4, H, W)


def pad_to_multiple(tensor, patch_size):
    """Pad CHW tensor so H,W are multiples of patch_size using reflect padding."""
    _, H, W = tensor.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h == 0 and pad_w == 0:
        return tensor, H, W

    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")  # (C, H+pad_h, W+pad_w)
    return padded, H, W


def run_model_on_full_image(model, udc_full_4ch, patch_size, batch_size):
    """
    Run model on full 4-ch image by tiling into non-overlapping patches.
    udc_full_4ch: (4, H, W) tensor in [0,1].
    Returns: pred_full_4ch: (4, H, W) tensor in [0,1].
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x, H_orig, W_orig = pad_to_multiple(udc_full_4ch, patch_size)
        C, H_pad, W_pad = x.shape

        x = x.unsqueeze(0)  # (1, C, H_pad, W_pad) to keep things simple

        # Prepare patch coordinates
        ys = list(range(0, H_pad, patch_size))
        xs = list(range(0, W_pad, patch_size))
        coords = [(y, x_) for y in ys for x_ in xs]

        pred_full = torch.zeros_like(x)

        # Process patches in batches
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = []
            for (y, x_) in batch_coords:
                patch = x[:, :, y:y + patch_size, x_:x_ + patch_size]  # (1, 4, ps, ps)
                patches.append(patch)
            patches = torch.cat(patches, dim=0).to(device)  # (B, 4, ps, ps)

            with autocast():
                out, *_ = model(patches)  # (B, 4, ps, ps)

            out = out.detach().cpu()

            # Write back into full canvas
            for b, (y, x_) in enumerate(batch_coords):
                pred_full[:, :, y:y + patch_size, x_:x_ + patch_size] = out[b:b + 1]

        pred_full = pred_full[:, :, :H_orig, :W_orig]  # crop
        return pred_full.squeeze(0)  # (4, H_orig, W_orig)


def compute_metrics_bayer_and_lpips(pred_4ch, gt_4ch, lpips_model):
    """
    pred_4ch, gt_4ch: (4, H, W) tensors in [0,1].
    Returns: psnr, ssim, lpips_rgb
    """
    # Bayer-domain PSNR/SSIM
    pred_np = pred_4ch.permute(1, 2, 0).cpu().numpy()
    gt_np   = gt_4ch.permute(1, 2, 0).cpu().numpy()

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=-1)

    # LPIPS on pseudo-RGB (first 3 channels)
    pred_rgb = pred_4ch[:3, :, :].unsqueeze(0)  # (1, 3, H, W)
    gt_rgb   = gt_4ch[:3, :, :].unsqueeze(0)
    # Scale to [-1,1] for LPIPS
    pred_rgb_lp = pred_rgb * 2 - 1
    gt_rgb_lp   = gt_rgb   * 2 - 1

    with torch.no_grad():
        lp = lpips_model(pred_rgb_lp, gt_rgb_lp).item()

    return psnr, ssim, lp


def main():
    input_dir = os.path.join(SUBSET_ROOT, SPLIT, "input")
    gt_dir    = os.path.join(SUBSET_ROOT, SPLIT, "gt")

    assert os.path.isdir(input_dir), f"Missing input dir: {input_dir}"
    assert os.path.isdir(gt_dir),    f"Missing gt dir: {gt_dir}"

    input_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    if NUM_EXAMPLES is not None:
        input_files = input_files[:NUM_EXAMPLES]

    print(f"--- [testing_quick] Split: {SPLIT}, Num images: {len(input_files)}")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(DRIVE_RESULTS_ROOT, exist_ok=True)

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

        # Output dirs
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

            udc_full, gt_full = load_full_npy_pair(inp_path, gt_path)
            udc_full = udc_full.to(DEVICE)

            # Run model
            pred_full = run_model_on_full_image(
                model, udc_full, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE
            ).cpu()

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

        # Write CSV
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "psnr_bayer", "ssim_bayer", "lpips_rgb"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        # Summary
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
        drive_model_dir = os.path.join(DRIVE_RESULTS_ROOT, f"{model_name}_{SPLIT}")
        os.makedirs(drive_model_dir, exist_ok=True)
        import shutil
        shutil.copy(metrics_csv_path, os.path.join(drive_model_dir, "metrics_patch.csv"))
        shutil.copy(summary_path,     os.path.join(drive_model_dir, "metrics_summary.txt"))
        print(f"Copied metrics to Drive: {drive_model_dir}")


if __name__ == "__main__":
    main()
