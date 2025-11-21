# File: testing_quick.py
"""
Quick evaluation on the UDC-SIT subset (data/UDC-SIT_subset).

- Loads full 4-ch .npy images from subset (train/val).
- Tiles into overlapping 256x256 patches, runs model, reconstructs full image.
- Uses Hann window blending to avoid patch seams.
- Computes:
    - PSNR / SSIM in Bayer (4-ch) domain on full images
    - LPIPS on simple RGB (first 3 channels)
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
import argparse
import numpy as np
from glob import glob

import torch
import torch.nn.functional as F
from torch.amp import autocast

import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

# ---------------- PATH SETUP ----------------
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)

from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent

# ---------------- CONFIG DEFAULTS ----------------
SUBSET_ROOT = "data/UDC-SIT_subset"
SPLIT = "val"           # default split; can change if needed
PATCH_SIZE = 256
BATCH_SIZE = 8          # patches per forward pass
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use CPU for LPIPS to avoid device mismatch headaches & save VRAM
LPIPS_DEVICE = "cpu"

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

SAVE_NPY = True
# Set this to e.g. 20 to only run on first 20 images
MAX_IMAGES_DEFAULT = None   # override by CLI or change to e.g. 20
# ---------------------------------------


def load_full_npy_pair(input_path, gt_path):
    """
    Load 4-ch full .npy (H, W, 4) → normalized tensors (4, H, W) in [0,1] on CPU.
    """
    udc = np.load(input_path)  # (H, W, 4)
    gt  = np.load(gt_path)

    assert udc.shape == gt.shape, f"Shape mismatch: {udc.shape} vs {gt.shape}"

    udc = torch.from_numpy(udc).permute(2, 0, 1).float() / 1023.0
    gt  = torch.from_numpy(gt).permute(2, 0, 1).float() / 1023.0
    return udc, gt  # (4, H, W), (4, H, W)


def pad_to_multiple(tensor, patch_size):
    """
    Pad CHW tensor so H,W are multiples of patch_size using reflect padding.
    Returns padded tensor + original H,W.
    """
    _, H, W = tensor.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h == 0 and pad_w == 0:
        return tensor, H, W

    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")  # (C, H+pad_h, W+pad_w)
    return padded, H, W


def run_model_on_full_image(
    model,
    udc_full_4ch,
    patch_size=256,
    batch_size=8,
    overlap=64,
):
    """
    Run model on full 4-ch image by tiling with overlap and Hann blending.

    Args:
        model: PyTorch model (expects (B,4,H,W) in [0,1])
        udc_full_4ch: (4, H, W) tensor in [0,1], on DEVICE
        patch_size: patch side length (e.g., 256)
        batch_size: number of patches per forward pass
        overlap: number of pixels overlapped between neighboring patches

    Returns:
        pred_full_4ch: (4, H, W) tensor in [0,1], on CPU.
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x = udc_full_4ch  # (4, H, W) on device
        x, H_orig, W_orig = pad_to_multiple(x, patch_size)
        C, H_pad, W_pad = x.shape

        x = x.unsqueeze(0)  # (1, C, H_pad, W_pad)

        stride = patch_size - overlap
        if stride <= 0:
            raise ValueError(
                f"overlap must be < patch_size, got overlap={overlap}, patch_size={patch_size}"
            )

        # Build grid of top-left coordinates
        ys = list(range(0, H_pad - patch_size + 1, stride))
        xs = list(range(0, W_pad - patch_size + 1, stride))
        if ys[-1] != H_pad - patch_size:
            ys.append(H_pad - patch_size)
        if xs[-1] != W_pad - patch_size:
            xs.append(W_pad - patch_size)
        coords = [(y, x_) for y in ys for x_ in xs]

        # Accumulators
        pred_accum = torch.zeros(1, C, H_pad, W_pad, device=device)
        weight_acc = torch.zeros(1, 1, H_pad, W_pad, device=device)

        # Hann window for smooth blending
        win_1d = torch.hann_window(patch_size, periodic=False, device=device)
        win_2d = win_1d[:, None] * win_1d[None, :]       # (ps, ps)
        win_2d = win_2d.unsqueeze(0).unsqueeze(0)        # (1,1,ps,ps)

        # Process patches in batches
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = []
            for (y, x_) in batch_coords:
                patch = x[:, :, y:y + patch_size, x_:x_ + patch_size]  # (1, 4, ps, ps)
                patches.append(patch)

            patches = torch.cat(patches, dim=0).to(device)  # (B, 4, ps, ps)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out, *_ = model(patches)  # (B, 4, ps, ps)

            # Apply blending window
            out_win = out * win_2d  # broadcast over channels

            for b, (y, x_) in enumerate(batch_coords):
                pred_accum[:, :, y:y + patch_size, x_:x_ + patch_size] += out_win[b:b + 1]
                weight_acc[:, :, y:y + patch_size, x_:x_ + patch_size] += win_2d

        # Normalize by weights
        weight_acc = torch.clamp(weight_acc, min=1e-6)
        pred_full = pred_accum / weight_acc  # (1, C, H_pad, W_pad)

        # Crop back to original size
        pred_full = pred_full[:, :, :H_orig, :W_orig]
        return pred_full.squeeze(0).cpu()  # (4, H_orig, W_orig) on CPU


def compute_metrics_bayer_and_lpips(pred_4ch_cpu, gt_4ch_cpu, lpips_model):
    """
    pred_4ch_cpu, gt_4ch_cpu: (4, H, W) tensors in [0,1] on CPU.
    Returns: psnr, ssim, lpips_rgb
    """
    # Bayer-domain PSNR/SSIM on CPU
    pred_np = pred_4ch_cpu.permute(1, 2, 0).numpy()
    gt_np   = gt_4ch_cpu.permute(1, 2, 0).numpy()

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=-1)

    # LPIPS on simple RGB (first 3 channels)
    pred_rgb = pred_4ch_cpu[:3, :, :].unsqueeze(0)  # (1, 3, H, W)
    gt_rgb   = gt_4ch_cpu[:3, :, :].unsqueeze(0)

    # Scale to [-1,1] and move to LPIPS_DEVICE
    pred_rgb_lp = (pred_rgb * 2 - 1).to(LPIPS_DEVICE)
    gt_rgb_lp   = (gt_rgb   * 2 - 1).to(LPIPS_DEVICE)

    with torch.no_grad():
        lp = lpips_model(pred_rgb_lp, gt_rgb_lp).item()

    return psnr, ssim, lp


def parse_args():
    parser = argparse.ArgumentParser(description="Quick UDC-SIT subset evaluation.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_IMAGES_DEFAULT,
        help="Max number of images to evaluate (e.g., 20). If None, uses all.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=SPLIT,
        choices=["train", "val"],
        help="Dataset split to evaluate on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_images = args.max_images
    split = args.split

    input_dir = os.path.join(SUBSET_ROOT, split, "input")
    gt_dir    = os.path.join(SUBSET_ROOT, split, "GT")

    assert os.path.isdir(input_dir), f"Missing input dir: {input_dir}"
    assert os.path.isdir(gt_dir),    f"Missing gt dir: {gt_dir}"

    input_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    if max_images is not None:
        input_files = input_files[:max_images]

    print(f"--- [testing_quick] Split: {split}, Num images: {len(input_files)} (max_images={max_images})")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(DRIVE_RESULTS_ROOT, exist_ok=True)

    # LPIPS model on CPU
    lpips_model = lpips.LPIPS(net='vgg').to(LPIPS_DEVICE)
    print(f"[testing_quick] LPIPS running on: {LPIPS_DEVICE}")

    for cfg in MODEL_CONFIGS:
        model_name = cfg["name"]
        model_type = cfg["model_type"]
        weights    = cfg["weights"]

        if not os.path.exists(weights):
            print(f"--- [testing_quick] Skipping {model_name}, weights not found: {weights}")
            continue

        print(f"\n=== Evaluating {model_name} ({model_type}) on subset {split} ===")

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
        model_result_dir = os.path.join(RESULTS_ROOT, f"{model_name}_{split}")
        os.makedirs(model_result_dir, exist_ok=True)

        if SAVE_NPY:
            npy_out_dir = os.path.join(model_result_dir, "npy")
            os.makedirs(npy_out_dir, exist_ok=True)
        else:
            npy_out_dir = None

        metrics_csv_path = os.path.join(model_result_dir, "metrics_full.csv")
        summary_path     = os.path.join(model_result_dir, "metrics_summary.txt")

        rows = []
        psnr_list = []
        ssim_list = []
        lpips_list = []

        for inp_path in tqdm(input_files, desc=f"{model_name} [{split}]"):
            base = os.path.basename(inp_path)
            gt_path = os.path.join(gt_dir, base)
            if not os.path.exists(gt_path):
                print(f"WARNING: GT not found for {base}, skipping.")
                continue

            # Load full image pair on CPU
            udc_full_cpu, gt_full_cpu = load_full_npy_pair(inp_path, gt_path)

            # Move input to DEVICE for model
            udc_full = udc_full_cpu.to(DEVICE)

            # Run model with overlap+blending
            pred_full_cpu = run_model_on_full_image(
                model,
                udc_full,
                patch_size=PATCH_SIZE,
                batch_size=BATCH_SIZE,
                overlap=64,  # you can tune (32 or 64 usually good)
            )

            # Save prediction as 4-ch .npy if requested
            if SAVE_NPY and npy_out_dir is not None:
                out_npy_path = os.path.join(npy_out_dir, base)
                np.save(out_npy_path, pred_full_cpu.permute(1, 2, 0).numpy())  # (H, W, 4)

            # Metrics (pred & gt on CPU)
            psnr, ssim, lpv = compute_metrics_bayer_and_lpips(pred_full_cpu, gt_full_cpu, lpips_model)

            rows.append({
                "filename": base,
                "psnr_bayer": psnr,
                "ssim_bayer": ssim,
                "lpips_rgb": lpv
            })
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpv)

        # Write per-image CSV
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
            f.write(f"Split: {split}\n")
            f.write(f"Num images: {len(psnr_list)}\n\n")
            f.write(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB\n")
            f.write(f"Mean SSIM (Bayer): {mean_ssim:.4f}\n")
            f.write(f"Mean LPIPS (RGB):  {mean_lpips:.4f}\n")

        print(f"\n--- [testing_quick] {model_name} results ({split}):")
        print(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB")
        print(f"Mean SSIM (Bayer): {mean_ssim:.4f}")
        print(f"Mean LPIPS (RGB):  {mean_lpips:.4f}")
        print(f"Saved CSV to {metrics_csv_path}")
        print(f"Saved summary to {summary_path}")

        # Copy CSV + summary to Drive
        import shutil
        drive_model_dir = os.path.join(DRIVE_RESULTS_ROOT, f"{model_name}_{split}")
        os.makedirs(drive_model_dir, exist_ok=True)
        shutil.copy(metrics_csv_path, os.path.join(drive_model_dir, "metrics_full.csv"))
        shutil.copy(summary_path,     os.path.join(drive_model_dir, "metrics_summary.txt"))
        print(f"Copied metrics to Drive: {drive_model_dir}")


if __name__ == "__main__":
    main()
