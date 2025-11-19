# File: eval_full_srgb.py
"""
Evaluate paper-ready PSNR/SSIM/LPIPS in sRGB space.

Requirements:
- You have already converted:
    GT  4-ch .npy -> sRGB PNGs (using visualize_sit.py)
    Pred 4-ch .npy -> sRGB PNGs (same script)
- Filenames of PNGs in GT and Pred dirs match (e.g., 0001.png, 0002.png, ...).

This script:
- Reads matching PNG pairs.
- Computes PSNR, SSIM (sRGB) and LPIPS (VGG) for each image.
- Saves a CSV of per-image metrics and a text summary.
- Copies metrics to Google Drive.

Adjust GT_SRGB_DIR, PRED_SRGB_DIR, MODEL_NAME, SPLIT as needed.
"""

import os
import csv
import numpy as np
from glob import glob

import torch
import lpips
import imageio.v2 as imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ---------------- CONFIG ----------------
# Example paths – change to your actual sRGB directory paths
GT_SRGB_DIR   = "/content/dataset/UDC-SIT_srgb/testing/gt"
PRED_SRGB_DIR = "results_full_srgb_images/student_testing"

MODEL_NAME = "student_full"
SPLIT = "testing"

RESULTS_ROOT = "results_full_srgb"
DRIVE_RESULTS_ROOT = "/content/drive/MyDrive/Computational Imaging Project/results_full_srgb"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------


def load_image_01(path):
    """
    Load PNG and return float32 array in [0,1], shape (H, W, 3).
    """
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:  # RGBA -> RGB
        img = img[:, :, :3]

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
    return img


def main():
    assert os.path.isdir(GT_SRGB_DIR),   f"Missing GT sRGB dir: {GT_SRGB_DIR}"
    assert os.path.isdir(PRED_SRGB_DIR), f"Missing Pred sRGB dir: {PRED_SRGB_DIR}"

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(DRIVE_RESULTS_ROOT, exist_ok=True)

    # We'll index images by the filenames in GT dir
    gt_files = sorted(glob(os.path.join(GT_SRGB_DIR, "*.png")))
    if not gt_files:
        raise RuntimeError(f"No PNGs found in GT dir: {GT_SRGB_DIR}")

    lpips_model = lpips.LPIPS(net='vgg').to(DEVICE)

    rows = []
    psnr_list = []
    ssim_list = []
    lpips_list = []

    for gt_path in gt_files:
        base = os.path.basename(gt_path)
        pred_path = os.path.join(PRED_SRGB_DIR, base)
        if not os.path.exists(pred_path):
            print(f"WARNING: Pred PNG not found for {base}, skipping.")
            continue

        gt_img   = load_image_01(gt_path)    # (H, W, 3)
        pred_img = load_image_01(pred_path)  # (H, W, 3)
        if gt_img.shape != pred_img.shape:
            print(f"WARNING: Shape mismatch for {base}, skipping.")
            continue

        # PSNR & SSIM in sRGB [0,1]
        psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=1.0)
        ssim = structural_similarity(gt_img, pred_img, data_range=1.0, channel_axis=-1)

        # LPIPS (VGG) in [-1,1]
        gt_t   = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)   # (1,3,H,W)
        pred_t = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        gt_lp   = gt_t * 2 - 1
        pred_lp = pred_t * 2 - 1

        with torch.no_grad():
            lpv = lpips_model(pred_lp, gt_lp).item()

        rows.append({
            "filename": base,
            "psnr_srgb": psnr,
            "ssim_srgb": ssim,
            "lpips_srgb": lpv,
        })
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpv)

    if not rows:
        raise RuntimeError("No valid GT/Pred pairs found; check your directories and filenames.")

    metrics_csv_path = os.path.join(
        RESULTS_ROOT, f"metrics_srgb_{MODEL_NAME}_{SPLIT}.csv"
    )
    summary_path = os.path.join(
        RESULTS_ROOT, f"metrics_srgb_{MODEL_NAME}_{SPLIT}_summary.txt"
    )

    # Write CSV
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr_srgb", "ssim_srgb", "lpips_srgb"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    mean_psnr  = float(np.mean(psnr_list))
    mean_ssim  = float(np.mean(ssim_list))
    mean_lpips = float(np.mean(lpips_list))

    with open(summary_path, "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Split: {SPLIT}\n")
        f.write(f"Num images: {len(psnr_list)}\n\n")
        f.write(f"Mean PSNR (sRGB): {mean_psnr:.4f} dB\n")
        f.write(f"Mean SSIM (sRGB): {mean_ssim:.4f}\n")
        f.write(f"Mean LPIPS (sRGB): {mean_lpips:.4f}\n")

    print("=== sRGB Evaluation Results ===")
    print(f"Model: {MODEL_NAME}, Split: {SPLIT}")
    print(f"Num images: {len(psnr_list)}")
    print(f"Mean PSNR (sRGB): {mean_psnr:.4f} dB")
    print(f"Mean SSIM (sRGB): {mean_ssim:.4f}")
    print(f"Mean LPIPS (sRGB): {mean_lpips:.4f}")
    print(f"Saved CSV to {metrics_csv_path}")
    print(f"Saved summary to {summary_path}")

    # Copy metrics to Drive
    import shutil
    shutil.copy(metrics_csv_path,
                os.path.join(DRIVE_RESULTS_ROOT,
                             f"metrics_srgb_{MODEL_NAME}_{SPLIT}.csv"))
    shutil.copy(summary_path,
                os.path.join(DRIVE_RESULTS_ROOT,
                             f"metrics_srgb_{MODEL_NAME}_{SPLIT}_summary.txt"))
    print(f"Copied sRGB metrics to Drive: {DRIVE_RESULTS_ROOT}")


if __name__ == "__main__":
    main()
