# File: eval_srgb_udc.py
"""
Unified sRGB evaluation for UDC-SIT teacher & student predictions.

Flow:
- Assumes you have full-resolution 4-ch .npy GT images at:
    <data_root>/<split>/GT/*.npy

- And predicted 4-ch .npy images (from testing_udc.py) at:
    <teacher_pred_dir>/*.npy
    <student_pred_dir>/*.npy
  where each file name matches the GT (e.g., "00001.npy").

- Converts both GT and predictions from 4-ch Bayer-like representation
  to approximate sRGB (H, W, 3) using a simple, consistent mapping:

    R = ch1
    G = 0.5 * (ch0 + ch3)
    B = ch2

- Computes PSNR, SSIM, and LPIPS (VGG) in sRGB domain.

Outputs:
- CSV + summary per model under `results_root`.
- Copies these to `drive_results_root` for backup.
"""

import os
import sys
import csv
import argparse
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import imageio.v2 as imageio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified sRGB evaluation for UDC-SIT teacher/student predictions."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/dataset/UDC-SIT",
        help="Root of UDC-SIT dataset; expects <data-root>/<split>/GT/*.npy",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Split under data-root (e.g., training, validation, testing, train, val, test).",
    )

    # Prediction dirs (4-ch .npy)
    parser.add_argument(
        "--teacher-pred-dir",
        type=str,
        default="results_full/teacher_full_validation/npy",
        help="Directory containing teacher 4-ch predictions (.npy). Leave empty to skip.",
    )
    parser.add_argument(
        "--student-pred-dir",
        type=str,
        default="results_full/student_full_validation/npy",
        help="Directory containing student 4-ch predictions (.npy). Leave empty to skip.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="If set, only evaluate on first N images (sorted by filename).",
    )

    parser.add_argument(
        "--results-root",
        type=str,
        default="results_srgb_metrics",
        help="Local folder to store evaluation CSV + summaries.",
    )
    parser.add_argument(
        "--drive-results-root",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/results_srgb_metrics",
        help="Drive folder where evaluation CSV + summaries are mirrored.",
    )

    return parser.parse_args()


def fourch_to_srgb(arr_4ch):
    """
    Convert (H, W, 4) in [0, 1] to approximate sRGB (H, W, 3) in [0, 1].

    Channel convention (UDC-SIT-style):
        ch0: GR
        ch1: R
        ch2: B
        ch3: GB

    We use a simple, consistent mapping:
        R = ch1
        G = 0.5 * (ch0 + ch3)
        B = ch2
    """
    if arr_4ch.ndim != 3 or arr_4ch.shape[2] != 4:
        raise ValueError(f"Expected (H, W, 4), got {arr_4ch.shape}")

    GR = arr_4ch[:, :, 0]
    R  = arr_4ch[:, :, 1]
    B  = arr_4ch[:, :, 2]
    GB = arr_4ch[:, :, 3]

    G = 0.5 * (GR + GB)
    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def load_4ch_npy(path):
    """
    Load 4-ch .npy (H, W, 4) in [0, 1023] and normalize to [0, 1].
    """
    arr = np.load(path)  # (H, W, 4)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected (H, W, 4) for {path}, got {arr.shape}")
    arr = arr.astype(np.float32) / 1023.0
    return arr


def compute_srgb_metrics(gt_rgb, pred_rgb, lpips_model, device):
    """
    gt_rgb, pred_rgb: (H, W, 3) float32 in [0, 1].
    Returns: psnr, ssim, lpips_val
    """
    # PSNR/SSIM
    psnr = peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=1.0)
    ssim = structural_similarity(gt_rgb, pred_rgb, data_range=1.0, channel_axis=-1)

    # LPIPS
    gt_t   = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    pred_t = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    gt_lp   = gt_t * 2.0 - 1.0
    pred_lp = pred_t * 2.0 - 1.0

    with torch.no_grad():
        lp = lpips_model(pred_lp, gt_lp).item()

    return psnr, ssim, lp


def eval_one_model(
    name,
    pred_dir,
    gt_dir,
    max_images,
    results_root,
    drive_results_root,
    lpips_model,
    device,
):
    """
    Evaluate one model (teacher or student) given prediction directory.
    """
    if not pred_dir or not os.path.isdir(pred_dir):
        print(f"--- [eval_srgb] Skipping {name}, pred_dir missing or empty: {pred_dir}")
        return

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(drive_results_root, exist_ok=True)

    pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
    if max_images is not None:
        pred_files = pred_files[:max_images]

    if len(pred_files) == 0:
        print(f"--- [eval_srgb] No .npy prediction files found in {pred_dir}.")
        return

    model_result_dir = os.path.join(results_root, f"{name}_srgb")
    os.makedirs(model_result_dir, exist_ok=True)

    csv_path    = os.path.join(model_result_dir, "metrics_srgb.csv")
    summary_path = os.path.join(model_result_dir, "metrics_summary.txt")

    rows = []
    psnr_list = []
    ssim_list = []
    lpips_list = []

    print(f"\n=== [eval_srgb] Evaluating {name} ===")
    print(f"GT dir:     {gt_dir}")
    print(f"Pred dir:   {pred_dir}")
    print(f"Num images: {len(pred_files)}")

    pbar = tqdm(pred_files, desc=f"[{name}] sRGB eval")
    for pred_path in pbar:
        fname = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            # try alternative naming or warn
            print(f"WARNING: GT not found for {fname}, skipping.")
            continue

        arr_gt   = load_4ch_npy(gt_path)    # (H, W, 4) in [0,1]
        arr_pred = load_4ch_npy(pred_path)  # (H, W, 4) in [0,1]

        if arr_gt.shape != arr_pred.shape:
            print(f"WARNING: shape mismatch for {fname}: GT={arr_gt.shape}, pred={arr_pred.shape}, skipping.")
            continue

        gt_rgb   = fourch_to_srgb(arr_gt)   # (H, W, 3)
        pred_rgb = fourch_to_srgb(arr_pred)

        psnr, ssim, lp = compute_srgb_metrics(gt_rgb, pred_rgb, lpips_model, device)

        rows.append({
            "filename": fname,
            "psnr_srgb": psnr,
            "ssim_srgb": ssim,
            "lpips_srgb": lp,
        })
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lp)

        pbar.set_postfix({"PSNR": f"{psnr:.2f}", "SSIM": f"{ssim:.3f}"})

    if len(psnr_list) == 0:
        print(f"--- [eval_srgb] No valid pairs evaluated for {name}.")
        return

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr_srgb", "ssim_srgb", "lpips_srgb"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summary
    mean_psnr  = float(np.mean(psnr_list))
    mean_ssim  = float(np.mean(ssim_list))
    mean_lpips = float(np.mean(lpips_list))

    with open(summary_path, "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Num images: {len(psnr_list)}\n\n")
        f.write(f"Mean PSNR (sRGB): {mean_psnr:.4f} dB\n")
        f.write(f"Mean SSIM (sRGB): {mean_ssim:.4f}\n")
        f.write(f"Mean LPIPS:       {mean_lpips:.44f}\n")

    print(f"\n--- [eval_srgb] {name} results:")
    print(f"Mean PSNR (sRGB): {mean_psnr:.4f} dB")
    print(f"Mean SSIM (sRGB): {mean_ssim:.4f}")
    print(f"Mean LPIPS:       {mean_lpips:.4f}")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved summary to {summary_path}")

    # Copy metrics to Drive
    try:
        import shutil
        drive_model_dir = os.path.join(drive_results_root, f"{name}_srgb")
        os.makedirs(drive_model_dir, exist_ok=True)
        shutil.copy(csv_path,    os.path.join(drive_model_dir, "metrics_srgb.csv"))
        shutil.copy(summary_path, os.path.join(drive_model_dir, "metrics_summary.txt"))
        print(f"Copied sRGB metrics to Drive: {drive_model_dir}")
    except Exception as e:
        print(f"Could not copy metrics to Drive for {name}: {e}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n--- [eval_srgb] Configuration ---")
    print(f"Data root:          {args.data_root}")
    print(f"Split:              {args.split}")
    print(f"Teacher pred dir:   {args.teacher_pred_dir}")
    print(f"Student pred dir:   {args.student_pred_dir}")
    print(f"Max images:         {args.max_images}")
    print(f"Results root:       {args.results_root}")
    print(f"Drive results root: {args.drive_results_root}")
    print(f"Device:             {device}")
    print("-----------------------------------\n")

    gt_dir = os.path.join(args.data_root, args.split, "GT")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")

    # LPIPS model
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    if args.teacher_pred_dir:
        eval_one_model(
            name="teacher_full",
            pred_dir=args.teacher_pred_dir,
            gt_dir=gt_dir,
            max_images=args.max_images,
            results_root=args.results_root,
            drive_results_root=args.drive_results_root,
            lpips_model=lpips_model,
            device=device,
        )

    if args.student_pred_dir:
        eval_one_model(
            name="student_full",
            pred_dir=args.student_pred_dir,
            gt_dir=gt_dir,
            max_images=args.max_images,
            results_root=args.results_root,
            drive_results_root=args.drive_results_root,
            lpips_model=lpips_model,
            device=device,
        )


if __name__ == "__main__":
    main()
