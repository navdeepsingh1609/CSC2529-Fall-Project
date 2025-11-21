# eval_srgb_udc.py
"""
Evaluate UDC-SIT predictions in sRGB space (PSNR / SSIM / LPIPS).

- Loads 4-ch .npy (H, W, 4) for input, GT, and model predictions.
- Normalizes to [0,1] (handles both [0,1023] and [0,1]).
- Uses a consistent 4ch -> sRGB mapping for *all* images:
    * lightweight ISP (channel mixing + optional WB + gamma)
- Computes PSNR / SSIM / LPIPS on sRGB images.
- Writes CSV + summary for:
    - input_baseline (input vs GT)
    - teacher_full   (teacher preds vs GT)
    - student_full   (student preds vs GT)
- Mirrors results to Google Drive if path is provided.
"""

import os
import sys
import argparse
import csv
from glob import glob

import numpy as np
from tqdm import tqdm

import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_and_normalize_4ch(path):
    """
    Load a 4-channel .npy file and normalize to [0,1].

    Handles both raw [0,1023] and already-normalized [0,1] inputs.
    Returns array of shape (H, W, 4), float32 in [0,1].
    """
    arr = np.load(path).astype(np.float32)

    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected (H, W, 4) array, got {arr.shape} for {path}")

    maxv = float(arr.max())
    if maxv == 0.0:
        return arr  # all zeros
    if maxv > 2.0:
        # Assume 10-bit raw [0,1023]
        arr = arr / 1023.0
    else:
        # Assume already in [0,1], just clip
        arr = np.clip(arr, 0.0, 1.0)
    return arr


def fourch_to_srgb(arr_4ch, wb_mode="channel_gains", gamma=2.2):
    """
    Convert 4-channel Bayer-like array (H, W, 4) in [0,1] to approximate sRGB (H, W, 3) in [0,1].

    This is a lightweight, differentiable-free ISP approximation:
    - Channel mixing to mimic demosaicing + white balance
    - Optional simple white balance (per-channel gains)
    - Global gamma correction

    wb_mode:
        - "none"           : no extra white balance
        - "channel_gains"  : fixed gains [1.2, 1.0, 1.5]
    gamma:
        - gamma value for output gamma correction (e.g., 2.2). If 1.0, no gamma applied.
    """
    if arr_4ch.ndim != 3 or arr_4ch.shape[2] != 4:
        raise ValueError(f"fourch_to_srgb expects (H, W, 4), got {arr_4ch.shape}")

    # Split channels
    c0 = arr_4ch[:, :, 0]
    c1 = arr_4ch[:, :, 1]
    c2 = arr_4ch[:, :, 2]
    c3 = arr_4ch[:, :, 3]

    # Heuristic mixing: treat c1 as R, c2 as G, c3 as B, c0 as extra brightness.
    R_lin = c1 + 0.1 * c0
    G_lin = c2 + 0.1 * c0
    B_lin = c3 + 0.1 * c0

    rgb_lin = np.stack([R_lin, G_lin, B_lin], axis=-1)

    # Simple white balance in linear domain
    if wb_mode == "channel_gains":
        gains = np.array([1.2, 1.0, 1.5], dtype=np.float32).reshape(1, 1, 3)
        rgb_lin = rgb_lin * gains
    # elif wb_mode == "none": do nothing
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)

    # Gamma correction to approximate display sRGB
    if gamma is not None and gamma != 1.0:
        rgb_srgb = np.power(rgb_lin, 1.0 / gamma)
    else:
        rgb_srgb = rgb_lin

    rgb_srgb = np.clip(rgb_srgb, 0.0, 1.0)
    return rgb_srgb.astype(np.float32)


def compute_srgb_metrics(pred_srgb, gt_srgb, lpips_model, device):
    """
    pred_srgb, gt_srgb: (H, W, 3) float32 np arrays in [0,1].
    Returns: psnr, ssim, lpips_val.
    """
    pred = np.clip(pred_srgb, 0.0, 1.0)
    gt   = np.clip(gt_srgb,   0.0, 1.0)

    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(gt, pred, data_range=1.0, channel_axis=-1)

    # LPIPS expects tensors in [-1,1]
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    gt_t   = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

    with torch.no_grad():
        lp = lpips_model(pred_t, gt_t).item()

    return psnr, ssim, lp


def evaluate_model(
    name,
    pred_dir,
    data_root,
    split,
    lpips_model,
    wb_mode,
    gamma,
    max_images,
    results_root,
    drive_results_root,
    device,
):
    """
    Evaluate one model (or baseline) in sRGB space.

    name:         label used for results folder (e.g., "teacher_full", "student_full", "input_baseline")
    pred_dir:     directory containing 4-ch .npy predictions (H, W, 4), or None for baseline input
    data_root:    root of UDC-SIT (contains {split}/input and {split}/GT)
    split:        "validation" or "testing"
    """
    gt_dir  = os.path.join(data_root, split, "GT")
    inp_dir = os.path.join(data_root, split, "input")

    if not os.path.isdir(gt_dir):
        print(f"[eval_srgb_udc] ERROR: GT directory not found: {gt_dir}")
        return

    gt_files = sorted(glob(os.path.join(gt_dir, "*.npy")))
    if max_images is not None:
        gt_files = gt_files[:max_images]

    if len(gt_files) == 0:
        print(f"[eval_srgb_udc] No GT .npy files found in {gt_dir}")
        return

    results_dir = os.path.join(results_root, f"{name}_{split}")
    os.makedirs(results_dir, exist_ok=True)

    if drive_results_root is not None and drive_results_root != "":
        drive_results_dir = os.path.join(drive_results_root, f"{name}_{split}")
        os.makedirs(drive_results_dir, exist_ok=True)
    else:
        drive_results_dir = None

    csv_path     = os.path.join(results_dir, "metrics_srgb.csv")
    summary_path = os.path.join(results_dir, "metrics_summary.txt")

    rows = []
    psnr_list = []
    ssim_list = []
    lpips_list = []

    print(f"\n--- [eval_srgb_udc] Evaluating '{name}' on split '{split}' ---")
    print(f"GT dir:    {gt_dir}")
    if pred_dir is not None:
        print(f"Pred dir:  {pred_dir}")
    else:
        print("Pred dir:  None (using input as baseline)")

    for gt_path in tqdm(gt_files, desc=f"[{name}] images", ncols=80):
        base = os.path.basename(gt_path)

        # Corresponding input / pred paths
        inp_path = os.path.join(inp_dir, base)
        if pred_dir is not None:
            pred_path = os.path.join(pred_dir, base)
        else:
            pred_path = inp_path  # baseline: input vs GT

        if not os.path.exists(inp_path):
            print(f"  [WARN] Missing input for {base}, skipping.")
            continue
        if not os.path.exists(pred_path):
            print(f"  [WARN] Missing prediction for {base} in {pred_dir}, skipping.")
            continue

        # Load 4-ch
        gt_4ch   = load_and_normalize_4ch(gt_path)
        pred_4ch = load_and_normalize_4ch(pred_path)

        # Convert to sRGB
        gt_srgb   = fourch_to_srgb(gt_4ch,   wb_mode=wb_mode, gamma=gamma)
        pred_srgb = fourch_to_srgb(pred_4ch, wb_mode=wb_mode, gamma=gamma)

        # Metrics
        psnr, ssim, lp = compute_srgb_metrics(pred_srgb, gt_srgb, lpips_model, device)
        rows.append(
            {
                "filename": base,
                "psnr_srgb": psnr,
                "ssim_srgb": ssim,
                "lpips_srgb": lp,
            }
        )
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lp)

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "psnr_srgb", "ssim_srgb", "lpips_srgb"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summary
    mean_psnr  = float(np.mean(psnr_list)) if psnr_list else float("nan")
    mean_ssim  = float(np.mean(ssim_list)) if ssim_list else float("nan")
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else float("nan")

    with open(summary_path, "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Num images: {len(psnr_list)}\n\n")
        f.write(f"Mean PSNR (sRGB): {mean_psnr:.4f} dB\n")
        f.write(f"Mean SSIM (sRGB): {mean_ssim:.4f}\n")
        f.write(f"Mean LPIPS (sRGB): {mean_lpips:.4f}\n")
        f.write(f"WB mode: {wb_mode}\n")
        f.write(f"Gamma:   {gamma}\n")

    print(f"\n--- [eval_srgb_udc] {name} results on '{split}':")
    print(f"Mean PSNR (sRGB): {mean_psnr:.4f} dB")
    print(f"Mean SSIM (sRGB): {mean_ssim:.4f}")
    print(f"Mean LPIPS (sRGB): {mean_lpips:.4f}")
    print(f"Saved CSV to     : {csv_path}")
    print(f"Saved summary to : {summary_path}")

    # Copy to Drive if requested
    if drive_results_dir is not None:
        import shutil

        shutil.copy(csv_path, os.path.join(drive_results_dir, "metrics_srgb.csv"))
        shutil.copy(summary_path, os.path.join(drive_results_dir, "metrics_summary.txt"))
        print(f"Copied metrics to Drive: {drive_results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate UDC-SIT predictions in sRGB space (PSNR/SSIM/LPIPS)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/dataset/UDC-SIT/UDC-SIT",
        help="Root of UDC-SIT full dataset (contains e.g. validation/GT, validation/input).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "testing"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--teacher-pred-dir",
        type=str,
        default="results_full/teacher_full_validation/npy",
        help="Directory with teacher 4-ch .npy predictions.",
    )
    parser.add_argument(
        "--student-pred-dir",
        type=str,
        default="results_full/student_full_validation/npy",
        help="Directory with student 4-ch .npy predictions.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max number of images to evaluate (for quick testing).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results_srgb_metrics",
        help="Where to store sRGB metrics CSVs + summaries.",
    )
    parser.add_argument(
        "--drive-results-root",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/results_srgb_metrics",
        help="Google Drive folder for sRGB metrics (will mirror results_root).",
    )
    parser.add_argument(
        "--wb-mode",
        type=str,
        default="channel_gains",
        choices=["none", "channel_gains"],
        help="White-balance mode applied in fourch_to_srgb.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="Gamma for fourch_to_srgb (1.0 = no gamma correction).",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== [eval_srgb_udc] Config ===")
    print(f"Data root:      {args.data_root}")
    print(f"Split:          {args.split}")
    print(f"Teacher preds:  {args.teacher_pred_dir}")
    print(f"Student preds:  {args.student_pred_dir}")
    print(f"Max images:     {args.max_images}")
    print(f"Results root:   {args.results_root}")
    print(f"Drive results:  {args.drive_results_root}")
    print(f"WB mode:        {args.wb_mode}")
    print(f"Gamma:          {args.gamma}")
    print(f"Device:         {device}")
    print("=================================")

    os.makedirs(args.results_root, exist_ok=True)
    if args.drive_results_root is not None and args.drive_results_root != "":
        os.makedirs(args.drive_results_root, exist_ok=True)

    # LPIPS model
    lpips_model = lpips.LPIPS(net="vgg").to(device)

    # Baseline: input vs GT
    evaluate_model(
        name="input_baseline",
        pred_dir=None,
        data_root=args.data_root,
        split=args.split,
        lpips_model=lpips_model,
        wb_mode=args.wb_mode,
        gamma=args.gamma,
        max_images=args.max_images,
        results_root=args.results_root,
        drive_results_root=args.drive_results_root,
        device=device,
    )

    # Teacher
    if args.teacher_pred_dir is not None and args.teacher_pred_dir != "":
        evaluate_model(
            name="teacher_full",
            pred_dir=args.teacher_pred_dir,
            data_root=args.data_root,
            split=args.split,
            lpips_model=lpips_model,
            wb_mode=args.wb_mode,
            gamma=args.gamma,
            max_images=args.max_images,
            results_root=args.results_root,
            drive_results_root=args.drive_results_root,
            device=device,
        )

    # Student
    if args.student_pred_dir is not None and args.student_pred_dir != "":
        evaluate_model(
            name="student_full",
            pred_dir=args.student_pred_dir,
            data_root=args.data_root,
            split=args.split,
            lpips_model=lpips_model,
            wb_mode=args.wb_mode,
            gamma=args.gamma,
            max_images=args.max_images,
            results_root=args.results_root,
            drive_results_root=args.drive_results_root,
            device=device,
        )


if __name__ == "__main__":
    main()
