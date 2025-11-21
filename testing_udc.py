# File: testing_udc.py
"""
Unified testing script for UDC-SIT teacher & student models on full 4-ch .npy images.

- Works for both subset and full dataset by changing:
    * --data-root (e.g., data/UDC-SIT_subset or /content/dataset/UDC-SIT)
    * --split (train / validation / testing / val / test)
    * --max-images (for quick runs)

- For each model (teacher / student) whose weights path is given:
    * Loads full 4-ch .npy images
    * Normalizes to [0,1], tiles into PATCH_SIZE x PATCH_SIZE patches
    * Runs the model and stitches patches back into full image
    * Computes Bayer-domain PSNR/SSIM and RGB LPIPS
    * Saves predicted 4-ch images as .npy (optional)
    * Saves CSV + summary, and mirrors metrics into Google Drive.
"""

import os
import sys
import csv
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Ensure external MambaIR path is available
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)

from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified RAW (4-ch) testing for teacher/student on UDC-SIT."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/dataset/UDC-SIT",
        help="Root of UDC-SIT data. Expects <data-root>/<split>/input and GT.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Split name under data-root (e.g., training, validation, testing, train, val, test).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size used when tiling full images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of patches per forward pass.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="If set, limit the number of images evaluated (for quick tests).",
    )

    parser.add_argument(
        "--teacher-weights",
        type=str,
        default="",
        help="Path to teacher checkpoint (.pth). If empty, teacher is skipped.",
    )
    parser.add_argument(
        "--student-weights",
        type=str,
        default="",
        help="Path to student checkpoint (.pth). If empty, student is skipped.",
    )

    parser.add_argument(
        "--results-root",
        type=str,
        default="results_full",
        help="Local root directory where metrics and npy predictions are stored.",
    )
    parser.add_argument(
        "--drive-results-root",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/results_full",
        help="Drive root where metrics are copied (npy not copied by default).",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="If set, save predicted 4-ch full images as .npy under results_root.",
    )

    return parser.parse_args()


def load_full_npy_pair(input_path, gt_path):
    """
    Load 4-ch full .npy (H, W, 4) -> normalized torch tensors (4, H, W) in [0,1].
    """
    udc = np.load(input_path)  # (H, W, 4)
    gt  = np.load(gt_path)
    if udc.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {udc.shape} vs {gt.shape} for {input_path}")

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

    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, H, W


def run_model_on_full_image(model, udc_full_4ch, patch_size, batch_size, device):
    """
    Run model on full 4-ch image by tiling into non-overlapping patches.
    udc_full_4ch: (4, H, W) tensor in [0,1] (on CPU or device).
    Returns: pred_full_4ch: (4, H, W) tensor in [0,1] on CPU.
    """
    model.eval()
    with torch.no_grad():
        x, H_orig, W_orig = pad_to_multiple(udc_full_4ch, patch_size)
        C, H_pad, W_pad = x.shape

        x = x.unsqueeze(0)  # (1, C, H_pad, W_pad)

        ys = list(range(0, H_pad, patch_size))
        xs = list(range(0, W_pad, patch_size))
        coords = [(y, x_) for y in ys for x_ in xs]

        pred_full = torch.zeros_like(x)

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

            for b, (y, x_) in enumerate(batch_coords):
                pred_full[:, :, y:y + patch_size, x_:x_ + patch_size] = out[b:b+1]

        pred_full = pred_full[:, :, :H_orig, :W_orig]  # crop
        return pred_full.squeeze(0)  # (4, H_orig, W_orig)


def compute_metrics_bayer_and_lpips(pred_4ch, gt_4ch, lpips_model, device):
    """
    pred_4ch, gt_4ch: (4, H, W) tensors in [0,1] on CPU.
    Returns: psnr, ssim, lpips_rgb
    """
    # Bayer PSNR/SSIM
    pred_np = pred_4ch.permute(1, 2, 0).numpy()
    gt_np   = gt_4ch.permute(1, 2, 0).numpy()

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=-1)

    # LPIPS on simple RGB (first 3 channels)
    pred_rgb = pred_4ch[:3, :, :].unsqueeze(0).to(device)  # (1, 3, H, W)
    gt_rgb   = gt_4ch[:3, :, :].unsqueeze(0).to(device)

    pred_rgb_lp = pred_rgb * 2.0 - 1.0
    gt_rgb_lp   = gt_rgb   * 2.0 - 1.0

    with torch.no_grad():
        lp = lpips_model(pred_rgb_lp, gt_rgb_lp).item()

    return psnr, ssim, lp


def evaluate_model_on_split(
    model_name,
    model_type,
    weights_path,
    args,
    device,
):
    """Evaluate a single model (teacher or student) on given split."""
    if not weights_path or not os.path.exists(weights_path):
        print(f"--- [testing_udc] Skipping {model_name}, weights not found or empty: {weights_path}")
        return

    # Data dirs
    input_dir = os.path.join(args.data_root, args.split, "input")
    gt_dir    = os.path.join(args.data_root, args.split, "GT")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")

    input_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    if args.max_images is not None:
        input_files = input_files[:args.max_images]

    print(f"\n=== [testing_udc] Evaluating {model_name} ({model_type}) on split '{args.split}' ===")
    print(f"Data root: {args.data_root}")
    print(f"Num images: {len(input_files)}")

    # Load model
    if model_type == "teacher":
        model = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(device)
    elif model_type == "student":
        model = UNetStudent(in_channels=4, out_channels=4).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # LPIPS
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    # Result dirs
    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(args.drive_results_root, exist_ok=True)
    model_result_dir = os.path.join(args.results_root, f"{model_name}_{args.split}")
    os.makedirs(model_result_dir, exist_ok=True)

    if args.save_npy:
        npy_out_dir = os.path.join(model_result_dir, "npy")
        os.makedirs(npy_out_dir, exist_ok=True)
    else:
        npy_out_dir = None

    metrics_csv_path = os.path.join(model_result_dir, "metrics_raw.csv")
    summary_path     = os.path.join(model_result_dir, "metrics_summary.txt")

    rows = []
    psnr_list = []
    ssim_list = []
    lpips_list = []

    pbar = tqdm(input_files, desc=f"[{model_name}] images")
    for inp_path in pbar:
        base = os.path.basename(inp_path)
        gt_path = os.path.join(gt_dir, base)
        if not os.path.exists(gt_path):
            print(f"WARNING: GT not found for {base}, skipping.")
            continue

        # Load full image pair
        udc_full, gt_full = load_full_npy_pair(inp_path, gt_path)
        udc_full = udc_full.to(device)
        gt_full  = gt_full  # keep on CPU for metrics to avoid extra copies

        # Model inference
        pred_full = run_model_on_full_image(
            model, udc_full, patch_size=args.patch_size,
            batch_size=args.batch_size, device=device
        ).cpu()  # (4, H, W) on CPU

        if args.save_npy and npy_out_dir is not None:
            out_npy_path = os.path.join(npy_out_dir, base)
            np.save(out_npy_path, pred_full.permute(1, 2, 0).numpy())  # (H, W, 4)

        # Metrics
        psnr, ssim, lpv = compute_metrics_bayer_and_lpips(
            pred_full, gt_full, lpips_model, device
        )

        rows.append({
            "filename": base,
            "psnr_bayer": psnr,
            "ssim_bayer": ssim,
            "lpips_rgb": lpv,
        })
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpv)

        pbar.set_postfix({"PSNR": f"{psnr:.2f}", "SSIM": f"{ssim:.3f}"})

    # Write CSV
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr_bayer", "ssim_bayer", "lpips_rgb"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summary
    mean_psnr  = float(np.mean(psnr_list))  if psnr_list else float("nan")
    mean_ssim  = float(np.mean(ssim_list))  if ssim_list else float("nan")
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else float("nan")

    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Num images: {len(psnr_list)}\n\n")
        f.write(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB\n")
        f.write(f"Mean SSIM (Bayer): {mean_ssim:.4f}\n")
        f.write(f"Mean LPIPS (RGB):  {mean_lpips:.4f}\n")

    print(f"\n--- [testing_udc] {model_name} results on '{args.split}':")
    print(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB")
    print(f"Mean SSIM (Bayer): {mean_ssim:.4f}")
    print(f"Mean LPIPS (RGB):  {mean_lpips:.4f}")
    print(f"Saved CSV to {metrics_csv_path}")
    print(f"Saved summary to {summary_path}")

    # Copy CSV + summary to Drive
    try:
        import shutil
        drive_model_dir = os.path.join(args.drive_results_root, f"{model_name}_{args.split}")
        os.makedirs(drive_model_dir, exist_ok=True)
        shutil.copy(metrics_csv_path, os.path.join(drive_model_dir, "metrics_raw.csv"))
        shutil.copy(summary_path,     os.path.join(drive_model_dir, "metrics_summary.txt"))
        print(f"Copied metrics to Drive: {drive_model_dir}")
    except Exception as e:
        print(f"Could not copy metrics to Drive: {e}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n--- [testing_udc] Configuration ---")
    print(f"Data root:      {args.data_root}")
    print(f"Split:          {args.split}")
    print(f"Patch size:     {args.patch_size}")
    print(f"Patch batch:    {args.batch_size}")
    print(f"Max images:     {args.max_images}")
    print(f"Teacher ckpt:   {args.teacher_weights}")
    print(f"Student ckpt:   {args.student_weights}")
    print(f"Results root:   {args.results_root}")
    print(f"Drive results:  {args.drive_results_root}")
    print(f"Save NPY:       {args.save_npy}")
    print(f"Device:         {device}")
    print("-----------------------------------\n")

    # Evaluate whichever models are specified
    if args.teacher_weights:
        evaluate_model_on_split(
            model_name="teacher_full",
            model_type="teacher",
            weights_path=args.teacher_weights,
            args=args,
            device=device,
        )

    if args.student_weights:
        evaluate_model_on_split(
            model_name="student_full",
            model_type="student",
            weights_path=args.student_weights,
            args=args,
            device=device,
        )


if __name__ == "__main__":
    main()