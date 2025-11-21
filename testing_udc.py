# File: testing_udc.py
"""
Unified UDC-SIT testing script.

- By default: runs **full-image inference** (no tiling) for teacher & student.
- Optionally: can use patch tiling (--use-tiling) or fall back to tiling on OOM.
- Computes PSNR/SSIM in Bayer (4-ch) and LPIPS on simple RGB (first 3 chans).
- Saves 4-ch predictions as .npy and metrics as CSV + summary.
- Copies metrics to Google Drive if drive_results_root exists.

Expected dataset structure:
    DATA_ROOT/
        training/input/*.npy
        training/GT/*.npy
        validation/input/*.npy
        validation/GT/*.npy
        testing/input/*.npy
        testing/GT/*.npy

Each .npy file is (H, W, 4) with values in [0, 1023].
"""

import os
import sys
import csv
import argparse
from glob import glob

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -------------------------------------------------------------------------
# Add external MambaIR path
# -------------------------------------------------------------------------
PROJECT_ROOT = os.getcwd()
MAMBAIR_PATH = os.path.join(PROJECT_ROOT, "models", "external", "MambaIR")
if MAMBAIR_PATH not in sys.path:
    sys.path.insert(0, MAMBAIR_PATH)

from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def load_full_npy_pair(input_path: str, gt_path: str):
    """
    Load (H, W, 4) .npy → normalized (4, H, W) torch tensors in [0,1].
    """
    udc = np.load(input_path)  # (H, W, 4)
    gt  = np.load(gt_path)

    assert udc.shape == gt.shape, f"Shape mismatch: {udc.shape} vs {gt.shape}"

    udc_t = torch.from_numpy(udc).permute(2, 0, 1).float() / 1023.0
    gt_t  = torch.from_numpy(gt).permute(2, 0, 1).float() / 1023.0
    return udc_t, gt_t  # (4, H, W), (4, H, W)


def run_model_tiled(
    model: torch.nn.Module,
    udc_full_4ch: torch.Tensor,
    patch_size: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Tiled inference (non-overlapping patches with reflect padding).
    udc_full_4ch: (4, H, W) on CPU or GPU, in [0,1].

    Returns:
        pred_full: (4, H, W) on CPU, in [0,1].
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x = udc_full_4ch.unsqueeze(0)  # (1, 4, H, W)
        _, _, H_orig, W_orig = x.shape

        # Pad so H,W are multiples of patch_size
        pad_h = (patch_size - H_orig % patch_size) % patch_size
        pad_w = (patch_size - W_orig % patch_size) % patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, H_pad, W_pad = x.shape

        x = x.to(device)

        ys = list(range(0, H_pad, patch_size))
        xs = list(range(0, W_pad, patch_size))
        coords = [(y, x_) for y in ys for x_ in xs]

        pred_full = torch.zeros_like(x)

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = []
            for (y, x_) in batch_coords:
                patch = x[:, :, y:y + patch_size, x_:x_ + patch_size]  # (1,4,ps,ps)
                patches.append(patch)
            patches = torch.cat(patches, dim=0)  # (B,4,ps,ps)

            with autocast():
                out, *_ = model(patches)  # (B,4,ps,ps)

            out = out.detach().cpu()
            # Stitch back
            for b, (y, x_) in enumerate(batch_coords):
                pred_full[:, :, y:y + patch_size, x_:x_ + patch_size] = out[b:b + 1]

        # Crop back to original size
        pred_full = pred_full[:, :, :H_orig, :W_orig]
        pred_full = pred_full.squeeze(0).clamp(0.0, 1.0)  # (4,H,W)
        return pred_full.cpu()


def run_model_full_image(
    model: torch.nn.Module,
    udc_full_4ch: torch.Tensor,
    patch_size: int,
    batch_size: int,
    use_tiling: bool,
) -> torch.Tensor:
    """
    Try full-image inference first.
    If use_tiling=True or full-image OOMs, fall back to run_model_tiled().
    """
    device = next(model.parameters()).device
    model.eval()

    # If explicitly requested, skip straight to tiling
    if use_tiling:
        return run_model_tiled(model, udc_full_4ch, patch_size, batch_size)

    with torch.no_grad():
        try:
            x = udc_full_4ch.unsqueeze(0).to(device)  # (1,4,H,W)
            with autocast():
                out, *_ = model(x)  # (1,4,H,W)
            pred = out.squeeze(0).detach().cpu().clamp(0.0, 1.0)
            return pred  # (4,H,W)
        except RuntimeError as e:
            # Fallback in case of CUDA OOM
            if "out of memory" in str(e).lower():
                print("[run_model_full_image] CUDA OOM on full image, falling back to tiled inference...")
                torch.cuda.empty_cache()
                return run_model_tiled(model, udc_full_4ch, patch_size, batch_size)
            else:
                raise


def compute_metrics_bayer_and_lpips(
    pred_4ch: torch.Tensor,
    gt_4ch: torch.Tensor,
    lpips_model: lpips.LPIPS,
):
    """
    pred_4ch, gt_4ch: (4, H, W) in [0,1] on CPU.
    Returns: psnr, ssim, lpips_rgb
    """
    pred_4ch = pred_4ch.clamp(0.0, 1.0)
    gt_4ch   = gt_4ch.clamp(0.0, 1.0)

    # PSNR/SSIM in Bayer / 4-ch
    pred_np = pred_4ch.permute(1, 2, 0).numpy()
    gt_np   = gt_4ch.permute(1, 2, 0).numpy()

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=-1)

    # LPIPS on pseudo-RGB (first 3 channels)
    device = next(lpips_model.parameters()).device
    pred_rgb = pred_4ch[:3, :, :].unsqueeze(0).to(device)  # (1,3,H,W)
    gt_rgb   = gt_4ch[:3, :, :].unsqueeze(0).to(device)

    # Scale to [-1,1] for LPIPS
    pred_rgb_lp = pred_rgb * 2.0 - 1.0
    gt_rgb_lp   = gt_rgb   * 2.0 - 1.0

    with torch.no_grad():
        lp = lpips_model(pred_rgb_lp, gt_rgb_lp).item()

    return psnr, ssim, lp


# -------------------------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------------------------
def evaluate_model_on_split(
    model_name: str,
    model_type: str,
    weights_path: str,
    args,
    device: str,
    lpips_model: lpips.LPIPS,
):
    """
    Evaluate a given model (teacher or student) on one split.
    """
    split_dir = os.path.join(args.data_root, args.split)
    input_dir = os.path.join(split_dir, "input")
    gt_dir    = os.path.join(split_dir, "GT")

    assert os.path.isdir(input_dir), f"Missing input dir: {input_dir}"
    assert os.path.isdir(gt_dir),    f"Missing GT dir: {gt_dir}"

    input_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    if args.max_images is not None:
        input_files = input_files[:args.max_images]

    num_images = len(input_files)
    print(f"Data root: {args.data_root}")
    print(f"Num images: {num_images}")

    # Build model
    if model_type == "teacher":
        print("--- [Teacher] Initializing with 4 in-channels and 4 out-channels.")
        model = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(device)
        print("--- [Teacher] Model initialized (MambaIR + Freq residual gating).")
    elif model_type == "student":
        print("--- [Student] Initializing with 4 in-channels and 4 out-channels.")
        model = UNetStudent(in_channels=4, out_channels=4).to(device)
        print("--- [Student] Model initialized with frequency blocks.")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load weights
    if not os.path.exists(weights_path):
        print(f"[{model_name}] Weights not found: {weights_path}, skipping.")
        return
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"[{model_name}] Error loading weights from {weights_path}: {e}")
        return

    model.eval()

    # Output directories
    model_result_dir = os.path.join(args.results_root, f"{model_name}_{args.split}")
    os.makedirs(model_result_dir, exist_ok=True)

    npy_out_dir = os.path.join(model_result_dir, "npy")
    os.makedirs(npy_out_dir, exist_ok=True)

    metrics_csv_path = os.path.join(model_result_dir, "metrics_raw.csv")
    summary_path     = os.path.join(model_result_dir, "metrics_summary.txt")

    # Metrics accumulators
    rows = []
    psnr_list  = []
    ssim_list  = []
    lpips_list = []

    # Progress bar
    pbar = tqdm(input_files, desc=f"[{model_name}] images", unit="img")

    for inp_path in pbar:
        base = os.path.basename(inp_path)
        gt_path = os.path.join(gt_dir, base)
        if not os.path.exists(gt_path):
            print(f"WARNING: GT not found for {base}, skipping.")
            continue

        # Load full 4-ch image
        udc_full, gt_full = load_full_npy_pair(inp_path, gt_path)  # (4,H,W)
        # gt_full stays on CPU
        udc_full = udc_full.to(device)

        # Inference (full-image if possible; falls back to tiling if needed)
        pred_full = run_model_full_image(
            model,
            udc_full,
            patch_size=args.patch_size,
            batch_size=args.patch_batch,
            use_tiling=args.use_tiling,
        )

        # Save prediction as (H, W, 4) .npy
        npy_path = os.path.join(npy_out_dir, base)
        np.save(npy_path, pred_full.permute(1, 2, 0).numpy())

        # Compute metrics (Bayer PSNR/SSIM + RGB LPIPS)
        psnr, ssim, lpv = compute_metrics_bayer_and_lpips(
            pred_full.cpu(), gt_full.cpu(), lpips_model
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

        # Update progress bar postfix
        mean_psnr  = float(np.mean(psnr_list))
        mean_ssim  = float(np.mean(ssim_list))
        mean_lpips = float(np.mean(lpips_list))
        pbar.set_postfix({
            "PSNR": f"{mean_psnr:.2f}",
            "SSIM": f"{mean_ssim:.3f}",
        })

    pbar.close()

    # Write CSV
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "psnr_bayer", "ssim_bayer", "lpips_rgb"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summary
    mean_psnr  = float(np.mean(psnr_list)) if psnr_list else float("nan")
    mean_ssim  = float(np.mean(ssim_list)) if ssim_list else float("nan")
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

    # Copy metrics to Drive (if configured)
    if args.drive_results_root is not None and os.path.isdir(os.path.dirname(args.drive_results_root)):
        drive_model_dir = os.path.join(args.drive_results_root, f"{model_name}_{args.split}")
        os.makedirs(drive_model_dir, exist_ok=True)
        import shutil
        shutil.copy(metrics_csv_path, os.path.join(drive_model_dir, "metrics_raw.csv"))
        shutil.copy(summary_path,     os.path.join(drive_model_dir, "metrics_summary.txt"))
        print(f"Copied metrics to Drive: {drive_model_dir}")
    else:
        print("[testing_udc] Drive results root not found or not set; skipping copy to Drive.")


def parse_args():
    parser = argparse.ArgumentParser(description="UDC-SIT full-image testing (teacher & student).")

    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/dataset/UDC-SIT",
        help="Root of UDC-SIT dataset (contains training/validation/testing folders).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["training", "validation", "testing"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size (only used if tiling is enabled or OOM fallback).",
    )
    parser.add_argument(
        "--patch-batch",
        type=int,
        default=8,
        help="Number of patches per batch for tiled inference.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Max number of images to evaluate (None = all).",
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        default="teacher_4ch_22epochs_bs8.pth",
        help="Path to teacher checkpoint.",
    )
    parser.add_argument(
        "--student-ckpt",
        type=str,
        default="student_distilled_4ch_full_data.pth",
        help="Path to student checkpoint.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results_quick",
        help="Local directory to store prediction .npy and metrics.",
    )
    parser.add_argument(
        "--drive-results-root",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/results_quick",
        help="Google Drive directory to copy metrics into (optional).",
    )
    parser.add_argument(
        "--use-tiling",
        action="store_true",
        help="Force tiled inference (otherwise do full-image and only fall back to tiling on OOM).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n--- [testing_udc] Configuration ---")
    print(f"Data root:      {args.data_root}")
    print(f"Split:          {args.split}")
    print(f"Patch size:     {args.patch_size}")
    print(f"Patch batch:    {args.patch_batch}")
    print(f"Max images:     {args.max_images}")
    print(f"Teacher ckpt:   {args.teacher_ckpt}")
    print(f"Student ckpt:   {args.student_ckpt}")
    print(f"Results root:   {args.results_root}")
    print(f"Drive results:  {args.drive_results_root}")
    print(f"Use tiling:     {args.use_tiling}")
    print(f"Device:         {device}")
    print("-----------------------------------\n")

    os.makedirs(args.results_root, exist_ok=True)

    # LPIPS model (perceptual)
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    print("Setting up LPIPS (VGG) on device:", device)

    # Teacher
    print(f"\n=== [testing_udc] Evaluating teacher_full (teacher) on split '{args.split}' ===")
    evaluate_model_on_split(
        model_name="teacher_full",
        model_type="teacher",
        weights_path=args.teacher_ckpt,
        args=args,
        device=device,
        lpips_model=lpips_model,
    )

    # Student
    print(f"\n=== [testing_udc] Evaluating student_full (student) on split '{args.split}' ===")
    evaluate_model_on_split(
        model_name="student_full",
        model_type="student",
        weights_path=args.student_ckpt,
        args=args,
        device=device,
        lpips_model=lpips_model,
    )


if __name__ == "__main__":
    main()
