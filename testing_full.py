# File: testing_full.py
"""
Full evaluation on the UDC-SIT dataset (validation + test).

- Loads full 4-ch .npy images from:
    DATA_ROOT/{split}/input/*.npy
    DATA_ROOT/{split}/GT/*.npy
- Tiles into overlapping 256x256 patches, runs model, reconstructs full image.
- Uses Hann window blending to avoid patch seams.
- Computes:
    - PSNR / SSIM in Bayer (4-ch) domain on full images
    - LPIPS on simple RGB (first 3 channels)
- Saves metrics locally and copies CSV + summary to Google Drive.
- Also saves predicted 4-ch .npy for later sRGB eval / visualization.

Adjust DATA_ROOT to match your actual extracted directory.
"""

import os
import sys
import csv
import time
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

# ---------------- CONFIG ----------------
# Set this to your full dataset root.
# Example if you extracted under /content/dataset/UDC-SIT:
#   /content/dataset/UDC-SIT/validation/input/*.npy
#   /content/dataset/UDC-SIT/validation/GT/*.npy
#   /content/dataset/UDC-SIT/testing/input/*.npy
#   /content/dataset/UDC-SIT/testing/GT/*.npy
DATA_ROOT = "/content/dataset/UDC-SIT"

# If your directories are different (e.g., data/UDC-SIT_full/UDC-SIT),
# change DATA_ROOT accordingly, e.g.:
# DATA_ROOT = "data/UDC-SIT_full/UDC-SIT"

SPLITS = ["validation", "testing"]   # adjust to ["validation", "test"] if your folder is called "test"

PATCH_SIZE = 256
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LPIPS on CPU to avoid device mismatch + VRAM usage
LPIPS_DEVICE = "cpu"

MODEL_CONFIGS = [
    {
        "name": "teacher_full",
        "model_type": "teacher",
        "weights": "teacher_4ch_22epochs_bs8.pth",          # adjust if needed
    },
    {
        "name": "student_full",
        "model_type": "student",
        "weights": "student_distilled_4ch_full_data.pth",   # adjust if needed
    },
]

RESULTS_ROOT = "results_full"
DRIVE_RESULTS_ROOT = "/content/drive/MyDrive/Computational Imaging Project/results_full"

SAVE_NPY = True
NUM_EXAMPLES = None   # None = all images, or set an int to restrict
# ---------------------------------------


def load_full_npy_pair(input_path, gt_path):
    """
    Load 4-ch full .npy (H, W, 4) → normalized tensors (4, H, W) in [0,1] on CPU.
    """
    udc = np.load(input_path)
    gt  = np.load(gt_path)

    assert udc.shape == gt.shape, f"Shape mismatch: {udc.shape} vs {gt.shape}"

    udc = torch.from_numpy(udc).permute(2, 0, 1).float() / 1023.0
    gt  = torch.from_numpy(gt).permute(2, 0, 1).float() / 1023.0
    return udc, gt


def pad_to_multiple(tensor, patch_size):
    """
    Pad CHW tensor so H,W are multiples of patch_size using reflect padding.
    """
    _, H, W = tensor.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h == 0 and pad_w == 0:
        return tensor, H, W

    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
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
        pred_full_4ch (CPU), elapsed_time_sec
    """
    device = next(model.parameters()).device
    model.eval()

    start_time = time.time()

    with torch.no_grad():
        x = udc_full_4ch
        x, H_orig, W_orig = pad_to_multiple(x, patch_size)
        C, H_pad, W_pad = x.shape

        x = x.unsqueeze(0)  # (1, C, H_pad, W_pad)

        stride = patch_size - overlap
        if stride <= 0:
            raise ValueError(
                f"overlap must be < patch_size, got overlap={overlap}, patch_size={patch_size}"
            )

        ys = list(range(0, H_pad - patch_size + 1, stride))
        xs = list(range(0, W_pad - patch_size + 1, stride))
        if ys[-1] != H_pad - patch_size:
            ys.append(H_pad - patch_size)
        if xs[-1] != W_pad - patch_size:
            xs.append(W_pad - patch_size)
        coords = [(y, x_) for y in ys for x_ in xs]

        pred_accum = torch.zeros(1, C, H_pad, W_pad, device=device)
        weight_acc = torch.zeros(1, 1, H_pad, W_pad, device=device)

        win_1d = torch.hann_window(patch_size, periodic=False, device=device)
        win_2d = win_1d[:, None] * win_1d[None, :]
        win_2d = win_2d.unsqueeze(0).unsqueeze(0)  # (1,1,ps,ps)

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = []
            for (y, x_) in batch_coords:
                patch = x[:, :, y:y + patch_size, x_:x_ + patch_size]
                patches.append(patch)
            patches = torch.cat(patches, dim=0).to(device)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out, *_ = model(patches)  # (B, 4, ps, ps)

            out_win = out * win_2d

            for b, (y, x_) in enumerate(batch_coords):
                pred_accum[:, :, y:y + patch_size, x_:x_ + patch_size] += out_win[b:b + 1]
                weight_acc[:, :, y:y + patch_size, x_:x_ + patch_size] += win_2d

        weight_acc = torch.clamp(weight_acc, min=1e-6)
        pred_full = pred_accum / weight_acc
        pred_full = pred_full[:, :, :H_orig, :W_orig]

    elapsed = time.time() - start_time
    return pred_full.squeeze(0).cpu(), elapsed


def compute_metrics_bayer_and_lpips(pred_4ch_cpu, gt_4ch_cpu, lpips_model):
    """
    pred_4ch_cpu, gt_4ch_cpu: (4, H, W) tensors in [0,1] on CPU.
    Returns: psnr, ssim, lpips_rgb
    """
    pred_np = pred_4ch_cpu.permute(1, 2, 0).numpy()
    gt_np   = gt_4ch_cpu.permute(1, 2, 0).numpy()

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=-1)

    # LPIPS on simple RGB
    pred_rgb = pred_4ch_cpu[:3, :, :].unsqueeze(0)
    gt_rgb   = gt_4ch_cpu[:3, :, :].unsqueeze(0)

    pred_rgb_lp = (pred_rgb * 2 - 1).to(LPIPS_DEVICE)
    gt_rgb_lp   = (gt_rgb   * 2 - 1).to(LPIPS_DEVICE)

    with torch.no_grad():
        lp = lpips_model(pred_rgb_lp, gt_rgb_lp).item()

    return psnr, ssim, lp


def evaluate_model_on_split(
    model,
    model_name,
    split,
    lpips_model,
):
    """
    Evaluate a single model on a given split (validation/testing).
    Saves:
      - 4-ch .npy predictions under results_full/{model_name}_{split}/npy
      - per-image metrics CSV
      - summary txt
    """
    input_dir = os.path.join(DATA_ROOT, split, "input")
    gt_dir    = os.path.join(DATA_ROOT, split, "GT")

    assert os.path.isdir(input_dir), f"Missing input dir: {input_dir}"
    assert os.path.isdir(gt_dir),    f"Missing GT dir: {gt_dir}"

    input_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    if NUM_EXAMPLES is not None:
        input_files = input_files[:NUM_EXAMPLES]

    print(f"\n=== [testing_full] Model: {model_name}, Split: {split} ===")
    print(f"Input dir: {input_dir}")
    print(f"GT dir:    {gt_dir}")
    print(f"Num images: {len(input_files)}")

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
    time_list = []

    for inp_path in tqdm(input_files, desc=f"{model_name} [{split}]"):
        base = os.path.basename(inp_path)
        gt_path = os.path.join(gt_dir, base)
        if not os.path.exists(gt_path):
            print(f"[WARNING] GT not found for {base}, skipping.")
            continue

        udc_full_cpu, gt_full_cpu = load_full_npy_pair(inp_path, gt_path)
        udc_full = udc_full_cpu.to(DEVICE)

        # Run model (full image) with overlap blending
        pred_full_cpu, elapsed = run_model_on_full_image(
            model,
            udc_full,
            patch_size=PATCH_SIZE,
            batch_size=BATCH_SIZE,
            overlap=64,
        )

        if SAVE_NPY and npy_out_dir is not None:
            out_npy_path = os.path.join(npy_out_dir, base)
            np.save(out_npy_path, pred_full_cpu.permute(1, 2, 0).numpy())

        psnr, ssim, lpv = compute_metrics_bayer_and_lpips(pred_full_cpu, gt_full_cpu, lpips_model)

        rows.append({
            "filename": base,
            "psnr_bayer": psnr,
            "ssim_bayer": ssim,
            "lpips_rgb": lpv,
            "time_sec": elapsed,
        })
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpv)
        time_list.append(elapsed)

    # Write CSV
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "psnr_bayer", "ssim_bayer", "lpips_rgb", "time_sec"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    mean_psnr  = float(np.mean(psnr_list)) if psnr_list else float("nan")
    mean_ssim  = float(np.mean(ssim_list)) if ssim_list else float("nan")
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else float("nan")
    mean_time  = float(np.mean(time_list)) if time_list else float("nan")

    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Num images: {len(psnr_list)}\n\n")
        f.write(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB\n")
        f.write(f"Mean SSIM (Bayer): {mean_ssim:.4f}\n")
        f.write(f"Mean LPIPS (RGB):  {mean_lpips:.4f}\n")
        f.write(f"Mean Time:         {mean_time*1000.0:.2f} ms/image\n")

    print(f"\n--- [testing_full] {model_name} [{split}] results:")
    print(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB")
    print(f"Mean SSIM (Bayer): {mean_ssim:.4f}")
    print(f"Mean LPIPS (RGB):  {mean_lpips:.4f}")
    print(f"Mean Time:         {mean_time*1000.0:.2f} ms/image")
    print(f"Saved CSV to {metrics_csv_path}")
    print(f"Saved summary to {summary_path}")

    # Copy metrics to Drive
    import shutil
    drive_model_dir = os.path.join(DRIVE_RESULTS_ROOT, f"{model_name}_{split}")
    os.makedirs(drive_model_dir, exist_ok=True)
    shutil.copy(metrics_csv_path, os.path.join(drive_model_dir, "metrics_full.csv"))
    shutil.copy(summary_path,     os.path.join(drive_model_dir, "metrics_summary.txt"))
    print(f"Copied metrics to Drive: {drive_model_dir}")


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(DRIVE_RESULTS_ROOT, exist_ok=True)

    lpips_model = lpips.LPIPS(net='vgg').to(LPIPS_DEVICE)
    print(f"[testing_full] LPIPS running on: {LPIPS_DEVICE}")
    print(f"[testing_full] Using DATA_ROOT: {DATA_ROOT}")

    for cfg in MODEL_CONFIGS:
        model_name = cfg["name"]
        model_type = cfg["model_type"]
        weights    = cfg["weights"]

        if not os.path.exists(weights):
            print(f"\n[testing_full] Skipping {model_name}, weights not found: {weights}")
            continue

        print(f"\n=== [testing_full] Loading {model_name} ({model_type}) from {weights} ===")

        if model_type == "teacher":
            model = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(DEVICE)
        elif model_type == "student":
            model = UNetStudent(in_channels=4, out_channels=4).to(DEVICE)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.load_state_dict(torch.load(weights, map_location=DEVICE))
        model.eval()

        for split in SPLITS:
            evaluate_model_on_split(
                model=model,
                model_name=model_name,
                split=split,
                lpips_model=lpips_model,
            )


if __name__ == "__main__":
    main()
