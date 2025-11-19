# File: testing_full.py
"""
Full evaluation on the UDC-SIT validation + test splits.

- Loads full 4-ch .npy images (H, W, 4) from:
    VAL_DIR/input/*.npy, VAL_DIR/GT/*.npy
    TEST_DIR/input/*.npy, TEST_DIR/GT/*.npy  (if TEST_DIR exists)
- Tiles into 256x256 patches, runs model, reconstructs full image.
- Computes PSNR, SSIM in Bayer domain and LPIPS on simple RGB.
- Saves predictions and metrics locally and copies CSV + summaries to Google Drive.

Assumptions:
- Each .npy is (H, W, 4) with values in [0, 1023].
- You already trained:
    teacher_4ch_22epochs_bs8.pth
    student_distilled_4ch_full_data.pth
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

# --- Add external MambaIR path if needed ---
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)

from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent

# ---------------- CONFIG ----------------
VAL_DIR  = "/content/dataset/UDC-SIT/validation"
TEST_DIR = "/content/drive/MyDrive/Computational Imaging Project/UDC-SIT/UDC-SIT/test"  # adjust if needed

PATCH_SIZE = 256
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = [
    {
        "name": "teacher_full",
        "model_type": "teacher",
        "weights": "teacher_4ch_22epochs_bs8.pth",
    },
    {
        "name": "student_full",
        "model_type": "student",
        "weights": "student_distilled_4ch_full_data.pth",
    },
]

RESULTS_ROOT = "results_full"
DRIVE_RESULTS_ROOT = "/content/drive/MyDrive/Computational Imaging Project/results_full"

SAVE_NPY = True
NUM_EXAMPLES_VAL  = None   # None = all
NUM_EXAMPLES_TEST = None   # None = all
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
        padded: (C, H_pad, W_pad)
        H_orig, W_orig: original spatial dims
    """
    _, H, W = tensor.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h == 0 and pad_w == 0:
        return tensor, H, W

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
        udc_full_4ch = udc_full_4ch.to(device)

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
                patch = x[:, :, y:y + patch_size, x_:x_ + patch_size]
                patches.append(patch)
            patches = torch.cat(patches, dim=0)  # (B, 4, ps, ps)

            if device.type == "cuda":
                with autocast():
                    out, *_ = model(patches)
            else:
                out, *_ = model(patches)

            for b, (y, x_) in enumerate(batch_coords):
                pred_full[:, :, y:y + patch_size, x_:x_ + patch_size] = out[b:b + 1]

        pred_full = pred_full[:, :, :H_orig, :W_orig]  # (1, 4, H_orig, W_orig)
        return pred_full.squeeze(0).cpu()


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
    # Bayer-domain PSNR/SSIM
    pred_np = pred_4ch.permute(1, 2, 0).cpu().numpy()
    gt_np   = gt_4ch.permute(1, 2, 0).cpu().numpy()

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=-1)

    # LPIPS on pseudo-RGB (first 3 channels)
    pred_rgb = pred_4ch[:3, :, :].unsqueeze(0)
    gt_rgb   = gt_4ch[:3, :, :].unsqueeze(0)

    lpips_device = next(lpips_model.parameters()).device
    pred_rgb = pred_rgb.to(lpips_device)
    gt_rgb   = gt_rgb.to(lpips_device)

    pred_rgb_lp = pred_rgb * 2 - 1
    gt_rgb_lp   = gt_rgb   * 2 - 1

    with torch.no_grad():
        lp = lpips_model(pred_rgb_lp, gt_rgb_lp).item()

    return psnr, ssim, lp


def evaluate_split(
    model,
    model_name,
    split_name,
    data_dir,
    lpips_model,
    results_root,
    drive_results_root,
    patch_size,
    batch_size,
    save_npy=True,
    num_examples=None,
):
    """
    Evaluate a single model on one split (val or test).

    Saves:
      - predictions as 4-ch .npy (if save_npy)
      - per-image metrics CSV
      - summary .txt
      - copies CSV+summary to Drive
    """
    input_dir = os.path.join(data_dir, "input")
    gt_dir    = os.path.join(data_dir, "GT")

    if not os.path.isdir(input_dir):
        print(f"[{split_name}] WARNING: Input dir not found: {input_dir} — skipping.")
        return

    if not os.path.isdir(gt_dir):
        print(f"[{split_name}] WARNING: GT dir not found: {gt_dir} — skipping (no metrics).")
        return

    input_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    if num_examples is not None:
        input_files = input_files[:num_examples]

    if not input_files:
        print(f"[{split_name}] WARNING: No .npy files in {input_dir} — skipping.")
        return

    print(f"\n=== [{model_name}] Evaluating split: {split_name} ===")
    print(f"Data dir: {data_dir}, Num images: {len(input_files)}")

    model_result_dir = os.path.join(results_root, f"{model_name}_{split_name}")
    os.makedirs(model_result_dir, exist_ok=True)

    if save_npy:
        npy_out_dir = os.path.join(model_result_dir, "npy")
        os.makedirs(npy_out_dir, exist_ok=True)
    else:
        npy_out_dir = None

    metrics_csv_path = os.path.join(model_result_dir, "metrics_full.csv")
    summary_path     = os.path.join(model_result_dir, "metrics_summary.txt")

    rows = []
    psnr_list  = []
    ssim_list  = []
    lpips_list = []

    for idx, inp_path in enumerate(input_files):
        base = os.path.basename(inp_path)
        gt_path = os.path.join(gt_dir, base)
        if not os.path.exists(gt_path):
            print(f"[{split_name}] WARNING: GT not found for {base}, skipping.")
            continue

        udc_full, gt_full = load_full_npy_pair(inp_path, gt_path)

        pred_full = run_model_on_full_image(
            model, udc_full, patch_size=patch_size, batch_size=batch_size
        )

        # Save predicted 4-ch .npy
        if save_npy and npy_out_dir is not None:
            out_npy_path = os.path.join(npy_out_dir, base)
            np.save(out_npy_path, pred_full.permute(1, 2, 0).numpy())  # (H, W, 4)

        psnr, ssim, lpv = compute_metrics_bayer_and_lpips(pred_full, gt_full, lpips_model)

        rows.append({
            "filename": base,
            "psnr_bayer": psnr,
            "ssim_bayer": ssim,
            "lpips_rgb": lpv,
        })
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpv)

    # Per-image CSV
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr_bayer", "ssim_bayer", "lpips_rgb"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    mean_psnr  = float(np.mean(psnr_list)) if psnr_list else float("nan")
    mean_ssim  = float(np.mean(ssim_list)) if ssim_list else float("nan")
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else float("nan")

    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Split: {split_name}\n")
        f.write(f"Num images: {len(psnr_list)}\n\n")
        f.write(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB\n")
        f.write(f"Mean SSIM (Bayer): {mean_ssim:.4f}\n")
        f.write(f"Mean LPIPS (RGB):  {mean_lpips:.4f}\n")

    print(f"\n--- [{model_name}] {split_name} results ---")
    print(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB")
    print(f"Mean SSIM (Bayer): {mean_ssim:.4f}")
    print(f"Mean LPIPS (RGB):  {mean_lpips:.4f}")
    print(f"Saved CSV to {metrics_csv_path}")
    print(f"Saved summary to {summary_path}")

    # Copy metrics to Drive
    import shutil
    drive_model_dir = os.path.join(drive_results_root, f"{model_name}_{split_name}")
    os.makedirs(drive_model_dir, exist_ok=True)
    shutil.copy(metrics_csv_path, os.path.join(drive_model_dir, "metrics_full.csv"))
    shutil.copy(summary_path,     os.path.join(drive_model_dir, "metrics_summary.txt"))
    print(f"Copied metrics to Drive: {drive_model_dir}")


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(DRIVE_RESULTS_ROOT, exist_ok=True)

    lpips_model = lpips.LPIPS(net='vgg').to(DEVICE)

    for cfg in MODEL_CONFIGS:
        model_name = cfg["name"]
        model_type = cfg["model_type"]
        weights    = cfg["weights"]

        if not os.path.exists(weights):
            print(f"--- [testing_full] Skipping {model_name}, weights not found: {weights}")
            continue

        print(f"\n=== Loading {model_name} ({model_type}) from {weights} ===")
        if model_type == "teacher":
            model = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(DEVICE)
        elif model_type == "student":
            model = UNetStudent(in_channels=4, out_channels=4).to(DEVICE)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.load_state_dict(torch.load(weights, map_location=DEVICE))
        model.eval()

        # Validation split
        if os.path.isdir(VAL_DIR):
            evaluate_split(
                model=model,
                model_name=model_name,
                split_name="val",
                data_dir=VAL_DIR,
                lpips_model=lpips_model,
                results_root=RESULTS_ROOT,
                drive_results_root=DRIVE_RESULTS_ROOT,
                patch_size=PATCH_SIZE,
                batch_size=BATCH_SIZE,
                save_npy=SAVE_NPY,
                num_examples=NUM_EXAMPLES_VAL,
            )
        else:
            print(f"[testing_full] VAL_DIR not found: {VAL_DIR}")

        # Test split
        if TEST_DIR and os.path.isdir(TEST_DIR):
            evaluate_split(
                model=model,
                model_name=model_name,
                split_name="test",
                data_dir=TEST_DIR,
                lpips_model=lpips_model,
                results_root=RESULTS_ROOT,
                drive_results_root=DRIVE_RESULTS_ROOT,
                patch_size=PATCH_SIZE,
                batch_size=BATCH_SIZE,
                save_npy=SAVE_NPY,
                num_examples=NUM_EXAMPLES_TEST,
            )
        else:
            print(f"[testing_full] TEST_DIR not found or not set; skipping test split.")


if __name__ == "__main__":
    main()
