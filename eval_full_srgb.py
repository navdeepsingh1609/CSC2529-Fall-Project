# File: eval_full_srgb.py
"""
Full-resolution sRGB evaluation for UDC-SIT.

Pipeline:
  - Start from full 4-ch .npy:
        DATA_ROOT/<split>/input/*.npy
        DATA_ROOT/<split>/GT/*.npy
  - Use model predictions saved by testing_full.py:
        results_full/<model_name>_<split_tag>/npy/*.npy
  - Convert each 4-ch file (input, GT, pred) to sRGB using the
    same rawpy-based ISP idea as utils/visualize_sit.py
    (background.dng, postprocess, crop).
  - Compute PSNR/SSIM in sRGB space:
        * Input vs GT (for reference)
        * Pred vs GT  (for quality)
  - Save per-image CSV and summary text, and copy to Google Drive.

Assumptions:
  - DATA_ROOT:
        /content/dataset/UDC-SIT
    with subdirs: "validation", "test" (or "testing" if you adjust below).
  - background.dng is either in:
        ./background.dng
    or:
        ./data/background.dng
  - testing_full.py has already been run and produced:
        results_full/teacher_full_val/npy
        results_full/teacher_full_test/npy
        results_full/student_full_val/npy
        results_full/student_full_test/npy
"""

import os
import csv
import numpy as np
from glob import glob

import rawpy
import imageio.v2 as imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ---------------- CONFIG ----------------

# Root where the original UDC-SIT .npy are extracted
DATA_ROOT = "/content/dataset/UDC-SIT"

# Where testing_full.py stored 4-ch predictions
PRED_ROOT = "results_full"

# Where to save sRGB metrics locally
RESULTS_SRGB_ROOT = "results_full_srgb"

# Where to mirror results in Google Drive
DRIVE_RESULTS_SRGB_ROOT = "/content/drive/MyDrive/Computational Imaging Project/results_full_srgb"

# Try to locate background.dng
_DEFAULT_BG1 = "background.dng"
_DEFAULT_BG2 = os.path.join("data", "background.dng")

# Models + splits to evaluate
MODEL_NAMES = ["teacher_full", "student_full"]

# Mapping between "tag" (used in PRED_ROOT) and dataset subdir name
SPLITS = [
    # tag,       data_subdir
    ("val",  "validation"),
    ("test", "test"),         # if your folder is "testing", change to "testing"
]

# Whether to also write out the sRGB PNGs (optional, but useful for eyeballing)
SAVE_SRGB_PNGS = False

# ----------------------------------------


def find_background_dng():
    """
    Find background.dng either in repo root or in data/.
    Raises FileNotFoundError if not found.
    """
    if os.path.exists(_DEFAULT_BG1):
        return _DEFAULT_BG1
    if os.path.exists(_DEFAULT_BG2):
        return _DEFAULT_BG2
    raise FileNotFoundError(
        f"Could not find background.dng in '{_DEFAULT_BG1}' or '{_DEFAULT_BG2}'. "
        "Please place the file in the repo root or under data/."
    )


def tensor4ch_to_srgb_array(tensor_4ch, background_path):
    """
    Convert a 4-channel tensor (CHW, [0,1]) to an sRGB image using
    the same rawpy pipeline idea as utils/visualize_sit.py.

    Args:
        tensor_4ch: torch.Tensor of shape (4, H, W), values in [0,1]
        background_path: path to background.dng

    Returns:
        sRGB image as numpy array (H_crop, W_crop, 3), float32 in [0,1]
    """
    import torch  # local import to keep this file general

    if tensor_4ch.ndim != 3:
        raise ValueError(f"Expected tensor_4ch to be 3D (4,H,W), got shape {tensor_4ch.shape}")
    if tensor_4ch.shape[0] != 4:
        # if someone gives HWC, fix it
        if tensor_4ch.shape[-1] == 4:
            tensor_4ch = tensor_4ch.permute(2, 0, 1)
        else:
            raise ValueError(f"Expected 4 channels, got shape {tensor_4ch.shape}")

    # Back to numpy, scale up to sensor-like range
    npy = tensor_4ch.to("cpu").detach().numpy() * 1023.0  # (4, H, W)
    _, H, W = npy.shape

    # Mimic visualize_sit.py: embed into the RAW mosaic of background.dng
    with rawpy.imread(background_path) as raw:
        # Bayer pattern views (these are views into raw.raw_image)
        GR = raw.raw_image[0::2, 0::2]
        R  = raw.raw_image[0::2, 1::2]
        B  = raw.raw_image[1::2, 0::2]
        GB = raw.raw_image[1::2, 1::2]

        # Zero out existing content
        GR[:, :] = 0
        R[:,  :] = 0
        B[:,  :] = 0
        GB[:, :] = 0

        # Fit patch into top-left of the RAW mosaic
        H_use = min(H, GR.shape[0])
        W_use = min(W, GR.shape[1])

        GR[:H_use, :W_use] = npy[0, :H_use, :W_use]
        R [:H_use, :W_use] = npy[1, :H_use, :W_use]
        B [:H_use, :W_use] = npy[2, :H_use, :W_use]
        GB[:H_use, :W_use] = npy[3, :H_use, :W_use]

        # Run ISP (demosaic + WB + etc.)
        rgb = raw.postprocess()  # uint8 or uint16, shape (H_raw, W_raw, 3)

    # Crop to the same region as visualize_sit.py
    start_row, start_col = 0, 464
    end_row, end_col     = 3584, 3024
    rgb = rgb[start_row:end_row, start_col:end_col]  # (Hc, Wc, 3)

    # Convert to float32 [0,1] for metrics
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.5:
        rgb /= 255.0

    return rgb  # (Hc, Wc, 3), float32 in [0,1]


def fourch_npy_to_srgb(npy_path, background_path):
    """
    Load a 4-ch .npy (H,W,4) from disk, normalize to [0,1] if needed,
    convert to CHW tensor, and call tensor4ch_to_srgb_array.

    Works for both:
      - Dataset .npy in [0,1023]
      - Predictions .npy in [0,1] or [0,1023]
    """
    import torch

    arr = np.load(npy_path).astype(np.float32)  # (H,W,4)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"Expected (H,W,4) in {npy_path}, got {arr.shape}")

    maxv = float(arr.max())
    if maxv > 2.0:
        # assume [0,1023]
        arr_norm = arr / 1023.0
    else:
        # assume already [0,1]
        arr_norm = arr

    tensor_4ch = torch.from_numpy(arr_norm).permute(2, 0, 1)  # (4,H,W)
    return tensor4ch_to_srgb_array(tensor_4ch, background_path)


def compute_srgb_metrics(gt_srgb, pred_srgb):
    """
    Compute PSNR & SSIM in sRGB space.

    gt_srgb, pred_srgb: (H,W,3) float32 in [0,1]
    """
    psnr = peak_signal_noise_ratio(gt_srgb, pred_srgb, data_range=1.0)
    ssim = structural_similarity(
        gt_srgb, pred_srgb, data_range=1.0, channel_axis=-1
    )
    return psnr, ssim


def evaluate_srgb_for_model_and_split(
    model_name,
    split_tag,
    split_data_subdir,
    data_root,
    preds_root,
    background_path,
    results_srgb_root,
    drive_results_srgb_root,
):
    """
    For a given model (teacher_full / student_full) and split (val / test):

      - Load GT 4-ch .npy and input 4-ch .npy from DATA_ROOT/split/...
      - Load predictions from results_full/<model_name>_<split_tag>/npy
      - Convert each to sRGB via the background.dng ISP
      - Compute:
            * PSNR/SSIM(input_srgb, gt_srgb)
            * PSNR/SSIM(pred_srgb,   gt_srgb)
      - Save CSV + summary; copy to Drive.
    """
    # Dataset dirs
    split_dir   = os.path.join(data_root, split_data_subdir)
    input_dir   = os.path.join(split_dir, "input")
    gt_dir      = os.path.join(split_dir, "GT")

    # Predictions dir
    pred_dir = os.path.join(preds_root, f"{model_name}_{split_tag}", "npy")

    if not os.path.isdir(input_dir):
        print(f"[eval_full_srgb] WARNING: Missing input dir: {input_dir}. Skipping.")
        return
    if not os.path.isdir(gt_dir):
        print(f"[eval_full_srgb] WARNING: Missing GT dir: {gt_dir}. Skipping.")
        return
    if not os.path.isdir(pred_dir):
        print(f"[eval_full_srgb] WARNING: Missing predictions dir: {pred_dir}. Skipping.")
        return

    gt_files = sorted(glob(os.path.join(gt_dir, "*.npy")))
    if not gt_files:
        print(f"[eval_full_srgb] WARNING: No GT .npy files in {gt_dir}. Skipping.")
        return

    print(f"\n=== [eval_full_srgb] Model: {model_name}, Split: {split_tag} ===")
    print(f"Data dir: {split_dir}")
    print(f"Predictions dir: {pred_dir}")
    print(f"Num images (GT): {len(gt_files)}")

    # Output dirs (local)
    model_split_root = os.path.join(results_srgb_root, f"{model_name}_{split_tag}")
    os.makedirs(model_split_root, exist_ok=True)

    if SAVE_SRGB_PNGS:
        png_root_input  = os.path.join(model_split_root, "png_input")
        png_root_gt     = os.path.join(model_split_root, "png_gt")
        png_root_pred   = os.path.join(model_split_root, "png_pred")
        os.makedirs(png_root_input, exist_ok=True)
        os.makedirs(png_root_gt,    exist_ok=True)
        os.makedirs(png_root_pred,  exist_ok=True)

    csv_path     = os.path.join(model_split_root, "metrics_srgb.csv")
    summary_path = os.path.join(model_split_root, "metrics_srgb_summary.txt")

    rows = []
    psnr_in_list   = []
    ssim_in_list   = []
    psnr_pred_list = []
    ssim_pred_list = []

    # Loop over GT files (these define the IDs)
    for gt_path in gt_files:
        fname = os.path.basename(gt_path)                # e.g. "0001_gt.npy" or "0001.npy"
        stem  = os.path.splitext(fname)[0]

        # We assume the input filename matches GT filename
        input_path = os.path.join(input_dir, fname)
        # Pred file is in pred_dir / fname (same naming convention as testing_full)
        pred_path  = os.path.join(pred_dir, fname)

        if not os.path.exists(input_path):
            print(f"  [WARN] Missing input for {fname}, skipping.")
            continue
        if not os.path.exists(pred_path):
            print(f"  [WARN] Missing prediction for {fname}, skipping.")
            continue

        # Convert each 4-ch .npy to sRGB
        try:
            srgb_inp = fourch_npy_to_srgb(input_path, background_path)
            srgb_gt  = fourch_npy_to_srgb(gt_path,    background_path)
            srgb_pred= fourch_npy_to_srgb(pred_path,  background_path)
        except Exception as e:
            print(f"  [ERROR] Failed to convert to sRGB for {fname}: {e}")
            continue

        # Optional: save PNGs for visual inspection
        if SAVE_SRGB_PNGS:
            png_name = stem + ".png"
            imageio.imwrite(os.path.join(png_root_input, png_name), (srgb_inp * 255).astype(np.uint8))
            imageio.imwrite(os.path.join(png_root_gt,    png_name), (srgb_gt  * 255).astype(np.uint8))
            imageio.imwrite(os.path.join(png_root_pred,  png_name), (srgb_pred* 255).astype(np.uint8))

        # Compute PSNR/SSIM
        psnr_in, ssim_in   = compute_srgb_metrics(srgb_gt, srgb_inp)
        psnr_pr, ssim_pr   = compute_srgb_metrics(srgb_gt, srgb_pred)

        rows.append({
            "filename": fname,
            "psnr_input_srgb": psnr_in,
            "ssim_input_srgb": ssim_in,
            "psnr_pred_srgb":  psnr_pr,
            "ssim_pred_srgb":  ssim_pr,
        })

        psnr_in_list.append(psnr_in)
        ssim_in_list.append(ssim_in)
        psnr_pred_list.append(psnr_pr)
        ssim_pred_list.append(ssim_pr)

    if not rows:
        print(f"[eval_full_srgb] No valid samples for model={model_name}, split={split_tag}.")
        return

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "filename",
            "psnr_input_srgb", "ssim_input_srgb",
            "psnr_pred_srgb",  "ssim_pred_srgb",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summary metrics
    mean_psnr_in   = float(np.mean(psnr_in_list))
    mean_ssim_in   = float(np.mean(ssim_in_list))
    mean_psnr_pred = float(np.mean(psnr_pred_list))
    mean_ssim_pred = float(np.mean(ssim_pred_list))

    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Split: {split_tag} (data subdir: {split_data_subdir})\n")
        f.write(f"Num images: {len(rows)}\n\n")
        f.write(f"Mean PSNR (input→GT, sRGB): {mean_psnr_in:.4f} dB\n")
        f.write(f"Mean SSIM (input→GT, sRGB): {mean_ssim_in:.4f}\n")
        f.write(f"Mean PSNR (pred→GT,  sRGB): {mean_psnr_pred:.4f} dB\n")
        f.write(f"Mean SSIM (pred→GT,  sRGB): {mean_ssim_pred:.4f}\n")

    print(f"\n[eval_full_srgb] Done model={model_name}, split={split_tag}")
    print(f"  Mean PSNR (input→GT): {mean_psnr_in:.4f} dB")
    print(f"  Mean SSIM (input→GT): {mean_ssim_in:.4f}")
    print(f"  Mean PSNR (pred→GT):  {mean_psnr_pred:.4f} dB")
    print(f"  Mean SSIM (pred→GT):  {mean_ssim_pred:.4f}")
    print(f"  CSV:     {csv_path}")
    print(f"  Summary: {summary_path}")

    # Copy to Drive
    try:
        drive_model_root = os.path.join(drive_results_srgb_root, f"{model_name}_{split_tag}")
        os.makedirs(drive_model_root, exist_ok=True)

        import shutil
        shutil.copy(csv_path,     os.path.join(drive_model_root, "metrics_srgb.csv"))
        shutil.copy(summary_path, os.path.join(drive_model_root, "metrics_srgb_summary.txt"))
        print(f"  Copied metrics to Drive: {drive_model_root}")
    except Exception as e:
        print(f"  [WARN] Could not copy metrics to Drive: {e}")


def main():
    os.makedirs(RESULTS_SRGB_ROOT, exist_ok=True)
    os.makedirs(DRIVE_RESULTS_SRGB_ROOT, exist_ok=True)

    background_path = find_background_dng()
    print(f"[eval_full_srgb] Using background.dng at: {background_path}")

    for model_name in MODEL_NAMES:
        for split_tag, split_data_subdir in SPLITS:
            evaluate_srgb_for_model_and_split(
                model_name=model_name,
                split_tag=split_tag,
                split_data_subdir=split_data_subdir,
                data_root=DATA_ROOT,
                preds_root=PRED_ROOT,
                background_path=background_path,
                results_srgb_root=RESULTS_SRGB_ROOT,
                drive_results_srgb_root=DRIVE_RESULTS_SRGB_ROOT,
            )


if __name__ == "__main__":
    main()
