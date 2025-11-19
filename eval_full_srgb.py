# File: eval_full_srgb.py
"""
Evaluate sRGB PSNR/SSIM using the official UDC-SIT ISP-style pipeline.

This script:
- Mimics utils/visualize_sit.py (background.dng + rawpy) to convert 4-ch Bayer .npy
  into sRGB images.
- For each model (teacher_full, student_full) and split (val, test if GT exists):
    * Reads GT, input, and predicted 4-ch .npy.
    * Converts each to sRGB with rawpy postprocess + crop.
    * Computes PSNR and SSIM in sRGB space (uint8, data_range=255).
- Saves per-image CSV and split-level summary, and copies results to Google Drive.

Assumptions:
- Dataset:
    VAL_DIR/input/*.npy, VAL_DIR/GT/*.npy
    TEST_DIR/input/*.npy, TEST_DIR/GT/*.npy (if available)
- Predictions from testing_full.py:
    results_full/<model_name>_<split>/npy/*.npy
- background.dng is present at BACKGROUND_DNG_PATH.
"""

import os
import sys
import csv
from glob import glob

import numpy as np
import torch
import rawpy as rp
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ---------------- CONFIG ----------------
VAL_DIR  = "/content/dataset/UDC-SIT/validation"
TEST_DIR = "/content/dataset/UDC-SIT/test"  # adjust if your test lives elsewhere

PRED_ROOT = "results_full"

# Models evaluated (must match names in testing_full.py)
MODEL_NAMES = ["teacher_full", "student_full"]

BACKGROUND_DNG_PATH = "background.dng"  # ensure this file exists in project root

RESULTS_SRGB_ROOT = "results_full_srgb"
DRIVE_RESULTS_SRGB_ROOT = "/content/drive/MyDrive/Computational Imaging Project/results_full_srgb"
# ---------------------------------------


def tensor4ch_to_srgb_array(tensor_4ch, background_path):
    """
    Replicate the core logic of utils/visualize_sit.save_4ch_npy_png,
    but return the sRGB image as a numpy array instead of saving.

    Args:
        tensor_4ch: (4, H, W) torch.Tensor in [0,1]
        background_path: path to background.dng

    Returns:
        srgb: (H_c, W_c, 3) np.ndarray, uint8 (after postprocess + crop)
    """
    # This fn_tonumpy matches visualize_sit.py
    fn_tonumpy = lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 1)
    data = rp.imread(background_path)

    npy = fn_tonumpy(tensor_4ch.unsqueeze(0))  # -> (1, H, W) with some transpose shenanigans
    npy = npy.squeeze() * 1023.0              # expected shape (4, ?, ?) after squeeze

    # Extract Bayer planes from raw mosaic (views into raw_image)
    GR = data.raw_image[0::2, 0::2]
    R  = data.raw_image[0::2, 1::2]
    B  = data.raw_image[1::2, 0::2]
    GB = data.raw_image[1::2, 1::2]

    # Reset them to zero (as in visualize_sit.py)
    GR[:, :] = 0
    R[:, :]  = 0
    B[:, :]  = 0
    GB[:, :] = 0

    # Fill with our 4-channel data
    w, h = npy.shape[1:]  # same as visualize_sit.py
    GR[:w, :h] = npy[0][:w][:h]
    R[:w,  :h] = npy[1][:w][:h]
    B[:w,  :h] = npy[2][:w][:h]
    GB[:w, :h] = npy[3][:w][:h]

    # Run ISP and crop, as in visualize_sit.py
    newData = data.postprocess()  # default 8-bit RGB
    start = (0, 464)
    end   = (3584, 3024)
    newData = newData[start[0]:end[0], start[1]:end[1]]

    return newData  # uint8, shape (H_c, W_c, 3)


def fourch_npy_to_srgb(npy_path, background_path):
    """
    Convenience wrapper: load (H,W,4) npy in [0,1023], convert to (4,H,W) tensor in [0,1],
    then pass through tensor4ch_to_srgb_array.

    Args:
        npy_path: path to 4-ch .npy
        background_path: path to background.dng

    Returns:
        srgb: (H_c, W_c, 3) uint8
    """
    img = np.load(npy_path).astype(np.float32) / 1023.0  # (H, W, 4)
    tensor_4ch = torch.from_numpy(img).permute(2, 0, 1)  # (4, H, W)
    tensor_4ch = torch.clamp(tensor_4ch, 0.0, 1.0)
    return tensor4ch_to_srgb_array(tensor_4ch, background_path)


def evaluate_srgb_for_model_and_split(
    model_name,
    split_name,
    data_dir,
    pred_root,
    background_path,
    results_root,
    drive_results_root,
):
    """
    For a given model and split (val/test), compute sRGB PSNR/SSIM using the ISP
    described in visualize_sit.py.

    - Reads GT and input npy from data_dir.
    - Reads predicted npy from pred_root/<model_name>_<split>/npy.
    - Writes per-image CSV + summary, copies to Drive.

    If GT is not available for this split, the function returns early.
    """
    input_dir = os.path.join(data_dir, "input")
    gt_dir    = os.path.join(data_dir, "GT")

    if not os.path.isdir(input_dir):
        print(f"[eval_full_srgb] Split '{split_name}': missing input dir {input_dir}, skipping.")
        return

    if not os.path.isdir(gt_dir):
        print(f"[eval_full_srgb] Split '{split_name}': missing GT dir {gt_dir}, skipping (no metrics).")
        return

    pred_npy_dir = os.path.join(pred_root, f"{model_name}_{split_name}", "npy")
    if not os.path.isdir(pred_npy_dir):
        print(f"[eval_full_srgb] Split '{split_name}': prediction dir not found: {pred_npy_dir}, skipping.")
        return

    input_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    if not input_files:
        print(f"[eval_full_srgb] Split '{split_name}': no input .npy files in {input_dir}, skipping.")
        return

    print(f"\n=== [eval_full_srgb] Model: {model_name}, Split: {split_name} ===")
    print(f"Data dir: {data_dir}")
    print(f"Predictions dir: {pred_npy_dir}")
    print(f"Num images: {len(input_files)}")

    model_result_dir = os.path.join(results_root, f"{model_name}_{split_name}")
    os.makedirs(model_result_dir, exist_ok=True)

    metrics_csv_path = os.path.join(model_result_dir, "metrics_srgb.csv")
    summary_path     = os.path.join(model_result_dir, "metrics_srgb_summary.txt")

    rows = []
    psnr_in_list  = []
    ssim_in_list  = []
    psnr_pred_list  = []
    ssim_pred_list  = []

    for inp_path in input_files:
        base = os.path.basename(inp_path)

        gt_path   = os.path.join(gt_dir, base)
        pred_path = os.path.join(pred_npy_dir, base)

        if not os.path.exists(gt_path):
            print(f"[eval_full_srgb] WARNING: GT not found for {base}, skipping.")
            continue
        if not os.path.exists(pred_path):
            print(f"[eval_full_srgb] WARNING: Prediction not found for {base}, skipping.")
            continue

        # Convert input, GT, prediction to sRGB via ISP
        srgb_inp  = fourch_npy_to_srgb(inp_path,  background_path)
        srgb_gt   = fourch_npy_to_srgb(gt_path,   background_path)
        srgb_pred = fourch_npy_to_srgb(pred_path, background_path)

        # Ensure shapes match
        if srgb_inp.shape != srgb_gt.shape or srgb_pred.shape != srgb_gt.shape:
            print(f"[eval_full_srgb] WARNING: Shape mismatch for {base}, skipping.")
            continue

        # PSNR/SSIM in sRGB (uint8, [0,255])
        psnr_in  = peak_signal_noise_ratio(srgb_gt, srgb_inp,  data_range=255)
        ssim_in  = structural_similarity(srgb_gt, srgb_inp,  data_range=255, channel_axis=-1)

        psnr_pred = peak_signal_noise_ratio(srgb_gt, srgb_pred, data_range=255)
        ssim_pred = structural_similarity(srgb_gt, srgb_pred, data_range=255, channel_axis=-1)

        rows.append({
            "filename": base,
            "psnr_input_srgb": psnr_in,
            "ssim_input_srgb": ssim_in,
            "psnr_pred_srgb": psnr_pred,
            "ssim_pred_srgb": ssim_pred,
        })

        psnr_in_list.append(psnr_in)
        ssim_in_list.append(ssim_in)
        psnr_pred_list.append(psnr_pred)
        ssim_pred_list.append(ssim_pred)

    # Write CSV
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "psnr_input_srgb",
                "ssim_input_srgb",
                "psnr_pred_srgb",
                "ssim_pred_srgb",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summary statistics
    mean_psnr_in   = float(np.mean(psnr_in_list))   if psnr_in_list   else float("nan")
    mean_ssim_in   = float(np.mean(ssim_in_list))   if ssim_in_list   else float("nan")
    mean_psnr_pred = float(np.mean(psnr_pred_list)) if psnr_pred_list else float("nan")
    mean_ssim_pred = float(np.mean(ssim_pred_list)) if ssim_pred_list else float("nan")

    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Split: {split_name}\n")
        f.write(f"Num images: {len(psnr_pred_list)}\n\n")
        f.write(f"Mean PSNR (Input vs GT, sRGB): {mean_psnr_in:.4f} dB\n")
        f.write(f"Mean SSIM (Input vs GT, sRGB): {mean_ssim_in:.4f}\n")
        f.write(f"Mean PSNR (Pred vs GT, sRGB):  {mean_psnr_pred:.4f} dB\n")
        f.write(f"Mean SSIM (Pred vs GT, sRGB):  {mean_ssim_pred:.4f}\n")

    print(f"\n--- [eval_full_srgb] {model_name} / {split_name} ---")
    print(f"Mean PSNR (Input vs GT, sRGB): {mean_psnr_in:.4f} dB")
    print(f"Mean SSIM (Input vs GT, sRGB): {mean_ssim_in:.4f}")
    print(f"Mean PSNR (Pred vs GT, sRGB):  {mean_psnr_pred:.4f} dB")
    print(f"Mean SSIM (Pred vs GT, sRGB):  {mean_ssim_pred:.4f}")
    print(f"Saved CSV to {metrics_csv_path}")
    print(f"Saved summary to {summary_path}")

    # Copy to Drive
    import shutil
    drive_model_dir = os.path.join(drive_results_root, f"{model_name}_{split_name}")
    os.makedirs(drive_model_dir, exist_ok=True)
    shutil.copy(metrics_csv_path, os.path.join(drive_model_dir, "metrics_srgb.csv"))
    shutil.copy(summary_path,     os.path.join(drive_model_dir, "metrics_srgb_summary.txt"))
    print(f"Copied sRGB metrics to Drive: {drive_model_dir}")


def main():
    if not os.path.exists(BACKGROUND_DNG_PATH):
        raise FileNotFoundError(
            f"background.dng not found at {BACKGROUND_DNG_PATH}. "
            "Copy it here or update BACKGROUND_DNG_PATH."
        )

    os.makedirs(RESULTS_SRGB_ROOT, exist_ok=True)
    os.makedirs(DRIVE_RESULTS_SRGB_ROOT, exist_ok=True)

    splits = {
        "val":  VAL_DIR,
        "test": TEST_DIR,
    }

    for model_name in MODEL_NAMES:
        for split_name, data_dir in splits.items():
            if not data_dir or not os.path.isdir(data_dir):
                print(f"[eval_full_srgb] Split '{split_name}' dir missing ({data_dir}), skipping.")
                continue

            evaluate_srgb_for_model_and_split(
                model_name=model_name,
                split_name=split_name,
                data_dir=data_dir,
                pred_root=PRED_ROOT,
                background_path=BACKGROUND_DNG_PATH,
                results_root=RESULTS_SRGB_ROOT,
                drive_results_root=DRIVE_RESULTS_SRGB_ROOT,
            )


if __name__ == "__main__":
    main()
