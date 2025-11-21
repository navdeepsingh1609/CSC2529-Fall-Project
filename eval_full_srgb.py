# File: eval_full_srgb.py
"""
Evaluate sRGB outputs (PNG/JPG) for UDC-SIT:

- Computes PSNR / SSIM on sRGB images.
- Computes LPIPS (VGG) on sRGB.
- Optional gray-world white balance on predictions.
- Works on intersection of filenames between GT and prediction dirs.

Usage example:

python eval_full_srgb.py \
    --gt-dir /content/dataset/UDC-SIT_srgb/validation/gt \
    --pred-dir /content/results_full_srgb_images/student_validation \
    --out-prefix student_full_val \
    --wb-grayworld 0 \
    --max-images 50

You MUST first generate sRGB images using visualize_sit.py, e.g.:

python utils/visualize_sit.py \
    --data-directory /content/dataset/UDC-SIT/validation/gt \
    --result-directory /content/dataset/UDC-SIT_srgb/validation/gt

python utils/visualize_sit.py \
    --data-directory /content/results_full/student_full_validation/npy \
    --result-directory /content/results_full_srgb_images/student_validation
"""

import os
import argparse
import numpy as np
from glob import glob

import imageio.v2 as imageio
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
import lpips


# LPIPS on CPU by default to avoid device headaches
LPIPS_DEVICE = "cpu"


def read_srgb_image(path):
    """
    Read an sRGB image (PNG/JPG) and return float32 array in [0,1], shape (H,W,3).
    Handles uint8, uint16, and grayscale images.
    """
    img = imageio.imread(path)

    # If grayscale, replicate channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # Sometimes 4-channel PNG (RGBA) — drop alpha
    if img.shape[-1] == 4:
        img = img[..., :3]

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        # Assume already float in [0,1] or similar
        img = img.astype(np.float32)

    img = np.clip(img, 0.0, 1.0)
    return img  # (H,W,3) float32


def gray_world_white_balance(img):
    """
    Simple gray-world white balance on sRGB image.

    Args:
        img: (H,W,3) float32 in [0,1]

    Returns:
        corrected image, same shape, float32 in [0,1]
    """
    H, W, C = img.shape
    assert C == 3, "Expect 3-channel image for gray-world WB."

    means = img.reshape(-1, 3).mean(axis=0) + 1e-6  # avoid div by zero
    mean_gray = means.mean()
    scale = mean_gray / means

    corrected = img * scale[None, None, :]
    corrected = np.clip(corrected, 0.0, 1.0)
    return corrected


def compute_metrics_srgb(gt_img, pred_img, lpips_model):
    """
    Compute PSNR, SSIM, LPIPS on sRGB.

    Args:
        gt_img:   (H,W,3) float32 in [0,1]
        pred_img: (H,W,3) float32 in [0,1]
        lpips_model: LPIPS object on LPIPS_DEVICE

    Returns:
        psnr, ssim, lpips_val
    """
    # PSNR / SSIM with skimage on numpy
    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=1.0)
    ssim = structural_similarity(gt_img, pred_img, data_range=1.0, channel_axis=-1)

    # LPIPS on torch
    gt_t = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    pr_t = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0)

    # Scale [0,1] -> [-1,1]
    gt_t = (gt_t * 2 - 1).to(LPIPS_DEVICE)
    pr_t = (pr_t * 2 - 1).to(LPIPS_DEVICE)

    with torch.no_grad():
        lp = lpips_model(pr_t, gt_t).item()

    return psnr, ssim, lp


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sRGB outputs for UDC-SIT.")
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="Directory with GT sRGB images (PNG/JPG).",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory with predicted sRGB images (PNG/JPG).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="eval",
        help="Prefix for output files (CSV + summary).",
    )
    parser.add_argument(
        "--wb-grayworld",
        type=int,
        default=0,
        help="If 1, apply gray-world white balance on predictions before metrics.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max number of images to evaluate (for quick debug).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_full_srgb_eval",
        help="Directory to save CSV and summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    out_prefix = args.out_prefix
    apply_wb = bool(args.wb_grayworld)
    max_images = args.max_images
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # Collect filenames
    gt_files = sorted(
        [os.path.basename(p) for p in glob(os.path.join(gt_dir, "*"))]
    )
    pred_files = sorted(
        [os.path.basename(p) for p in glob(os.path.join(pred_dir, "*"))]
    )

    intersect = sorted(list(set(gt_files).intersection(set(pred_files))))
    if max_images is not None:
        intersect = intersect[:max_images]

    if len(intersect) == 0:
        print("No common filenames between GT and prediction dirs. Check paths.")
        return

    print(f"[eval_full_srgb] GT dir:   {gt_dir}")
    print(f"[eval_full_srgb] Pred dir: {pred_dir}")
    print(f"[eval_full_srgb] Num common images: {len(intersect)}")
    if apply_wb:
        print("[eval_full_srgb] Gray-world white balance: ON (predictions only)")
    else:
        print("[eval_full_srgb] Gray-world white balance: OFF")

    # LPIPS model
    lpips_model = lpips.LPIPS(net='vgg').to(LPIPS_DEVICE)
    print(f"[eval_full_srgb] LPIPS running on: {LPIPS_DEVICE}")

    # Prepare accumulators
    rows = []
    psnr_list = []
    ssim_list = []
    lpips_list = []

    for fname in tqdm(intersect, desc="Evaluating sRGB"):
        gt_path = os.path.join(gt_dir, fname)
        pr_path = os.path.join(pred_dir, fname)

        gt_img = read_srgb_image(gt_path)
        pr_img = read_srgb_image(pr_path)

        # Optional gray-world WB on prediction
        if apply_wb:
            pr_img = gray_world_white_balance(pr_img)

        psnr, ssim, lpv = compute_metrics_srgb(gt_img, pr_img, lpips_model)

        rows.append({
            "filename": fname,
            "psnr_srgb": psnr,
            "ssim_srgb": ssim,
            "lpips_srgb": lpv,
        })
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpv)

    # Save CSV
    csv_path = os.path.join(out_dir, f"{out_prefix}_metrics.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr_srgb", "ssim_srgb", "lpips_srgb"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summary
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_lpips = float(np.mean(lpips_list))

    summary_path = os.path.join(out_dir, f"{out_prefix}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"GT dir:   {gt_dir}\n")
        f.write(f"Pred dir: {pred_dir}\n")
        f.write(f"Num images: {len(intersect)}\n")
        f.write(f"Gray-world WB: {apply_wb}\n\n")
        f.write(f"Mean PSNR (sRGB): {mean_psnr:.4f} dB\n")
        f.write(f"Mean SSIM (sRGB): {mean_ssim:.4f}\n")
        f.write(f"Mean LPIPS (sRGB): {mean_lpips:.4f}\n")

    print("\n[eval_full_srgb] Done.")
    print(f"Saved CSV to:     {csv_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
