# File: eval_srgb_udc.py
"""
Evaluate sRGB-domain metrics for UDC-SIT.

We assume you already ran visualize_sit.py (or equivalent) to produce
sRGB images for:
  - Input  (baseline)
  - GT     (reference)
  - Teacher predictions (optional)
  - Student predictions (optional)

Directory layout example:

  DATA_ROOT/
    validation/
      input/        *.png (or .jpg/.jpeg)
      GT/           *.png
    testing/
      input/
      GT/

Teacher / student predictions (sRGB) can be anywhere, as long as filenames
match those in GT/, e.g.:

  /content/results_full_srgb_images/teacher_validation/*.png
  /content/results_full_srgb_images/student_validation/*.png

This script:
  - Loads GT and corresponding predictions
  - Optionally applies global white-balance & gamma
  - Computes PSNR, SSIM, and LPIPS
  - Writes metrics CSV + summary
  - Optionally mirrors them into Google Drive
"""

import os
import sys
import argparse
import shutil
from glob import glob

import numpy as np
import imageio.v2 as imageio

from tqdm.auto import tqdm

import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ----------------- Helpers ----------------- #

def list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    return sorted(files)


def load_image_float(path):
    """
    Load image as float32 in [0,1], shape (H, W, 3).
    """
    img = imageio.imread(path)
    if img.ndim == 2:
        # gray → 3-ch
        img = np.stack([img] * 3, axis=-1)

    img = img.astype(np.float32)

    if img.max() > 1.0:
        img /= 255.0

    if img.shape[2] > 3:
        img = img[:, :, :3]

    return np.clip(img, 0.0, 1.0)


def apply_white_balance(img, mode="none"):
    """
    img: (H, W, 3) float32 [0,1]
    mode: "none" or "grayworld"
    """
    if mode == "none":
        return img

    if mode == "grayworld":
        # Gray-world: scale each channel so its mean = global gray mean.
        eps = 1e-6
        means = img.reshape(-1, 3).mean(axis=0) + eps  # (3,)
        gray_mean = means.mean()
        scale = gray_mean / means  # (3,)
        img_wb = img * scale[None, None, :]
        return np.clip(img_wb, 0.0, 1.0)

    raise ValueError(f"Unknown wb-mode: {mode}")


def apply_gamma(img, gamma=1.0):
    """
    img: (H, W, 3) float32 [0,1]
    gamma: >0, 1.0 = no change

    We treat img as already roughly in sRGB-like space, so we use a simple
    power-law correction: img_out = img^(1/gamma).
    """
    if abs(gamma - 1.0) < 1e-6:
        return img
    img = np.clip(img, 0.0, 1.0)
    return np.power(img, 1.0 / gamma)


def align_images(gt, pred):
    """
    Crop both images to the common minimum H, W (safety in case of small mismatches).
    """
    H = min(gt.shape[0], pred.shape[0])
    W = min(gt.shape[1], pred.shape[1])
    gt_c = gt[:H, :W, :]
    pr_c = pred[:H, :W, :]
    return gt_c, pr_c


def compute_metrics_srgb(gt_img, pred_img, lpips_model, device):
    """
    gt_img, pred_img: (H, W, 3) float32 [0,1]
    Returns: psnr, ssim, lpips_val
    """
    gt_img, pred_img = align_images(gt_img, pred_img)

    # PSNR / SSIM
    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=1.0)
    ssim = structural_similarity(gt_img, pred_img, data_range=1.0, channel_axis=-1)

    # LPIPS
    gt_t = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    pr_t = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0).to(device)

    gt_t_lp = gt_t * 2 - 1
    pr_t_lp = pr_t * 2 - 1

    with torch.no_grad():
        lp = lpips_model(pr_t_lp, gt_t_lp).item()

    return psnr, ssim, lp


def evaluate_pair_list(
    label,
    gt_files,
    pred_dir,
    lpips_model,
    device,
    results_root,
    wb_mode="none",
    gamma=1.0,
    input_fallback_dir=None,
):
    """
    label: "input", "teacher", or "student"
    gt_files: list of GT image paths
    pred_dir: directory with prediction images having same filenames as GT
              (ignored if label == "input" and input_fallback_dir is provided)
    input_fallback_dir: if label == "input", look here for baseline images

    Returns: (csv_path, summary_path)
    """
    os.makedirs(results_root, exist_ok=True)
    csv_path = os.path.join(results_root, f"metrics_srgb_{label}.csv")
    summary_path = os.path.join(results_root, f"metrics_srgb_{label}_summary.txt")

    rows = []
    psnrs, ssims, lpips_vals = [], [], []

    print(f"\n=== [eval_srgb_udc] Evaluating '{label}' ===")
    for gt_path in tqdm(gt_files, desc=f"[{label}] images"):
        base = os.path.basename(gt_path)

        # GT image
        gt_img = load_image_float(gt_path)

        # Pred image choice
        if label == "input" and input_fallback_dir is not None:
            pred_path = os.path.join(input_fallback_dir, base)
        else:
            if pred_dir is None:
                print(f"  WARNING: pred_dir is None for label={label}, skipping {base}")
                continue
            pred_path = os.path.join(pred_dir, base)

        if not os.path.exists(pred_path):
            print(f"  WARNING: prediction not found ({label}): {pred_path}, skipping.")
            continue

        pred_img = load_image_float(pred_path)

        # WB + gamma (same transform on GT & pred)
        gt_proc = apply_gamma(apply_white_balance(gt_img, mode=wb_mode), gamma=gamma)
        pr_proc = apply_gamma(apply_white_balance(pred_img, mode=wb_mode), gamma=gamma)

        psnr, ssim, lp = compute_metrics_srgb(gt_proc, pr_proc, lpips_model, device)

        rows.append({
            "filename": base,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lp,
        })
        psnrs.append(psnr)
        ssims.append(ssim)
        lpips_vals.append(lp)

    if not rows:
        print(f"[eval_srgb_udc] No valid image pairs found for label={label}.")
        return csv_path, summary_path

    # Write CSV
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "lpips"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    mean_lp   = float(np.mean(lpips_vals))

    with open(summary_path, "w") as f:
        f.write(f"Label: {label}\n")
        f.write(f"Num images: {len(psnrs)}\n")
        f.write(f"WB mode: {wb_mode}\n")
        f.write(f"Gamma:   {gamma:.3f}\n\n")
        f.write(f"Mean PSNR: {mean_psnr:.4f} dB\n")
        f.write(f"Mean SSIM: {mean_ssim:.4f}\n")
        f.write(f"Mean LPIPS: {mean_lp:.4f}\n")

    print(f"\n--- [eval_srgb_udc] {label} results ---")
    print(f"Mean PSNR:  {mean_psnr:.4f} dB")
    print(f"Mean SSIM:  {mean_ssim:.4f}")
    print(f"Mean LPIPS: {mean_lp:.4f}")
    print(f"Saved CSV to:     {csv_path}")
    print(f"Saved summary to: {summary_path}")

    return csv_path, summary_path


# ----------------- Main ----------------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluate sRGB metrics for UDC-SIT.")

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/UDC-SIT_srgb",
        help="Root folder with sRGB images (contains split/input and split/GT).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split (e.g., validation, testing).",
    )
    parser.add_argument(
        "--teacher-pred-dir",
        type=str,
        default=None,
        help="Directory with teacher sRGB predictions (filenames match GT).",
    )
    parser.add_argument(
        "--student-pred-dir",
        type=str,
        default=None,
        help="Directory with student sRGB predictions (filenames match GT).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max number of images to evaluate (for quick tests).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results_srgb_eval",
        help="Where to store evaluation CSVs and summaries.",
    )
    parser.add_argument(
        "--drive-results-root",
        type=str,
        default=None,
        help="If set, mirror CSV + summary into this Drive folder.",
    )
    parser.add_argument(
        "--wb-mode",
        type=str,
        default="none",
        help="White-balance mode: 'none' or 'grayworld'.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction factor (1.0 = no change).",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== [eval_srgb_udc] Config ===")
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

    # Directories
    split_root = os.path.join(args.data_root, args.split)
    gt_dir     = os.path.join(split_root, "GT")
    input_dir  = os.path.join(split_root, "input")

    assert os.path.isdir(gt_dir), f"GT dir not found: {gt_dir}"
    if not os.path.isdir(input_dir):
        print(f"[eval_srgb_udc] WARNING: input dir not found: {input_dir} (baseline 'input' will be skipped).")

    gt_files = list_images(gt_dir)
    if args.max_images is not None:
        gt_files = gt_files[:args.max_images]

    if len(gt_files) == 0:
        print(f"[eval_srgb_udc] No GT images found in {gt_dir}")
        sys.exit(1)

    os.makedirs(args.results_root, exist_ok=True)

    # LPIPS model
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    csv_paths = []
    summary_paths = []

    # 1) Baseline input vs GT (if input/ exists)
    if os.path.isdir(input_dir):
        out_dir_input = os.path.join(args.results_root, f"{args.split}_input")
        csv_i, sum_i = evaluate_pair_list(
            label="input",
            gt_files=gt_files,
            pred_dir=None,
            lpips_model=lpips_model,
            device=device,
            results_root=out_dir_input,
            wb_mode=args.wb_mode,
            gamma=args.gamma,
            input_fallback_dir=input_dir,
        )
        csv_paths.append((csv_i, "input"))
        summary_paths.append((sum_i, "input"))

    # 2) Teacher
    if args.teacher_pred_dir is not None:
        out_dir_teacher = os.path.join(args.results_root, f"{args.split}_teacher")
        csv_t, sum_t = evaluate_pair_list(
            label="teacher",
            gt_files=gt_files,
            pred_dir=args.teacher_pred_dir,
            lpips_model=lpips_model,
            device=device,
            results_root=out_dir_teacher,
            wb_mode=args.wb_mode,
            gamma=args.gamma,
        )
        csv_paths.append((csv_t, "teacher"))
        summary_paths.append((sum_t, "teacher"))
    else:
        print("[eval_srgb_udc] No teacher_pred_dir provided; skipping teacher evaluation.")

    # 3) Student
    if args.student_pred_dir is not None:
        out_dir_student = os.path.join(args.results_root, f"{args.split}_student")
        csv_s, sum_s = evaluate_pair_list(
            label="student",
            gt_files=gt_files,
            pred_dir=args.student_pred_dir,
            lpips_model=lpips_model,
            device=device,
            results_root=out_dir_student,
            wb_mode=args.wb_mode,
            gamma=args.gamma,
        )
        csv_paths.append((csv_s, "student"))
        summary_paths.append((sum_s, "student"))
    else:
        print("[eval_srgb_udc] No student_pred_dir provided; skipping student evaluation.")

    # Mirror to Drive if requested
    if args.drive_results_root is not None:
        os.makedirs(args.drive_results_root, exist_ok=True)
        for csv_path, label in csv_paths:
            if not os.path.exists(csv_path):
                continue
            base_dir = os.path.dirname(csv_path)
            drive_dir = os.path.join(
                args.drive_results_root,
                f"{args.split}_{label}"
            )
            os.makedirs(drive_dir, exist_ok=True)
            # copy CSV + summary
            shutil.copy(csv_path, drive_dir)
            # find summary
            matching_summaries = [s for s, lab in summary_paths if lab == label]
            for sp in matching_summaries:
                if os.path.exists(sp):
                    shutil.copy(sp, drive_dir)
        print(f"[eval_srgb_udc] Mirrored results to Drive root: {args.drive_results_root}")


if __name__ == "__main__":
    main()
