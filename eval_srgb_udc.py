# File: eval_srgb_udc.py
import os
import argparse
import csv
from glob import glob

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def fourch_to_rgb(arr_4ch: np.ndarray) -> np.ndarray:
    """
    Convert a 4-channel UDC-SIT array to a simple pseudo-RGB image in [0,1].

    Assumes arr_4ch is (H, W, 4) with values either in [0,1023] or [0,1].
    Uses a simple mapping:
        R = C0
        G = 0.5*(C1 + C2)
        B = C3
    """
    if arr_4ch.ndim != 3 or arr_4ch.shape[2] != 4:
        raise ValueError(f"Expected (H,W,4) array, got {arr_4ch.shape}")

    # Normalize to [0,1] if needed
    if arr_4ch.max() > 2.0:
        arr = arr_4ch.astype(np.float32) / 1023.0
    else:
        arr = arr_4ch.astype(np.float32)

    r = arr[:, :, 0]
    g = 0.5 * (arr[:, :, 1] + arr[:, :, 2])
    b = arr[:, :, 3]

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def apply_white_balance(rgb: np.ndarray, mode: str = "none") -> np.ndarray:
    """
    Simple white-balance correction applied to an sRGB image in [0,1].

    mode:
        - "none": no change
        - "gray": Gray-world assumption (equalize per-channel means)
    """
    if mode == "none":
        return rgb

    if mode == "gray":
        h, w, c = rgb.shape
        flat = rgb.reshape(-1, c)
        means = flat.mean(axis=0) + 1e-6
        gray = means.mean()
        scale = gray / means
        balanced = rgb * scale[None, None, :]
        return np.clip(balanced, 0.0, 1.0)

    # Unknown mode, just return original
    return rgb


def apply_gamma(rgb: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to an sRGB image in [0,1].

    If gamma == 1.0, returns input unchanged.
    """
    if gamma is None or abs(gamma - 1.0) < 1e-6:
        return rgb
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.power(rgb, 1.0 / gamma)


def compute_psnr_ssim(gt_rgb: np.ndarray, pred_rgb: np.ndarray):
    """
    Compute PSNR and SSIM for two sRGB images in [0,1].

    Both arrays are (H,W,3).
    """
    if gt_rgb.shape != pred_rgb.shape:
        raise ValueError(f"Shape mismatch: {gt_rgb.shape} vs {pred_rgb.shape}")

    psnr = peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=1.0)
    ssim = structural_similarity(gt_rgb, pred_rgb, data_range=1.0, channel_axis=-1)
    return psnr, ssim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate teacher/student models on UDC-SIT in sRGB space (pseudo-ISP)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/dataset/UDC-SIT/UDC-SIT",
        help="Root of UDC-SIT dataset containing train/validation/testing folders.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "testing"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--teacher-pred-dir",
        type=str,
        default="results_quick/teacher_full_validation/npy",
        help="Directory containing teacher 4-ch .npy predictions.",
    )
    parser.add_argument(
        "--student-pred-dir",
        type=str,
        default="results_quick/student_full_validation/npy",
        help="Directory containing student 4-ch .npy predictions.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="If set, limit the number of images evaluated.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results_srgb_metrics",
        help="Local directory to store sRGB metric CSV + summary.",
    )
    parser.add_argument(
        "--drive-results-root",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/results_srgb_metrics",
        help="Google Drive directory to mirror sRGB metrics. Leave empty to disable.",
    )
    parser.add_argument(
        "--wb-mode",
        type=str,
        default="none",
        choices=["none", "gray"],
        help="White balance mode applied to all sRGB images before metrics.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction applied to all sRGB images before metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Included only to mirror training/testing scripts; not used here.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = os.path.join(args.data_root, args.split, "input")
    gt_dir = os.path.join(args.data_root, args.split, "GT")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"[eval_srgb_udc] Missing input dir: {input_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"[eval_srgb_udc] Missing GT dir: {gt_dir}")

    os.makedirs(args.results_root, exist_ok=True)
    if args.drive_results_root is not None and len(args.drive_results_root) > 0:
        os.makedirs(args.drive_results_root, exist_ok=True)

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
    print("=================================")

    gt_files = sorted(glob(os.path.join(gt_dir, "*.npy")))
    if args.max_images is not None:
        gt_files = gt_files[: args.max_images]

    teacher_dir = args.teacher_pred_dir
    student_dir = args.student_pred_dir

    metrics_rows = []

    for gt_path in gt_files:
        fname = os.path.basename(gt_path)
        inp_path = os.path.join(input_dir, fname)

        if not os.path.exists(inp_path):
            print(f"[eval_srgb_udc] WARNING: Missing input for {fname}, skipping.")
            continue

        # Teacher and student predictions may or may not exist
        teacher_path = (
            os.path.join(teacher_dir, fname) if teacher_dir and os.path.isdir(teacher_dir) else None
        )
        if teacher_path and not os.path.exists(teacher_path):
            teacher_path = None

        student_path = (
            os.path.join(student_dir, fname) if student_dir and os.path.isdir(student_dir) else None
        )
        if student_path and not os.path.exists(student_path):
            student_path = None

        # Load 4-ch arrays
        gt_arr = np.load(gt_path)  # (H,W,4) in [0,1023]
        inp_arr = np.load(inp_path)

        # Convert to sRGB
        inp_rgb = fourch_to_rgb(inp_arr)
        gt_rgb = fourch_to_rgb(gt_arr)

        teacher_rgb = None
        if teacher_path is not None:
            teacher_arr = np.load(teacher_path)
            teacher_rgb = fourch_to_rgb(teacher_arr)

        student_rgb = None
        if student_path is not None:
            student_arr = np.load(student_path)
            student_rgb = fourch_to_rgb(student_arr)

        # Apply WB + gamma
        inp_rgb = apply_gamma(apply_white_balance(inp_rgb, args.wb_mode), args.gamma)
        gt_rgb = apply_gamma(apply_white_balance(gt_rgb, args.wb_mode), args.gamma)

        if teacher_rgb is not None:
            teacher_rgb = apply_gamma(
                apply_white_balance(teacher_rgb, args.wb_mode), args.gamma
            )
        if student_rgb is not None:
            student_rgb = apply_gamma(
                apply_white_balance(student_rgb, args.wb_mode), args.gamma
            )

        # Metrics: input vs GT, teacher vs GT, student vs GT
        psnr_in, ssim_in = compute_psnr_ssim(gt_rgb, inp_rgb)
        psnr_teacher, ssim_teacher = (np.nan, np.nan)
        psnr_student, ssim_student = (np.nan, np.nan)

        if teacher_rgb is not None:
            psnr_teacher, ssim_teacher = compute_psnr_ssim(gt_rgb, teacher_rgb)
        if student_rgb is not None:
            psnr_student, ssim_student = compute_psnr_ssim(gt_rgb, student_rgb)

        metrics_rows.append(
            {
                "filename": fname,
                "psnr_input": psnr_in,
                "ssim_input": ssim_in,
                "psnr_teacher": psnr_teacher,
                "ssim_teacher": ssim_teacher,
                "psnr_student": psnr_student,
                "ssim_student": ssim_student,
            }
        )

    # Save CSV + summary
    csv_path = os.path.join(args.results_root, f"{args.split}_metrics_srgb.csv")
    summary_path = os.path.join(args.results_root, f"{args.split}_metrics_srgb_summary.txt")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "psnr_input",
                "ssim_input",
                "psnr_teacher",
                "ssim_teacher",
                "psnr_student",
                "ssim_student",
            ],
        )
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    # Summary: mean metrics
    def safe_mean(vals):
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    mean_psnr_in = safe_mean([r["psnr_input"] for r in metrics_rows])
    mean_ssim_in = safe_mean([r["ssim_input"] for r in metrics_rows])

    mean_psnr_teacher = safe_mean([r["psnr_teacher"] for r in metrics_rows])
    mean_ssim_teacher = safe_mean([r["ssim_teacher"] for r in metrics_rows])

    mean_psnr_student = safe_mean([r["psnr_student"] for r in metrics_rows])
    mean_ssim_student = safe_mean([r["ssim_student"] for r in metrics_rows])

    with open(summary_path, "w") as f:
        f.write(f"Split: {args.split}\n")
        f.write(f"Num images: {len(metrics_rows)}\n\n")
        f.write("Input vs GT:\n")
        f.write(f"  Mean PSNR: {mean_psnr_in:.4f} dB\n")
        f.write(f"  Mean SSIM: {mean_ssim_in:.4f}\n\n")
        f.write("Teacher vs GT:\n")
        f.write(f"  Mean PSNR: {mean_psnr_teacher:.4f} dB\n")
        f.write(f"  Mean SSIM: {mean_ssim_teacher:.4f}\n\n")
        f.write("Student vs GT:\n")
        f.write(f"  Mean PSNR: {mean_psnr_student:.4f} dB\n")
        f.write(f"  Mean SSIM: {mean_ssim_student:.4f}\n")

    print(f"[eval_srgb_udc] Saved CSV to {csv_path}")
    print(f"[eval_srgb_udc] Saved summary to {summary_path}")
    print("Summary (mean metrics):")
    print(f"  Input   -> PSNR: {mean_psnr_in:.4f} dB, SSIM: {mean_ssim_in:.4f}")
    print(f"  Teacher -> PSNR: {mean_psnr_teacher:.4f} dB, SSIM: {mean_ssim_teacher:.4f}")
    print(f"  Student -> PSNR: {mean_psnr_student:.4f} dB, SSIM: {mean_ssim_student:.4f}")

    # Copy to Drive
    if args.drive_results_root is not None and len(args.drive_results_root) > 0:
        import shutil

        os.makedirs(args.drive_results_root, exist_ok=True)
        drive_csv = os.path.join(args.drive_results_root, os.path.basename(csv_path))
        drive_summary = os.path.join(args.drive_results_root, os.path.basename(summary_path))
        shutil.copy(csv_path, drive_csv)
        shutil.copy(summary_path, drive_summary)
        print(f"[eval_srgb_udc] Copied results to Drive: {args.drive_results_root}")


if __name__ == "__main__":
    main()
