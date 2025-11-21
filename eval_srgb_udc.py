# File: eval_srgb_udc.py
"""
Evaluate UDC-SIT models in both Bayer and sRGB space directly from 4-channel .npy files.

- Assumes data_root has splits like:
    data_root/
        training/
            input/*.npy
            GT/*.npy
        validation/
            input/*.npy
            GT/*.npy
        testing/
            input/*.npy
            GT/*.npy   (if GT available)

- Assumes testing_udc.py produced prediction .npy files in:
    results_quick/teacher_full_<split>/npy/*.npy
    results_quick/student_full_<split>/npy/*.npy

This script:
- Loads GT + input + (optionally) teacher & student predictions (4ch .npy).
- Normalizes everything to [0,1].
- Computes:
    * PSNR / SSIM in Bayer (4-ch).
    * PSNR / SSIM / LPIPS in sRGB (using a simple Bayer→RGB ISP).
- Writes a single CSV with all metrics and a text summary.
- Optionally copies results to Google Drive.

Usage example (after testing_udc.py):

python eval_srgb_udc.py \
  --data-root /content/dataset/UDC-SIT \
  --split validation \
  --teacher-pred-dir results_quick/teacher_full_validation/npy \
  --student-pred-dir results_quick/student_full_validation/npy \
  --max-images 200 \
  --results-root results_srgb_metrics \
  --drive-results-root "/content/drive/MyDrive/Computational Imaging Project/results_srgb_metrics" \
  --wb-mode channel_gains \
  --gamma 2.2
"""

import os
import argparse
import csv
from glob import glob

import numpy as np
import torch
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------- Bayer4 -> sRGB conversion ---------------- #

def bayer4_to_srgb(
    bayer4: np.ndarray,
    wb_mode: str = "none",
    gamma: float = 1.0,
    r_gain: float = 1.9,
    b_gain: float = 1.9,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Convert a (H, W, 4) Bayer-like UDC-SIT patch in [0,1] to an sRGB-ish (H, W, 3) image in [0,1].

    Channels are assumed to be:
      0: GR, 1: R, 2: B, 3: GB

    wb_mode:
      - "none":    no extra WB, just green from GR+GB and simple gains.
      - "grayworld": gray-world white balance using channel means.
      - "channel_gains": apply fixed r_gain / b_gain (keeps G as is).

    gamma:
      - 1.0   -> linear output
      - 2.2   -> simple display gamma (common choice).
    """
    assert bayer4.ndim == 3 and bayer4.shape[2] == 4, f"Expected (H,W,4), got {bayer4.shape}"

    # Ensure float32 and clip to [0,1]
    x = np.clip(bayer4.astype(np.float32), 0.0, 1.0)

    GR = x[..., 0]
    R  = x[..., 1]
    B  = x[..., 2]
    GB = x[..., 3]

    G = 0.5 * (GR + GB)

    Rn = R.copy()
    Gn = G.copy()
    Bn = B.copy()

    if wb_mode == "grayworld":
        meanR = Rn.mean() + eps
        meanG = Gn.mean() + eps
        meanB = Bn.mean() + eps
        gray  = (meanR + meanG + meanB) / 3.0

        Rn *= gray / meanR
        Gn *= gray / meanG
        Bn *= gray / meanB

    elif wb_mode == "channel_gains":
        Rn = Rn * r_gain
        Bn = Bn * b_gain
        # G untouched

    # Stack and gamma-encode
    rgb = np.stack([Rn, Gn, Bn], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)

    if gamma != 1.0:
        rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / gamma)

    return rgb


def compute_metrics_for_pair(
    gt_4: np.ndarray,
    pred_4: np.ndarray,
    lpips_model: lpips.LPIPS,
    device: torch.device,
    wb_mode: str,
    gamma: float,
):
    """
    Compute Bayer and sRGB metrics between GT and prediction.

    gt_4, pred_4: (H, W, 4) numpy arrays in [0,1].
    Returns a dict with:
      psnr_bayer, ssim_bayer,
      psnr_srgb, ssim_srgb,
      lpips_srgb
    """
    # Safety clamp
    gt_4   = np.clip(gt_4,   0.0, 1.0)
    pred_4 = np.clip(pred_4, 0.0, 1.0)

    # Bayer metrics
    psnr_bayer = peak_signal_noise_ratio(gt_4, pred_4, data_range=1.0)
    ssim_bayer = structural_similarity(gt_4, pred_4, data_range=1.0, channel_axis=-1)

    # sRGB metrics
    gt_srgb   = bayer4_to_srgb(gt_4,   wb_mode=wb_mode, gamma=gamma)
    pred_srgb = bayer4_to_srgb(pred_4, wb_mode=wb_mode, gamma=gamma)

    psnr_srgb = peak_signal_noise_ratio(gt_srgb, pred_srgb, data_range=1.0)
    ssim_srgb = structural_similarity(gt_srgb, pred_srgb, data_range=1.0, channel_axis=-1)

    # LPIPS on sRGB
    gt_t   = torch.from_numpy(gt_srgb).permute(2, 0, 1).unsqueeze(0).to(device).float()
    pred_t = torch.from_numpy(pred_srgb).permute(2, 0, 1).unsqueeze(0).to(device).float()

    # [0,1] -> [-1,1]
    gt_t   = gt_t * 2.0 - 1.0
    pred_t = pred_t * 2.0 - 1.0

    with torch.no_grad():
        lpv = float(lpips_model(pred_t, gt_t).item())

    return {
        "psnr_bayer": psnr_bayer,
        "ssim_bayer": ssim_bayer,
        "psnr_srgb":  psnr_srgb,
        "ssim_srgb":  ssim_srgb,
        "lpips_srgb": lpv,
    }


# ---------------- Main script ---------------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluate UDC-SIT models in sRGB + Bayer space from 4-ch .npy.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/dataset/UDC-SIT",
        help="Root directory containing splits (training/validation/testing) with 4-ch .npy files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["training", "validation", "testing"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--teacher-pred-dir",
        type=str,
        default=None,
        help="Directory with teacher prediction .npy files (4-ch, matching GT filenames).",
    )
    parser.add_argument(
        "--student-pred-dir",
        type=str,
        default=None,
        help="Directory with student prediction .npy files (4-ch, matching GT filenames).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to evaluate (None = all).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results_srgb_metrics",
        help="Where to save CSV + summary locally.",
    )
    parser.add_argument(
        "--drive-results-root",
        type=str,
        default=None,
        help="Optional: directory on Google Drive to copy CSV + summary.",
    )
    parser.add_argument(
        "--wb-mode",
        type=str,
        default="none",
        choices=["none", "grayworld", "channel_gains"],
        help="White-balance mode for sRGB conversion.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma for sRGB conversion (e.g., 2.2).",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_dir = os.path.join(args.data_root, args.split)
    gt_dir    = os.path.join(split_dir, "GT")
    inp_dir   = os.path.join(split_dir, "input")

    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"[eval_srgb_udc] GT dir not found: {gt_dir}")
    if not os.path.isdir(inp_dir):
        raise FileNotFoundError(f"[eval_srgb_udc] input dir not found: {inp_dir}")

    gt_files = sorted(glob(os.path.join(gt_dir, "*.npy")))
    if args.max_images is not None:
        gt_files = gt_files[:args.max_images]

    if not gt_files:
        print(f"[eval_srgb_udc] No GT .npy files found in {gt_dir}")
        return

    os.makedirs(args.results_root, exist_ok=True)
    if args.drive_results_root is not None:
        os.makedirs(args.drive_results_root, exist_ok=True)

    teacher_dir = args.teacher_pred_dir
    student_dir = args.student_pred_dir

    print("\n=== [eval_srgb_udc] Config ===")
    print(f"Data root:      {args.data_root}")
    print(f"Split:          {args.split}")
    print(f"GT dir:         {gt_dir}")
    print(f"Input dir:      {inp_dir}")
    print(f"Teacher preds:  {teacher_dir}")
    print(f"Student preds:  {student_dir}")
    print(f"Max images:     {args.max_images}")
    print(f"Results root:   {args.results_root}")
    print(f"Drive results:  {args.drive_results_root}")
    print(f"WB mode:        {args.wb_mode}")
    print(f"Gamma:          {args.gamma}")
    print(f"Device:         {device}")
    print("=================================\n")

    # LPIPS model
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    # Output CSV + summary
    csv_path = os.path.join(args.results_root, f"metrics_srgb_{args.split}.csv")
    summary_path = os.path.join(args.results_root, f"metrics_srgb_{args.split}_summary.txt")

    fieldnames = [
        "filename",
        # Input baseline (raw sensor)
        "psnr_in_bayer", "ssim_in_bayer",
        "psnr_in_srgb",  "ssim_in_srgb", "lpips_in_srgb",
        # Teacher
        "psnr_teacher_bayer", "ssim_teacher_bayer",
        "psnr_teacher_srgb",  "ssim_teacher_srgb", "lpips_teacher_srgb",
        # Student
        "psnr_student_bayer", "ssim_student_bayer",
        "psnr_student_srgb",  "ssim_student_srgb", "lpips_student_srgb",
    ]

    rows = []

    # Accumulators for means
    acc = {
        "in_psnr_bayer": [],
        "in_ssim_bayer": [],
        "in_psnr_srgb":  [],
        "in_ssim_srgb":  [],
        "in_lpips":      [],

        "t_psnr_bayer": [],
        "t_ssim_bayer": [],
        "t_psnr_srgb":  [],
        "t_ssim_srgb":  [],
        "t_lpips":      [],

        "s_psnr_bayer": [],
        "s_ssim_bayer": [],
        "s_psnr_srgb":  [],
        "s_ssim_srgb":  [],
        "s_lpips":      [],
    }

    # Main loop
    for gt_path in tqdm(gt_files, desc="[eval_srgb_udc] images"):
        base = os.path.splitext(os.path.basename(gt_path))[0]
        inp_path = os.path.join(inp_dir, base + ".npy")

        if not os.path.exists(inp_path):
            print(f"[eval_srgb_udc] WARNING: Missing input for {base}, skipping.")
            continue

        # Load GT + input, normalize to [0,1] from [0,1023]
        gt_4  = np.load(gt_path).astype(np.float32) / 1023.0
        inp_4 = np.load(inp_path).astype(np.float32) / 1023.0

        if gt_4.shape != inp_4.shape or gt_4.shape[-1] != 4:
            print(f"[eval_srgb_udc] WARNING: Shape mismatch or non-4ch for {base}, skipping.")
            continue

        # Input baseline metrics
        m_in = compute_metrics_for_pair(
            gt_4, inp_4, lpips_model, device, args.wb_mode, args.gamma
        )

        row = {
            "filename": base,
            "psnr_in_bayer": m_in["psnr_bayer"],
            "ssim_in_bayer": m_in["ssim_bayer"],
            "psnr_in_srgb":  m_in["psnr_srgb"],
            "ssim_in_srgb":  m_in["ssim_srgb"],
            "lpips_in_srgb": m_in["lpips_srgb"],
        }

        acc["in_psnr_bayer"].append(m_in["psnr_bayer"])
        acc["in_ssim_bayer"].append(m_in["ssim_bayer"])
        acc["in_psnr_srgb"].append(m_in["psnr_srgb"])
        acc["in_ssim_srgb"].append(m_in["ssim_srgb"])
        acc["in_lpips"].append(m_in["lpips_srgb"])

        # Teacher metrics (if preds provided)
        if teacher_dir is not None and os.path.isdir(teacher_dir):
            t_path = os.path.join(teacher_dir, base + ".npy")
            if os.path.exists(t_path):
                t_4 = np.load(t_path).astype(np.float32)
                # Predictions from testing_udc.py are already in [0,1]
                if t_4.shape[-1] != 4:
                    print(f"[eval_srgb_udc] WARNING: Teacher pred not 4-ch for {base}, skipping teacher metrics.")
                    t_metrics = None
                else:
                    m_t = compute_metrics_for_pair(
                        gt_4, t_4, lpips_model, device, args.wb_mode, args.gamma
                    )
                    row["psnr_teacher_bayer"] = m_t["psnr_bayer"]
                    row["ssim_teacher_bayer"] = m_t["ssim_bayer"]
                    row["psnr_teacher_srgb"]  = m_t["psnr_srgb"]
                    row["ssim_teacher_srgb"]  = m_t["ssim_srgb"]
                    row["lpips_teacher_srgb"] = m_t["lpips_srgb"]

                    acc["t_psnr_bayer"].append(m_t["psnr_bayer"])
                    acc["t_ssim_bayer"].append(m_t["ssim_bayer"])
                    acc["t_psnr_srgb"].append(m_t["psnr_srgb"])
                    acc["t_ssim_srgb"].append(m_t["ssim_srgb"])
                    acc["t_lpips"].append(m_t["lpips_srgb"])
            else:
                # No teacher pred for this file
                row["psnr_teacher_bayer"] = ""
                row["ssim_teacher_bayer"] = ""
                row["psnr_teacher_srgb"]  = ""
                row["ssim_teacher_srgb"]  = ""
                row["lpips_teacher_srgb"] = ""
        else:
            # Teacher dir not provided
            row["psnr_teacher_bayer"] = ""
            row["ssim_teacher_bayer"] = ""
            row["psnr_teacher_srgb"]  = ""
            row["ssim_teacher_srgb"]  = ""
            row["lpips_teacher_srgb"] = ""

        # Student metrics (if preds provided)
        if student_dir is not None and os.path.isdir(student_dir):
            s_path = os.path.join(student_dir, base + ".npy")
            if os.path.exists(s_path):
                s_4 = np.load(s_path).astype(np.float32)
                if s_4.shape[-1] != 4:
                    print(f"[eval_srgb_udc] WARNING: Student pred not 4-ch for {base}, skipping student metrics.")
                else:
                    m_s = compute_metrics_for_pair(
                        gt_4, s_4, lpips_model, device, args.wb_mode, args.gamma
                    )
                    row["psnr_student_bayer"] = m_s["psnr_bayer"]
                    row["ssim_student_bayer"] = m_s["ssim_bayer"]
                    row["psnr_student_srgb"]  = m_s["psnr_srgb"]
                    row["ssim_student_srgb"]  = m_s["ssim_srgb"]
                    row["lpips_student_srgb"] = m_s["lpips_srgb"]

                    acc["s_psnr_bayer"].append(m_s["psnr_bayer"])
                    acc["s_ssim_bayer"].append(m_s["ssim_bayer"])
                    acc["s_psnr_srgb"].append(m_s["psnr_srgb"])
                    acc["s_ssim_srgb"].append(m_s["ssim_srgb"])
                    acc["s_lpips"].append(m_s["lpips_srgb"])
            else:
                row["psnr_student_bayer"] = ""
                row["ssim_student_bayer"] = ""
                row["psnr_student_srgb"]  = ""
                row["ssim_student_srgb"]  = ""
                row["lpips_student_srgb"] = ""
        else:
            row["psnr_student_bayer"] = ""
            row["ssim_student_bayer"] = ""
            row["psnr_student_srgb"]  = ""
            row["ssim_student_srgb"]  = ""
            row["lpips_student_srgb"] = ""

        rows.append(row)

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Compute means safely
    def mean_or_nan(values):
        if not values:
            return float("nan")
        return float(np.mean(values))

    summary_lines = []
    summary_lines.append(f"Split: {args.split}")
    summary_lines.append(f"Num images (with GT): {len(rows)}\n")

    summary_lines.append("=== Input (raw UDC) vs GT ===")
    summary_lines.append(f"Mean PSNR Bayer: {mean_or_nan(acc['in_psnr_bayer']):.4f} dB")
    summary_lines.append(f"Mean SSIM Bayer: {mean_or_nan(acc['in_ssim_bayer']):.4f}")
    summary_lines.append(f"Mean PSNR sRGB:  {mean_or_nan(acc['in_psnr_srgb']):.4f} dB")
    summary_lines.append(f"Mean SSIM sRGB:  {mean_or_nan(acc['in_ssim_srgb']):.4f}")
    summary_lines.append(f"Mean LPIPS sRGB: {mean_or_nan(acc['in_lpips']):.4f}\n")

    if acc["t_psnr_bayer"]:
        summary_lines.append("=== Teacher vs GT ===")
        summary_lines.append(f"Mean PSNR Bayer: {mean_or_nan(acc['t_psnr_bayer']):.4f} dB")
        summary_lines.append(f"Mean SSIM Bayer: {mean_or_nan(acc['t_ssim_bayer']):.4f}")
        summary_lines.append(f"Mean PSNR sRGB:  {mean_or_nan(acc['t_psnr_srgb']):.4f} dB")
        summary_lines.append(f"Mean SSIM sRGB:  {mean_or_nan(acc['t_ssim_srgb']):.4f}")
        summary_lines.append(f"Mean LPIPS sRGB: {mean_or_nan(acc['t_lpips']):.4f}\n")

    if acc["s_psnr_bayer"]:
        summary_lines.append("=== Student vs GT ===")
        summary_lines.append(f"Mean PSNR Bayer: {mean_or_nan(acc['s_psnr_bayer']):.4f} dB")
        summary_lines.append(f"Mean SSIM Bayer: {mean_or_nan(acc['s_ssim_bayer']):.4f}")
        summary_lines.append(f"Mean PSNR sRGB:  {mean_or_nan(acc['s_psnr_srgb']):.4f} dB")
        summary_lines.append(f"Mean SSIM sRGB:  {mean_or_nan(acc['s_ssim_srgb']):.4f}")
        summary_lines.append(f"Mean LPIPS sRGB: {mean_or_nan(acc['s_lpips']):.4f}\n")

    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print("\n=== [eval_srgb_udc] Summary ===")
    print("\n".join(summary_lines))
    print(f"\nSaved CSV to:     {csv_path}")
    print(f"Saved summary to: {summary_path}")

    # Optional: copy to Drive
    if args.drive_results_root is not None:
        import shutil

        drive_csv     = os.path.join(args.drive_results_root, os.path.basename(csv_path))
        drive_summary = os.path.join(args.drive_results_root, os.path.basename(summary_path))

        shutil.copy(csv_path, drive_csv)
        shutil.copy(summary_path, drive_summary)

        print(f"Copied CSV to Drive:     {drive_csv}")
        print(f"Copied summary to Drive: {drive_summary}")


if __name__ == "__main__":
    main()
