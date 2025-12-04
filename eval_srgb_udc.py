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


def naive_bayer4_to_rgb(arr_4ch: np.ndarray) -> np.ndarray:
    """
    Naive 4ch -> RGB mapping (GR, R, B, GB) matching the reference script.

    ch0, ch3: greens, ch1: red, ch2: blue.
    Returns float32 RGB in [0,1].
    """
    if arr_4ch.ndim != 3 or arr_4ch.shape[2] != 4:
        raise ValueError(f"Expected (H,W,4) array, got {arr_4ch.shape}")

    arr = arr_4ch.astype(np.float32)
    if arr.max() > 2.0:
        arr = arr / 1023.0

    R = arr[:, :, 1]
    G = 0.5 * (arr[:, :, 0] + arr[:, :, 3])
    B = arr[:, :, 2]

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def apply_wb_gamma(rgb: np.ndarray, wb=(1.1, 1.0, 1.3), gamma: float = 2.2) -> np.ndarray:
    """
    Apply per-channel white balance and gamma correction to [0,1] RGB.
    """
    rgb_wb = rgb * np.array(wb, dtype=np.float32)[None, None, :]
    rgb_wb = np.clip(rgb_wb, 0.0, 1.0)
    rgb_gamma = np.power(rgb_wb, 1.0 / gamma)
    return np.clip(rgb_gamma, 0.0, 1.0)


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


def center_crop(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Center-crop an array (H,W,C) to (target_h, target_w, C).
    """
    H, W = arr.shape[:2]
    if H == target_h and W == target_w:
        return arr
    top = max((H - target_h) // 2, 0)
    left = max((W - target_w) // 2, 0)
    return arr[top : top + target_h, left : left + target_w, ...]


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
<<<<<<< HEAD
        default="/content/drive/MyDrive/Computational Imaging Project/Results/Model1",
=======
        default="/content/drive/MyDrive/Computational Imaging Project/Results/Model2",
>>>>>>> model2
        help="Local directory to store sRGB metric CSV + summary.",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="run1",
        help="Subfolder name created under both results roots for this run.",
    )
    parser.add_argument(
        "--drive-results-root",
        type=str,
<<<<<<< HEAD
        default="/content/drive/MyDrive/Computational Imaging Project/Results/Model1",
=======
        default="/content/drive/MyDrive/Computational Imaging Project/Results/Model2",
>>>>>>> model2
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

    results_root = os.path.join(args.results_root, args.results_name) if args.results_name else args.results_root
    drive_results_root = (
        os.path.join(args.drive_results_root, args.results_name)
        if args.drive_results_root is not None and len(args.drive_results_root) > 0 and args.results_name
        else args.drive_results_root
    )

    # Avoid Drive copy when paths are identical
    if drive_results_root and os.path.abspath(drive_results_root) == os.path.abspath(results_root):
        print("[eval_srgb_udc] Drive results root equals local results root; skipping Drive copy.")
        drive_results_root = None

    input_dir = os.path.join(args.data_root, args.split, "input")
    gt_dir = os.path.join(args.data_root, args.split, "GT")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"[eval_srgb_udc] Missing input dir: {input_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"[eval_srgb_udc] Missing GT dir: {gt_dir}")

    os.makedirs(results_root, exist_ok=True)
    if drive_results_root is not None and len(str(drive_results_root)) > 0:
        os.makedirs(drive_results_root, exist_ok=True)

    print("=== [eval_srgb_udc] Config ===")
    print(f"Data root:      {args.data_root}")
    print(f"Split:          {args.split}")
    print(f"Teacher preds:  {args.teacher_pred_dir}")
    print(f"Student preds:  {args.student_pred_dir}")
    print(f"Max images:     {args.max_images}")
    print(f"Results root:   {results_root}")
    print(f"Drive results:  {drive_results_root}")
    print(f"WB mode:        {args.wb_mode}")
    print(f"Gamma:          {args.gamma}")
    print("=================================")

    gt_files = sorted(glob(os.path.join(gt_dir, "*.npy")))
    if args.max_images is not None:
        gt_files = gt_files[: args.max_images]

    teacher_dir = args.teacher_pred_dir
    student_dir = args.student_pred_dir

    # Collect metrics per visualization method
    method_metrics = {
        "pseudo_rgb": [],          # existing mapping fourch_to_rgb + WB/Gamma
        "naive_rgb": [],           # naive Bayer->RGB
        "naive_rgb_wbgamma": [],   # naive Bayer->RGB + WB+Gamma
    }

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
        teacher_arr = np.load(teacher_path) if teacher_path is not None else None
        student_arr = np.load(student_path) if student_path is not None else None

        # Align shapes (center crop to smallest H/W among available arrays)
        Hs = [gt_arr.shape[0], inp_arr.shape[0]]
        Ws = [gt_arr.shape[1], inp_arr.shape[1]]
        if teacher_arr is not None:
            Hs.append(teacher_arr.shape[0]); Ws.append(teacher_arr.shape[1])
        if student_arr is not None:
            Hs.append(student_arr.shape[0]); Ws.append(student_arr.shape[1])
        target_h, target_w = min(Hs), min(Ws)

        gt_arr = center_crop(gt_arr, target_h, target_w)
        inp_arr = center_crop(inp_arr, target_h, target_w)
        if teacher_arr is not None:
            teacher_arr = center_crop(teacher_arr, target_h, target_w)
        if student_arr is not None:
            student_arr = center_crop(student_arr, target_h, target_w)

        # Pseudo-RGB (existing)
        inp_rgb_p = apply_gamma(apply_white_balance(fourch_to_rgb(inp_arr), args.wb_mode), args.gamma)
        gt_rgb_p = apply_gamma(apply_white_balance(fourch_to_rgb(gt_arr), args.wb_mode), args.gamma)
        teacher_rgb_p = (
            apply_gamma(apply_white_balance(fourch_to_rgb(teacher_arr), args.wb_mode), args.gamma)
            if teacher_arr is not None
            else None
        )
        student_rgb_p = (
            apply_gamma(apply_white_balance(fourch_to_rgb(student_arr), args.wb_mode), args.gamma)
            if student_arr is not None
            else None
        )

        # Naive RGB
        inp_rgb_n = naive_bayer4_to_rgb(inp_arr)
        gt_rgb_n = naive_bayer4_to_rgb(gt_arr)
        teacher_rgb_n = naive_bayer4_to_rgb(teacher_arr) if teacher_arr is not None else None
        student_rgb_n = naive_bayer4_to_rgb(student_arr) if student_arr is not None else None

        # Naive RGB + WB + Gamma
        inp_rgb_ng = apply_wb_gamma(inp_rgb_n)
        gt_rgb_ng = apply_wb_gamma(gt_rgb_n)
        teacher_rgb_ng = apply_wb_gamma(teacher_rgb_n) if teacher_rgb_n is not None else None
        student_rgb_ng = apply_wb_gamma(student_rgb_n) if student_rgb_n is not None else None

        def append_metrics(method_key, gt_rgb, inp_rgb, teacher_rgb, student_rgb):
            psnr_in, ssim_in = compute_psnr_ssim(gt_rgb, inp_rgb)
            psnr_teacher, ssim_teacher = (np.nan, np.nan)
            psnr_student, ssim_student = (np.nan, np.nan)

            if teacher_rgb is not None:
                psnr_teacher, ssim_teacher = compute_psnr_ssim(gt_rgb, teacher_rgb)
            if student_rgb is not None:
                psnr_student, ssim_student = compute_psnr_ssim(gt_rgb, student_rgb)

            method_metrics[method_key].append(
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

        append_metrics("pseudo_rgb", gt_rgb_p, inp_rgb_p, teacher_rgb_p, student_rgb_p)
        append_metrics("naive_rgb", gt_rgb_n, inp_rgb_n, teacher_rgb_n, student_rgb_n)
        append_metrics("naive_rgb_wbgamma", gt_rgb_ng, inp_rgb_ng, teacher_rgb_ng, student_rgb_ng)

    # Save per-method CSVs and summary
    def safe_mean(vals):
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    summary_lines = []

    for method_key, rows in method_metrics.items():
        csv_path = os.path.join(results_root, f"{args.split}_metrics_srgb_{method_key}.csv")
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
            for row in rows:
                writer.writerow(row)

        mean_psnr_in = safe_mean([r["psnr_input"] for r in rows])
        mean_ssim_in = safe_mean([r["ssim_input"] for r in rows])
        mean_psnr_teacher = safe_mean([r["psnr_teacher"] for r in rows])
        mean_ssim_teacher = safe_mean([r["ssim_teacher"] for r in rows])
        mean_psnr_student = safe_mean([r["psnr_student"] for r in rows])
        mean_ssim_student = safe_mean([r["ssim_student"] for r in rows])

        summary_lines.append(
            (
                method_key,
                len(rows),
                mean_psnr_in,
                mean_ssim_in,
                mean_psnr_teacher,
                mean_ssim_teacher,
                mean_psnr_student,
                mean_ssim_student,
                csv_path,
            )
        )

        print(f"[eval_srgb_udc] Saved CSV ({method_key}) to {csv_path}")

        if drive_results_root is not None and len(str(drive_results_root)) > 0:
            import shutil
            os.makedirs(drive_results_root, exist_ok=True)
            drive_csv = os.path.join(drive_results_root, os.path.basename(csv_path))
            shutil.copy(csv_path, drive_csv)
            print(f"[eval_srgb_udc] Copied ({method_key}) to Drive: {drive_csv}")

    summary_path = os.path.join(results_root, f"{args.split}_metrics_srgb_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Split: {args.split}\n\n")
        for (
            method_key,
            n_rows,
            mean_psnr_in,
            mean_ssim_in,
            mean_psnr_teacher,
            mean_ssim_teacher,
            mean_psnr_student,
            mean_ssim_student,
            csv_path,
        ) in summary_lines:
            f.write(f"[{method_key}] Num images: {n_rows}\n")
            f.write(f"  Input   -> PSNR: {mean_psnr_in:.4f} dB, SSIM: {mean_ssim_in:.4f}\n")
            f.write(f"  Teacher -> PSNR: {mean_psnr_teacher:.4f} dB, SSIM: {mean_ssim_teacher:.4f}\n")
            f.write(f"  Student -> PSNR: {mean_psnr_student:.4f} dB, SSIM: {mean_ssim_student:.4f}\n")
            f.write(f"  CSV: {csv_path}\n\n")

    print(f"[eval_srgb_udc] Saved summary to {summary_path}")
    print("=== Summary (mean metrics per method) ===")
    for (
        method_key,
        n_rows,
        mean_psnr_in,
        mean_ssim_in,
        mean_psnr_teacher,
        mean_ssim_teacher,
        mean_psnr_student,
        mean_ssim_student,
        _,
    ) in summary_lines:
        print(f"{method_key}: N={n_rows}")
        print(f"  Input   -> PSNR: {mean_psnr_in:.4f} dB, SSIM: {mean_ssim_in:.4f}")
        print(f"  Teacher -> PSNR: {mean_psnr_teacher:.4f} dB, SSIM: {mean_ssim_teacher:.4f}")
        print(f"  Student -> PSNR: {mean_psnr_student:.4f} dB, SSIM: {mean_ssim_student:.4f}")

    if drive_results_root is not None and len(str(drive_results_root)) > 0:
        import shutil
        drive_summary = os.path.join(drive_results_root, os.path.basename(summary_path))
        shutil.copy(summary_path, drive_summary)
        print(f"[eval_srgb_udc] Copied summary to Drive: {drive_summary}")


if __name__ == "__main__":
    main()
