# File: viz_srgb_udc.py
import os
import argparse
from glob import glob

import numpy as np
import matplotlib.pyplot as plt


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


def save_panel(
    out_path: str,
    inp_rgb: np.ndarray,
    teacher_rgb: np.ndarray,
    student_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    title: str = "",
):
    """
    Save a 1x4 comparison panel (Input, Teacher, Student, GT) as PNG.

    Any of teacher_rgb / student_rgb can be None; a placeholder will be shown.
    """
    num_cols = 4
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    if title:
        fig.suptitle(title, fontsize=14)

    panels = [
        ("Input", inp_rgb),
        ("Teacher", teacher_rgb),
        ("Student", student_rgb),
        ("GT", gt_rgb),
    ]

    for ax, (name, img) in zip(axes, panels):
        ax.axis("off")
        ax.set_title(name)
        if img is not None:
            ax.imshow(np.clip(img, 0.0, 1.0))
        else:
            ax.imshow(np.zeros((32, 32, 3), dtype=np.float32))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize UDC-SIT teacher/student predictions vs input & GT in sRGB."
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
        help="Dataset split to visualize.",
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
        default=50,
        help="Maximum number of images to visualize.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results_srgb_viz",
        help="Local directory to store visualization PNGs.",
    )
    parser.add_argument(
        "--drive-results-root",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/results_srgb_viz",
        help="Google Drive directory to mirror visualization PNGs. Leave empty to disable.",
    )
    parser.add_argument(
        "--wb-mode",
        type=str,
        default="none",
        choices=["none", "gray"],
        help="White balance mode applied to all sRGB images before visualization.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction applied to all sRGB images before visualization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = os.path.join(args.data_root, args.split, "input")
    gt_dir = os.path.join(args.data_root, args.split, "GT")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"[viz_srgb_udc] Missing input dir: {input_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"[viz_srgb_udc] Missing GT dir: {gt_dir}")

    teacher_dir = args.teacher_pred_dir
    student_dir = args.student_pred_dir

    split_viz_dir = os.path.join(args.results_root, args.split)
    os.makedirs(split_viz_dir, exist_ok=True)

    if args.drive_results_root is not None and len(args.drive_results_root) > 0:
        os.makedirs(args.drive_results_root, exist_ok=True)

    print("=== [viz_srgb_udc] Config ===")
    print(f"Data root:      {args.data_root}")
    print(f"Split:          {args.split}")
    print(f"Teacher preds:  {teacher_dir}")
    print(f"Student preds:  {student_dir}")
    print(f"Max images:     {args.max_images}")
    print(f"Viz root:       {args.results_root}")
    print(f"Drive viz root: {args.drive_results_root}")
    print(f"WB mode:        {args.wb_mode}")
    print(f"Gamma:          {args.gamma}")
    print("=================================")

    gt_files = sorted(glob(os.path.join(gt_dir, "*.npy")))
    if args.max_images is not None:
        gt_files = gt_files[: args.max_images]

    for gt_path in gt_files:
        fname = os.path.basename(gt_path)
        base = os.path.splitext(fname)[0]
        inp_path = os.path.join(input_dir, fname)

        if not os.path.exists(inp_path):
            print(f"[viz_srgb_udc] WARNING: Missing input for {fname}, skipping.")
            continue

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

        # Load arrays
        gt_arr = np.load(gt_path)
        inp_arr = np.load(inp_path)

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

        # WB + gamma
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

        panel_title = f"{args.split} - {base}"
        panel_path = os.path.join(split_viz_dir, f"{base}_panel.png")
        save_panel(panel_path, inp_rgb, teacher_rgb, student_rgb, gt_rgb, title=panel_title)
        print(f"[viz_srgb_udc] Saved panel: {panel_path}")

    # Copy viz to Drive
    if args.drive_results_root is not None and len(args.drive_results_root) > 0:
        import shutil

        drive_split_dir = os.path.join(args.drive_results_root, args.split)
        shutil.copytree(split_viz_dir, drive_split_dir, dirs_exist_ok=True)
        print(f"[viz_srgb_udc] Copied panels to Drive: {drive_split_dir}")


if __name__ == "__main__":
    main()
