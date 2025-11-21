# File: viz_srgb_udc.py
"""
Unified visualization panels for UDC-SIT in sRGB.

For each image (up to --max-images):
- Load input 4-ch npy: <data_root>/<split>/input/<name>.npy
- Load GT 4-ch npy:    <data_root>/<split>/GT/<name>.npy
- Load teacher pred:   <teacher_pred_dir>/<name>.npy (optional)
- Load student pred:   <student_pred_dir>/<name>.npy (optional)

Convert 4-ch -> sRGB using the same mapping as eval_srgb_udc.py and generate
a 2x2 panel:

    [ Input ]   [ Teacher ]
    [ Student ] [ Ground Truth ]

If teacher/student predictions are missing, those slots show "N/A".

Outputs:
- Per-image panel PNGs under `viz_root/<split>/<name>_panel.png`.
- Optionally mirrored under `drive_viz_root/<split>/`.
"""

import os
import sys
import argparse
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified sRGB visualization for UDC-SIT teacher/student predictions."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/dataset/UDC-SIT",
        help="Root of UDC-SIT dataset; expects <data-root>/<split>/(input|GT)/*.npy",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Split under data-root (e.g., training, validation, testing, train, val, test).",
    )

    parser.add_argument(
        "--teacher-pred-dir",
        type=str,
        default="results_full/teacher_full_validation/npy",
        help="Directory containing teacher 4-ch predictions (.npy). Leave empty if none.",
    )
    parser.add_argument(
        "--student-pred-dir",
        type=str,
        default="results_full/student_full_validation/npy",
        help="Directory containing student 4-ch predictions (.npy). Leave empty if none.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Max number of images to visualize (sorted by filename).",
    )

    parser.add_argument(
        "--viz-root",
        type=str,
        default="viz_srgb_panels",
        help="Local root folder for visualization outputs.",
    )
    parser.add_argument(
        "--drive-viz-root",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/viz_srgb_panels",
        help="Drive root folder to mirror visualization outputs.",
    )

    return parser.parse_args()


def load_4ch_npy(path):
    """
    Load (H, W, 4) [0..1023] -> (H, W, 4) [0..1].
    """
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected (H, W, 4) at {path}, got {arr.shape}")
    return arr.astype(np.float32) / 1023.0


def fourch_to_srgb(arr_4ch):
    """
    Same mapping as in eval_srgb_udc.py.
    ch0: GR, ch1: R, ch2: B, ch3: GB
    """
    GR = arr_4ch[:, :, 0]
    R  = arr_4ch[:, :, 1]
    B  = arr_4ch[:, :, 2]
    GB = arr_4ch[:, :, 3]

    G = 0.5 * (GR + GB)
    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def make_panel(input_rgb, teacher_rgb, student_rgb, gt_rgb, title):
    """
    Build a 2x2 matplotlib figure and return the figure.
    teacher_rgb or student_rgb can be None (slot becomes blank).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=14)

    def show(ax, img, label):
        ax.set_title(label)
        ax.axis("off")
        if img is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
        else:
            ax.imshow(np.clip(img, 0.0, 1.0))

    show(axes[0, 0], input_rgb,   "Input (UDC)")
    show(axes[0, 1], teacher_rgb, "Teacher pred" if teacher_rgb is not None else "Teacher pred (N/A)")
    show(axes[1, 0], student_rgb, "Student pred" if student_rgb is not None else "Student pred (N/A)")
    show(axes[1, 1], gt_rgb,      "Ground Truth")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def main():
    args = parse_args()

    print("\n--- [viz_srgb_udc] Configuration ---")
    print(f"Data root:        {args.data_root}")
    print(f"Split:            {args.split}")
    print(f"Teacher pred dir: {args.teacher_pred_dir}")
    print(f"Student pred dir: {args.student_pred_dir}")
    print(f"Max images:       {args.max_images}")
    print(f"Viz root:         {args.viz_root}")
    print(f"Drive viz root:   {args.drive_viz_root}")
    print("-----------------------------------\n")

    input_dir = os.path.join(args.data_root, args.split, "input")
    gt_dir    = os.path.join(args.data_root, args.split, "GT")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")

    input_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    if args.max_images is not None:
        input_files = input_files[:args.max_images]

    os.makedirs(args.viz_root, exist_ok=True)
    os.makedirs(args.drive_viz_root, exist_ok=True)

    split_viz_dir = os.path.join(args.viz_root, args.split)
    os.makedirs(split_viz_dir, exist_ok=True)

    teacher_dir = args.teacher_pred_dir if args.teacher_pred_dir and os.path.isdir(args.teacher_pred_dir) else None
    student_dir = args.student_pred_dir if args.student_pred_dir and os.path.isdir(args.student_pred_dir) else None

    print(f"--- [viz_srgb_udc] Num images to visualize: {len(input_files)}")

    pbar = tqdm(input_files, desc=f"[viz {args.split}]")
    for inp_path in pbar:
        fname = os.path.basename(inp_path)
        name_noext = os.path.splitext(fname)[0]

        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            print(f"WARNING: GT not found for {fname}, skipping.")
            continue

        # Base input & GT
        arr_inp = load_4ch_npy(inp_path)
        arr_gt  = load_4ch_npy(gt_path)
        if arr_inp.shape != arr_gt.shape:
            print(f"WARNING: shape mismatch input/GT for {fname}, skipping.")
            continue

        inp_rgb = fourch_to_srgb(arr_inp)
        gt_rgb  = fourch_to_srgb(arr_gt)

        # Teacher & Student predictions (optional)
        teacher_rgb = None
        student_rgb = None

        if teacher_dir is not None:
            t_path = os.path.join(teacher_dir, fname)
            if os.path.exists(t_path):
                arr_t = load_4ch_npy(t_path)
                if arr_t.shape == arr_gt.shape:
                    teacher_rgb = fourch_to_srgb(arr_t)
                else:
                    print(f"WARNING: teacher pred shape mismatch for {fname}, skipping teacher.")
            else:
                # silent if missing
                pass

        if student_dir is not None:
            s_path = os.path.join(student_dir, fname)
            if os.path.exists(s_path):
                arr_s = load_4ch_npy(s_path)
                if arr_s.shape == arr_gt.shape:
                    student_rgb = fourch_to_srgb(arr_s)
                else:
                    print(f"WARNING: student pred shape mismatch for {fname}, skipping student.")
            else:
                # silent if missing
                pass

        panel_title = f"{args.split} - {name_noext}"
        fig = make_panel(inp_rgb, teacher_rgb, student_rgb, gt_rgb, title=panel_title)

        panel_path = os.path.join(split_viz_dir, f"{name_noext}_panel.png")
        fig.savefig(panel_path, dpi=150)
        plt.close(fig)

    # Copy whole split folder to Drive
    try:
        import shutil
        drive_split_dir = os.path.join(args.drive_viz_root, args.split)
        os.makedirs(drive_split_dir, exist_ok=True)
        # Copy panel PNGs
        for panel_file in glob(os.path.join(split_viz_dir, "*.png")):
            base = os.path.basename(panel_file)
            shutil.copy(panel_file, os.path.join(drive_split_dir, base))
        print(f"--- [viz_srgb_udc] Copied panels to Drive: {drive_split_dir}")
    except Exception as e:
        print(f"Could not copy visualization panels to Drive: {e}")


if __name__ == "__main__":
    main()
