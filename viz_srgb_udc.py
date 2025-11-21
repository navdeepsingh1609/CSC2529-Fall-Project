# viz_srgb_udc.py
"""
Visualize UDC-SIT predictions in sRGB space.

- Loads 4-ch .npy (H, W, 4) for input, GT, and predictions.
- Normalizes to [0,1] (handles [0,1023] or [0,1]).
- Uses same 4ch -> sRGB mapping as eval_srgb_udc.py.
- For each image, creates a 1x4 panel:
    Input | Teacher | Student | GT
- Saves panels locally and mirrors them to Google Drive.
"""

import os
import argparse
from glob import glob

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_and_normalize_4ch(path):
    """
    Load a 4-channel .npy file and normalize to [0,1].

    Handles both raw [0,1023] and already-normalized [0,1] inputs.
    Returns array of shape (H, W, 4), float32 in [0,1].
    """
    arr = np.load(path).astype(np.float32)

    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected (H, W, 4) array, got {arr.shape} for {path}")

    maxv = float(arr.max())
    if maxv == 0.0:
        return arr  # all zeros
    if maxv > 2.0:
        arr = arr / 1023.0
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return arr


def fourch_to_srgb(arr_4ch, wb_mode="channel_gains", gamma=2.2):
    """
    Convert 4-channel Bayer-like array (H, W, 4) in [0,1] to approximate sRGB (H, W, 3) in [0,1].

    This is a lightweight visualization-only ISP approximation.
    """
    if arr_4ch.ndim != 3 or arr_4ch.shape[2] != 4:
        raise ValueError(f"fourch_to_srgb expects (H, W, 4), got {arr_4ch.shape}")

    c0 = arr_4ch[:, :, 0]
    c1 = arr_4ch[:, :, 1]
    c2 = arr_4ch[:, :, 2]
    c3 = arr_4ch[:, :, 3]

    R_lin = c1 + 0.1 * c0
    G_lin = c2 + 0.1 * c0
    B_lin = c3 + 0.1 * c0

    rgb_lin = np.stack([R_lin, G_lin, B_lin], axis=-1)

    if wb_mode == "channel_gains":
        gains = np.array([1.2, 1.0, 1.5], dtype=np.float32).reshape(1, 1, 3)
        rgb_lin = rgb_lin * gains
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)

    if gamma is not None and gamma != 1.0:
        rgb_srgb = np.power(rgb_lin, 1.0 / gamma)
    else:
        rgb_srgb = rgb_lin

    rgb_srgb = np.clip(rgb_srgb, 0.0, 1.0)
    return rgb_srgb.astype(np.float32)


def save_panel(
    out_path,
    inp_srgb,
    teacher_srgb,
    student_srgb,
    gt_srgb,
    title="",
):
    """
    Save a 1x4 panel figure: Input | Teacher | Student | GT
    directly to out_path using Matplotlib (no numpy conversion of canvas).
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.ravel()

    images = [inp_srgb, teacher_srgb, student_srgb, gt_srgb]
    labels = ["Input", "Teacher", "Student", "GT"]

    for ax, img, label in zip(axes, images, labels):
        ax.axis("off")
        if img is not None:
            ax.imshow(np.clip(img, 0.0, 1.0))
            ax.set_title(label, fontsize=10)
        else:
            ax.set_title(f"{label}\n(N/A)", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85 if title else 0.9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_split(
    data_root,
    split,
    teacher_pred_dir,
    student_pred_dir,
    max_images,
    viz_root,
    drive_viz_root,
    wb_mode,
    gamma,
):
    gt_dir  = os.path.join(data_root, split, "GT")
    inp_dir = os.path.join(data_root, split, "input")

    if not os.path.isdir(gt_dir):
        print(f"[viz_srgb_udc] ERROR: GT dir not found: {gt_dir}")
        return
    if not os.path.isdir(inp_dir):
        print(f"[viz_srgb_udc] ERROR: input dir not found: {inp_dir}")
        return

    gt_files = sorted(glob(os.path.join(gt_dir, "*.npy")))
    if max_images is not None:
        gt_files = gt_files[:max_images]

    if len(gt_files) == 0:
        print(f"[viz_srgb_udc] No GT .npy files in {gt_dir}")
        return

    split_viz_dir = os.path.join(viz_root, split)
    os.makedirs(split_viz_dir, exist_ok=True)

    if drive_viz_root is not None and drive_viz_root != "":
        drive_split_dir = os.path.join(drive_viz_root, split)
        os.makedirs(drive_split_dir, exist_ok=True)
    else:
        drive_split_dir = None

    print(f"\n--- [viz_srgb_udc] Visualizing split '{split}' ---")
    print(f"GT dir:       {gt_dir}")
    print(f"Input dir:    {inp_dir}")
    print(f"Teacher preds:{teacher_pred_dir}")
    print(f"Student preds:{student_pred_dir}")
    print(f"Output dir:   {split_viz_dir}")

    for gt_path in tqdm(gt_files, desc=f"[viz {split}] images", ncols=80):
        base = os.path.basename(gt_path)
        name = os.path.splitext(base)[0]

        inp_path = os.path.join(inp_dir, base)
        if not os.path.exists(inp_path):
            print(f"  [WARN] Missing input for {base}, skipping.")
            continue

        teacher_path = (
            os.path.join(teacher_pred_dir, base)
            if teacher_pred_dir is not None and teacher_pred_dir != ""
            else None
        )
        student_path = (
            os.path.join(student_pred_dir, base)
            if student_pred_dir is not None and student_pred_dir != ""
            else None
        )

        # Load and convert to sRGB
        inp_4ch = load_and_normalize_4ch(inp_path)
        gt_4ch  = load_and_normalize_4ch(gt_path)

        inp_srgb = fourch_to_srgb(inp_4ch, wb_mode=wb_mode, gamma=gamma)
        gt_srgb  = fourch_to_srgb(gt_4ch,  wb_mode=wb_mode, gamma=gamma)

        teacher_srgb = None
        student_srgb = None

        if teacher_path is not None and os.path.exists(teacher_path):
            teacher_4ch   = load_and_normalize_4ch(teacher_path)
            teacher_srgb  = fourch_to_srgb(teacher_4ch, wb_mode=wb_mode, gamma=gamma)

        if student_path is not None and os.path.exists(student_path):
            student_4ch   = load_and_normalize_4ch(student_path)
            student_srgb  = fourch_to_srgb(student_4ch, wb_mode=wb_mode, gamma=gamma)

        panel_title = f"{split} - {name}"
        panel_path  = os.path.join(split_viz_dir, f"{name}_panel.png")

        save_panel(
            out_path=panel_path,
            inp_srgb=inp_srgb,
            teacher_srgb=teacher_srgb,
            student_srgb=student_srgb,
            gt_srgb=gt_srgb,
            title=panel_title,
        )

        if drive_split_dir is not None:
            import shutil

            shutil.copy(panel_path, os.path.join(drive_split_dir, os.path.basename(panel_path)))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize UDC-SIT predictions in sRGB (Input | Teacher | Student | GT panels)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/dataset/UDC-SIT/UDC-SIT",
        help="Root of UDC-SIT dataset (contains e.g. validation/GT, validation/input).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "testing"],
        help="Which split to visualize.",
    )
    parser.add_argument(
        "--teacher-pred-dir",
        type=str,
        default="results_full/teacher_full_validation/npy",
        help="Directory with teacher 4-ch .npy predictions.",
    )
    parser.add_argument(
        "--student-pred-dir",
        type=str,
        default="results_full/student_full_validation/npy",
        help="Directory with student 4-ch .npy predictions.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Max number of images to visualize.",
    )
    parser.add_argument(
        "--viz-root",
        type=str,
        default="results_srgb_viz",
        help="Where to store sRGB comparison panels.",
    )
    parser.add_argument(
        "--drive-viz-root",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/results_srgb_viz",
        help="Google Drive folder to mirror viz panels.",
    )
    parser.add_argument(
        "--wb-mode",
        type=str,
        default="channel_gains",
        choices=["none", "channel_gains"],
        help="White-balance mode applied in fourch_to_srgb.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="Gamma for fourch_to_srgb (1.0 = no gamma correction).",
    )

    args = parser.parse_args()

    print("=== [viz_srgb_udc] Config ===")
    print(f"Data root:      {args.data_root}")
    print(f"Split:          {args.split}")
    print(f"Teacher preds:  {args.teacher_pred_dir}")
    print(f"Student preds:  {args.student_pred_dir}")
    print(f"Max images:     {args.max_images}")
    print(f"Viz root:       {args.viz_root}")
    print(f"Drive viz root: {args.drive_viz_root}")
    print(f"WB mode:        {args.wb_mode}")
    print(f"Gamma:          {args.gamma}")
    print("=================================")

    os.makedirs(args.viz_root, exist_ok=True)
    if args.drive_viz_root is not None and args.drive_viz_root != "":
        os.makedirs(args.drive_viz_root, exist_ok=True)

    process_split(
        data_root=args.data_root,
        split=args.split,
        teacher_pred_dir=args.teacher_pred_dir,
        student_pred_dir=args.student_pred_dir,
        max_images=args.max_images,
        viz_root=args.viz_root,
        drive_viz_root=args.drive_viz_root,
        wb_mode=args.wb_mode,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
