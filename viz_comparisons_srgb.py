# File: viz_comparisons_srgb.py
"""
Create 2x2 comparison panels in sRGB:

Row 1: Input | Teacher
Row 2: Student | GT

Any of input/teacher/student can be omitted; GT is required.

Usage example:

python viz_comparisons_srgb.py \
    --gt-dir /content/dataset/UDC-SIT_srgb/validation/gt \
    --input-dir /content/dataset/UDC-SIT_srgb/validation/input \
    --teacher-dir /content/results_full_srgb_images/teacher_validation \
    --student-dir /content/results_full_srgb_images/student_validation \
    --out-dir viz_panels/validation \
    --max-images 50
"""

import os
import argparse
import numpy as np
from glob import glob

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_srgb_image(path):
    """
    Read an sRGB image (PNG/JPG) and return float32 array in [0,1], shape (H,W,3).
    Handles uint8, uint16, grayscale, and RGBA.
    """
    img = imageio.imread(path)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] == 4:
        img = img[..., :3]

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)

    img = np.clip(img, 0.0, 1.0)
    return img


def add_axis_image(ax, img, title, missing=False):
    """
    Helper to show an image or a blank placeholder on a given axis.
    """
    ax.axis("off")
    if img is None or missing:
        ax.set_title(f"{title} (missing)", fontsize=10)
        ax.set_facecolor("black")
        return
    ax.imshow(img)
    ax.set_title(title, fontsize=10)


def parse_args():
    parser = argparse.ArgumentParser(description="Create sRGB comparison panels.")
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="Directory with GT sRGB images.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory with input sRGB images (optional).",
    )
    parser.add_argument(
        "--teacher-dir",
        type=str,
        default=None,
        help="Directory with teacher sRGB images (optional).",
    )
    parser.add_argument(
        "--student-dir",
        type=str,
        default=None,
        help="Directory with student sRGB images (optional).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save comparison panels.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max number of panels to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    gt_dir = args.gt_dir
    inp_dir = args.input_dir
    teacher_dir = args.teacher_dir
    student_dir = args.student_dir
    out_dir = args.out_dir
    max_images = args.max_images

    os.makedirs(out_dir, exist_ok=True)

    # Base filenames from GT (we assume GT exists for everything we care about)
    gt_files = sorted(
        [os.path.basename(p) for p in glob(os.path.join(gt_dir, "*"))]
    )
    if max_images is not None:
        gt_files = gt_files[:max_images]

    print(f"[viz_comparisons_srgb] GT dir: {gt_dir}")
    if inp_dir:
        print(f"[viz_comparisons_srgb] Input dir: {inp_dir}")
    if teacher_dir:
        print(f"[viz_comparisons_srgb] Teacher dir: {teacher_dir}")
    if student_dir:
        print(f"[viz_comparisons_srgb] Student dir: {student_dir}")
    print(f"[viz_comparisons_srgb] Output panels: {out_dir}")
    print(f"[viz_comparisons_srgb] Num GT images: {len(gt_files)}")

    for fname in tqdm(gt_files, desc="Generating panels"):
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            # Shouldn't happen since we built list from GT, but just in case
            continue

        gt_img = read_srgb_image(gt_path)

        # Optional images
        inp_img = None
        teacher_img = None
        student_img = None

        if inp_dir is not None:
            inp_path = os.path.join(inp_dir, fname)
            if os.path.exists(inp_path):
                inp_img = read_srgb_image(inp_path)

        if teacher_dir is not None:
            t_path = os.path.join(teacher_dir, fname)
            if os.path.exists(t_path):
                teacher_img = read_srgb_image(t_path)

        if student_dir is not None:
            s_path = os.path.join(student_dir, fname)
            if os.path.exists(s_path):
                student_img = read_srgb_image(s_path)

        # Build figure
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))

        add_axis_image(axes[0, 0], inp_img, "Input", missing=(inp_img is None))
        add_axis_image(axes[0, 1], teacher_img, "Teacher", missing=(teacher_img is None))
        add_axis_image(axes[1, 0], student_img, "Student", missing=(student_img is None))
        add_axis_image(axes[1, 1], gt_img, "Ground Truth", missing=False)

        fig.suptitle(fname, fontsize=10)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_name = os.path.splitext(fname)[0] + "_panel.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print("[viz_comparisons_srgb] Done.")


if __name__ == "__main__":
    main()
