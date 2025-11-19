# File: testing_full.py
import os, sys, glob, time, zipfile, csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import skimage.metrics
import lpips

# Try to import Colab download utilities (will fail gracefully outside Colab)
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    colab_files = None
    IN_COLAB = False

# ===================== CONFIG =====================

# Validation split (local, after you extracted UDC-SIT)
VAL_DIR = "/content/dataset/UDC-SIT/validation"

# Test split (you said it lives on Drive like this)
# This directory is expected to contain:
#   test/input/*.npy
#   test/GT/*.npy
TEST_DIR = "/content/drive/MyDrive/Computational Imaging Project/UDC-SIT/UDC-SIT/test"

# Teacher & Student weights (full training)
TEACHER_WEIGHTS_PATH = "teacher_4ch_22epochs_bs8.pth"          # adjust if your file name differs
STUDENT_WEIGHTS_PATH = "student_distilled_4ch_full_data.pth"   # adjust if needed

# How many examples to visualize per split
NUM_PLOT_EXAMPLES_VAL  = 5
NUM_PLOT_EXAMPLES_TEST = 5    # set to 0 if you don't want plots for test

PATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Where to save 4-channel patches (input / GT / teacher / student)
PATCH_SAVE_DIR = "patch_outputs_full"
os.makedirs(PATCH_SAVE_DIR, exist_ok=True)

# Where to save CSV metrics
METRICS_DIR = "metrics_full"
os.makedirs(METRICS_DIR, exist_ok=True)

# ===================================================

# --- Make sure MambaIR is on sys.path ---
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)
print(f"Added {mambair_path} to sys.path")

from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent


# ============ Helper functions ============

def load_center_patch_4ch(npy_path, patch_size=256):
    """
    Load a full UDC-SIT .npy file (H,W,4 in [0,1023]) and
    return a center patch of size (patch_size, patch_size, 4).
    """
    arr = np.load(npy_path).astype(np.float32)   # (H,W,4)
    H, W, C = arr.shape
    ps = patch_size
    r0 = (H - ps) // 2
    c0 = (W - ps) // 2
    r1, c1 = r0 + ps, c0 + ps
    patch = arr[r0:r1, c0:c1, :]   # (ps,ps,4)
    return patch  # still in [0,1023]


def bayer4_to_rgb_balanced(bayer_4ch, r_gain=1.9, b_gain=1.9):
    """
    Simple demosaic-ish viewer:
      - Treat channels as (GR, R, B, GB)
      - G = average of GR and GB
      - Apply a fixed gain to R and B to reduce green cast.

    Input:  (H,W,4) or (4,H,W) in [0,1023] or [0,1]
    Output: (H,W,3) in [0,1] for visualization / LPIPS.
    """
    arr = np.asarray(bayer_4ch, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")

    # Ensure we have (H,W,4)
    if arr.shape[-1] == 4:        # (H,W,4)
        GR = arr[..., 0]
        R  = arr[..., 1]
        B  = arr[..., 2]
        GB = arr[..., 3]
    elif arr.shape[0] == 4:       # (4,H,W)
        GR = arr[0]
        R  = arr[1]
        B  = arr[2]
        GB = arr[3]
    else:
        raise ValueError(f"Expected shape (...,4), got {arr.shape}")

    # Scale to [0,1] if we're in [0,1023]
    max_val = arr.max()
    if max_val > 2.0:
        scale = 1023.0
    else:
        scale = 1.0

    G  = (GR + GB) / (2.0 * scale)
    Rn = (R / scale) * r_gain
    Bn = (B / scale) * b_gain

    rgb = np.stack([Rn, G, Bn], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def numpy_to_lpips_tensor(img_np, device):
    """
    Convert an RGB image in [0,1] (H,W,3) to LPIPS tensor in [-1,1], shape (1,3,H,W).
    """
    t = torch.from_numpy(img_np).float().to(device)
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    t = t * 2.0 - 1.0                    # [0,1] -> [-1,1]
    return t


def summarize_results(results_per_image):
    """
    Compute mean metrics over all images in a split.
    """
    if not results_per_image:
        return {}

    keys = [
        "psnr_input", "ssim_input", "lpips_input",
        "psnr_teacher", "ssim_teacher", "lpips_teacher", "time_teacher_ms",
        "psnr_student", "ssim_student", "lpips_student", "time_student_ms",
    ]
    sums = {k: 0.0 for k in keys}

    for res in results_per_image:
        sums["psnr_input"]     += res["psnr_input"]
        sums["ssim_input"]     += res["ssim_input"]
        sums["lpips_input"]    += res["lpips_input"]

        sums["psnr_teacher"]   += res["psnr_teacher"]
        sums["ssim_teacher"]   += res["ssim_teacher"]
        sums["lpips_teacher"]  += res["lpips_teacher"]
        sums["time_teacher_ms"]+= res["time_teacher"] * 1000.0

        sums["psnr_student"]   += res["psnr_student"]
        sums["ssim_student"]   += res["ssim_student"]
        sums["lpips_student"]  += res["lpips_student"]
        sums["time_student_ms"]+= res["time_student"] * 1000.0

    n = len(results_per_image)
    means = {k: v / n for k, v in sums.items()}
    return means


def save_metrics_csv(results_per_image, csv_path):
    """
    Save per-image metrics to a CSV file.
    """
    if not results_per_image:
        return

    fieldnames = [
        "id",
        "psnr_input", "ssim_input", "lpips_input",
        "psnr_teacher", "ssim_teacher", "lpips_teacher", "time_teacher_ms",
        "psnr_student", "ssim_student", "lpips_student", "time_student_ms",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results_per_image:
            row = {
                "id": res["id"],
                "psnr_input": res["psnr_input"],
                "ssim_input": res["ssim_input"],
                "lpips_input": res["lpips_input"],
                "psnr_teacher": res["psnr_teacher"],
                "ssim_teacher": res["ssim_teacher"],
                "lpips_teacher": res["lpips_teacher"],
                "time_teacher_ms": res["time_teacher"] * 1000.0,
                "psnr_student": res["psnr_student"],
                "ssim_student": res["ssim_student"],
                "lpips_student": res["lpips_student"],
                "time_student_ms": res["time_student"] * 1000.0,
            }
            writer.writerow(row)


def evaluate_split(
    split_name,
    data_dir,
    teacher,
    student,
    lpips_model,
    patch_size,
    num_plot_examples,
    patch_save_dir,
    metrics_dir,
    device,
):
    """
    Evaluate teacher & student on a given split (val or test).
    - Computes per-image metrics
    - Optionally plots a grid of first num_plot_examples
    - Saves 4-ch patches to patch_save_dir
    - Saves per-image metrics as CSV to metrics_dir
    """
    input_dir = os.path.join(data_dir, "input")
    gt_dir    = os.path.join(data_dir, "GT")

    if not os.path.isdir(input_dir):
        print(f"[{split_name}] WARNING: Input dir not found: {input_dir} — skipping this split.")
        return

    if not os.path.isdir(gt_dir):
        print(f"[{split_name}] WARNING: GT dir not found: {gt_dir} — skipping this split (no metrics without GT).")
        return

    input_paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not input_paths:
        print(f"[{split_name}] WARNING: No .npy files found in {input_dir} — skipping.")
        return

    print(f"\n=== Evaluating split: {split_name} ===")
    print(f"Data dir: {data_dir}")
    print(f"Num images: {len(input_paths)}")

    results_per_image = []

    # For plotting:
    plot_inputs   = []
    plot_teachers = []
    plot_students = []
    plot_gts      = []
    plot_ids      = []

    for idx, input_path in enumerate(input_paths):
        img_id = os.path.splitext(os.path.basename(input_path))[0]
        gt_path = os.path.join(gt_dir, os.path.basename(input_path))

        if not os.path.exists(gt_path):
            print(f"  [WARNING] Missing GT for {img_id}, skipping.")
            continue

        print(f"[{split_name}] ({idx+1}/{len(input_paths)}) Image ID: {img_id}")

        # 1) Load center patches (H,W,4) in [0,1023]
        inp_patch_4ch = load_center_patch_4ch(input_path, patch_size)
        gt_patch_4ch  = load_center_patch_4ch(gt_path,   patch_size)

        # 2) Normalize to [0,1] and convert to (1,4,H,W)
        inp_norm = (inp_patch_4ch / 1023.0).astype(np.float32)
        gt_norm  = (gt_patch_4ch  / 1023.0).astype(np.float32)

        inp_chw = np.transpose(inp_norm, (2, 0, 1))  # (4,H,W)
        gt_chw  = np.transpose(gt_norm,  (2, 0, 1))  # (4,H,W)

        inp_bchw = torch.from_numpy(inp_chw).unsqueeze(0).to(device)  # (1,4,H,W)
        gt_bchw  = torch.from_numpy(gt_chw).unsqueeze(0).to(device)

        # 3) Forward through models
        with torch.no_grad():
            t_start = time.time()
            teacher_out_bchw, _, _ = teacher(inp_bchw)
            t_time = time.time() - t_start

            t_start = time.time()
            student_out_bchw, _ = student(inp_bchw)
            s_time = time.time() - t_start

        # 4) Clamp outputs to [0,1] for metrics and visualization
        teacher_out_bchw = torch.clamp(teacher_out_bchw, 0.0, 1.0)
        student_out_bchw = torch.clamp(student_out_bchw, 0.0, 1.0)

        # 5) Back to numpy (H,W,4) in [0,1] for metrics
        inp_4n     = inp_norm
        gt_4n      = gt_norm
        teacher_4n = teacher_out_bchw.squeeze(0).permute(1, 2, 0).cpu().numpy()
        student_4n = student_out_bchw.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 6) 4-channel PSNR/SSIM (Bayer space)
        psnr_input   = skimage.metrics.peak_signal_noise_ratio(gt_4n, inp_4n, data_range=1.0)
        ssim_input   = skimage.metrics.structural_similarity(gt_4n, inp_4n, data_range=1.0, channel_axis=-1)

        psnr_teacher = skimage.metrics.peak_signal_noise_ratio(gt_4n, teacher_4n, data_range=1.0)
        ssim_teacher = skimage.metrics.structural_similarity(gt_4n, teacher_4n, data_range=1.0, channel_axis=-1)

        psnr_student = skimage.metrics.peak_signal_noise_ratio(gt_4n, student_4n, data_range=1.0)
        ssim_student = skimage.metrics.structural_similarity(gt_4n, student_4n, data_range=1.0, channel_axis=-1)

        # 7) LPIPS on balanced RGB projection (same as your good script)
        inp_rgb     = bayer4_to_rgb_balanced(inp_patch_4ch)
        gt_rgb      = bayer4_to_rgb_balanced(gt_patch_4ch)
        teacher_rgb = bayer4_to_rgb_balanced(teacher_4n * 1023.0)
        student_rgb = bayer4_to_rgb_balanced(student_4n * 1023.0)

        gt_lp      = numpy_to_lpips_tensor(gt_rgb,      device)
        inp_lp     = numpy_to_lpips_tensor(inp_rgb,     device)
        teacher_lp = numpy_to_lpips_tensor(teacher_rgb, device)
        student_lp = numpy_to_lpips_tensor(student_rgb, device)

        with torch.no_grad():
            lpips_input   = lpips_model(gt_lp, inp_lp).item()
            lpips_teacher = lpips_model(gt_lp, teacher_lp).item()
            lpips_student = lpips_model(gt_lp, student_lp).item()

        # 8) Save metrics for this image
        res = {
            "id": img_id,
            "psnr_input": psnr_input,
            "ssim_input": ssim_input,
            "lpips_input": lpips_input,

            "psnr_teacher": psnr_teacher,
            "ssim_teacher": ssim_teacher,
            "lpips_teacher": lpips_teacher,
            "time_teacher": t_time,

            "psnr_student": psnr_student,
            "ssim_student": ssim_student,
            "lpips_student": lpips_student,
            "time_student": s_time,
        }
        results_per_image.append(res)

        print(f"  Input   -> PSNR: {psnr_input:.2f}, SSIM: {ssim_input:.4f}, LPIPS: {lpips_input:.4f}")
        print(f"  Teacher -> PSNR: {psnr_teacher:.2f}, SSIM: {ssim_teacher:.4f}, LPIPS: {lpips_teacher:.4f}, Time: {t_time*1000:.2f} ms")
        print(f"  Student -> PSNR: {psnr_student:.2f}, SSIM: {ssim_student:.4f}, LPIPS: {lpips_student:.4f}, Time: {s_time*1000:.2f} ms")

        # 9) Save 4-ch patches as .npy (in original [0,1023] scale)
        np.save(os.path.join(patch_save_dir, f"{split_name}_{img_id}_input_patch.npy"),   inp_patch_4ch)
        np.save(os.path.join(patch_save_dir, f"{split_name}_{img_id}_gt_patch.npy"),      gt_patch_4ch)
        np.save(os.path.join(patch_save_dir, f"{split_name}_{img_id}_teacher_patch.npy"), teacher_4n * 1023.0)
        np.save(os.path.join(patch_save_dir, f"{split_name}_{img_id}_student_patch.npy"), student_4n * 1023.0)

        # 10) Store images for plotting (only first N examples)
        if len(plot_inputs) < num_plot_examples:
            plot_ids.append(img_id)
            plot_inputs.append(inp_rgb)
            plot_teachers.append(teacher_rgb)
            plot_students.append(student_rgb)
            plot_gts.append(gt_rgb)

    # ============ Summary metrics ============
    print(f"\n=== Per-image metrics for split: {split_name} (4-ch PSNR/SSIM + balanced RGB LPIPS) ===")
    for res in results_per_image:
        print(f"\nImage: {res['id']}")
        print(f"  Input   -> PSNR: {res['psnr_input']:.2f}, SSIM: {res['ssim_input']:.4f}, LPIPS: {res['lpips_input']:.4f}")
        print(f"  Teacher -> PSNR: {res['psnr_teacher']:.2f}, SSIM: {res['ssim_teacher']:.4f}, LPIPS: {res['lpips_teacher']:.4f}, Time: {res['time_teacher']*1000:.2f} ms")
        print(f"  Student -> PSNR: {res['psnr_student']:.2f}, SSIM: {res['ssim_student']:.4f}, LPIPS: {res['lpips_student']:.4f}, Time: {res['time_student']*1000:.2f} ms")

    means = summarize_results(results_per_image)
    if means:
        print(f"\n=== Averages for split: {split_name} ===")
        print(f"psnr_input        {means['psnr_input']:.6f}")
        print(f"ssim_input        {means['ssim_input']:.6f}")
        print(f"lpips_input       {means['lpips_input']:.6f}")
        print(f"psnr_teacher      {means['psnr_teacher']:.6f}")
        print(f"ssim_teacher      {means['ssim_teacher']:.6f}")
        print(f"lpips_teacher     {means['lpips_teacher']:.6f}")
        print(f"psnr_student      {means['psnr_student']:.6f}")
        print(f"ssim_student      {means['ssim_student']:.6f}")
        print(f"lpips_student     {means['lpips_student']:.6f}")
        print(f"time_teacher_ms   {means['time_teacher_ms']:.6f}")
        print(f"time_student_ms   {means['time_student_ms']:.6f}")

    # ============ Save CSV ============
    csv_path = os.path.join(metrics_dir, f"{split_name}_metrics.csv")
    save_metrics_csv(results_per_image, csv_path)
    print(f"\nSaved per-image metrics CSV for split '{split_name}' to: {csv_path}")

    # ============ Plot grid ============
    if plot_inputs:
        rows = len(plot_inputs)
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(rows):
            img_id = plot_ids[i]

            axes[i, 0].imshow(plot_inputs[i])
            axes[i, 0].set_title(f"{split_name}:{img_id} - Input")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(plot_teachers[i])
            axes[i, 1].set_title("Teacher")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(plot_students[i])
            axes[i, 2].set_title("Student")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(plot_gts[i])
            axes[i, 3].set_title("GT")
            axes[i, 3].axis("off")

        plt.tight_layout()
        plt.show()

    return results_per_image


# ===================== MAIN =====================

print(f"Device: {DEVICE}")
print("Loading models...")

teacher = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(DEVICE)
teacher.load_state_dict(torch.load(TEACHER_WEIGHTS_PATH, map_location=DEVICE))
teacher.eval()
print(f"Loaded Teacher: {TEACHER_WEIGHTS_PATH}")

student = UNetStudent(in_channels=4, out_channels=4).to(DEVICE)
student.load_state_dict(torch.load(STUDENT_WEIGHTS_PATH, map_location=DEVICE))
student.eval()
print(f"Loaded Student: {STUDENT_WEIGHTS_PATH}")

lpips_model = lpips.LPIPS(net='vgg').to(DEVICE)

# ---- Validation split ----
val_results = evaluate_split(
    split_name="val",
    data_dir=VAL_DIR,
    teacher=teacher,
    student=student,
    lpips_model=lpips_model,
    patch_size=PATCH_SIZE,
    num_plot_examples=NUM_PLOT_EXAMPLES_VAL,
    patch_save_dir=PATCH_SAVE_DIR,
    metrics_dir=METRICS_DIR,
    device=DEVICE,
)

# ---- Test split (if available) ----
if TEST_DIR and os.path.isdir(TEST_DIR):
    test_results = evaluate_split(
        split_name="test",
        data_dir=TEST_DIR,
        teacher=teacher,
        student=student,
        lpips_model=lpips_model,
        patch_size=PATCH_SIZE,
        num_plot_examples=NUM_PLOT_EXAMPLES_TEST,
        patch_save_dir=PATCH_SAVE_DIR,
        metrics_dir=METRICS_DIR,
        device=DEVICE,
    )
else:
    print("\n[testing_full] TEST_DIR not found or not set; skipping test split.")

# ---- Zip and (optionally) download patches ----
zip_path = "patch_outputs_full.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(PATCH_SAVE_DIR):
        full_path = os.path.join(PATCH_SAVE_DIR, fname)
        zf.write(full_path, arcname=fname)

print(f"\nZipped 4-channel patches to: {zip_path}")

if IN_COLAB and colab_files is not None:
    try:
        colab_files.download(zip_path)
    except Exception as e:
        print(f"Auto-download failed: {e}")
        print("You can manually run in a cell:")
        print("from google.colab import files; files.download('patch_outputs_full.zip')")
else:
    print("Not running in Colab kernel; download the zip manually if needed.")
