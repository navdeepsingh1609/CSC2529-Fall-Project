import os, sys, glob, time, zipfile
import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage.metrics
import lpips
import pandas as pd
from google.colab import files

# ===================== CONFIG =====================
DATA_ROOT = "/content/dataset/UDC-SIT"
SPLIT = "testing"   # <-- change to "validation" or "training" as needed
TEACHER_WEIGHTS_PATH = "teacher_4ch_26_epochs_bs9.pth"
STUDENT_WEIGHTS_PATH = "student_distilled_4ch_full_data.pth"
NUM_EXAMPLES = None    # None = use ALL images in this split, or set e.g. 5
PATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SAVE_DIR = f"patch_outputs_{SPLIT}"  # 4-ch .npy patches will be saved here
os.makedirs(PATCH_SAVE_DIR, exist_ok=True)
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
    return a center patch of size (patch_size, patch_size, 4) in [0,1023].
    """
    arr = np.load(npy_path).astype(np.float32)   # (H,W,4)
    H, W, C = arr.shape
    ps = patch_size
    r0 = (H - ps) // 2
    c0 = (W - ps) // 2
    r1, c1 = r0 + ps, c0 + ps
    patch = arr[r0:r1, c0:c1, :]   # (ps,ps,4)
    return patch


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

    G = (GR + GB) / (2.0 * scale)
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

# ============ Load models ============

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

# ============ Collect file list ============

input_dir = os.path.join(DATA_ROOT, SPLIT, "input")
gt_dir    = os.path.join(DATA_ROOT, SPLIT, "GT")

input_paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
if not input_paths:
    raise RuntimeError(f"No .npy files found in {input_dir}")

if NUM_EXAMPLES is not None:
    input_paths = input_paths[:NUM_EXAMPLES]

have_gt = os.path.isdir(gt_dir) and len(glob.glob(os.path.join(gt_dir, "*.npy"))) > 0

print(f"\nEvaluating split: {SPLIT}")
print(f"Found {len(input_paths)} input files.")
print(f"Ground-truth available: {have_gt}")

# ============ Metrics containers ============

results_per_image = []  # each element is a dict with metrics

# For plotting: store RGB views (optional)
plot_inputs   = []
plot_teachers = []
plot_students = []
plot_gts      = []

# ============ Main loop ============

for idx, input_path in enumerate(input_paths):
    img_id = os.path.splitext(os.path.basename(input_path))[0]
    gt_path = os.path.join(gt_dir, os.path.basename(input_path)) if have_gt else None

    print(f"\n[{idx+1}/{len(input_paths)}] Image ID: {img_id}")

    inp_patch_4ch = load_center_patch_4ch(input_path, PATCH_SIZE)

    if have_gt:
        gt_patch_4ch  = load_center_patch_4ch(gt_path, PATCH_SIZE)

    # Normalize to [0,1] and convert to (1,4,H,W)
    inp_norm = (inp_patch_4ch / 1023.0).astype(np.float32)  # (H,W,4)
    inp_chw  = np.transpose(inp_norm, (2, 0, 1))            # (4,H,W)
    inp_bchw = torch.from_numpy(inp_chw).unsqueeze(0).to(DEVICE)

    if have_gt:
        gt_norm = (gt_patch_4ch / 1023.0).astype(np.float32)
        gt_chw  = np.transpose(gt_norm, (2, 0, 1))
        gt_bchw = torch.from_numpy(gt_chw).unsqueeze(0).to(DEVICE)

    # 1) Forward through models
    with torch.no_grad():
        t_start = time.time()
        teacher_out_bchw, _, _ = teacher(inp_bchw)
        t_time = time.time() - t_start

        t_start = time.time()
        student_out_bchw, _ = student(inp_bchw)
        s_time = time.time() - t_start

    # 2) Clamp outputs to [0,1] for metrics
    teacher_out_bchw = torch.clamp(teacher_out_bchw, 0.0, 1.0)
    student_out_bchw = torch.clamp(student_out_bchw, 0.0, 1.0)

    # 3) Back to numpy (H,W,4) in [0,1]
    teacher_4n = teacher_out_bchw.squeeze(0).permute(1, 2, 0).cpu().numpy()
    student_4n = student_out_bchw.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Initialize metric dict for this image
    m = {
        "id": img_id,
        "psnr_input":   None,
        "ssim_input":   None,
        "lpips_input":  None,
        "psnr_teacher": None,
        "ssim_teacher": None,
        "lpips_teacher": None,
        "time_teacher_ms": t_time * 1000.0,
        "psnr_student": None,
        "ssim_student": None,
        "lpips_student": None,
        "time_student_ms": s_time * 1000.0,
    }

    if have_gt:
        # 4) 4-channel PSNR/SSIM (Bayer space)
        inp_4n = inp_norm
        gt_4n  = gt_norm

        m["psnr_input"]   = skimage.metrics.peak_signal_noise_ratio(gt_4n, inp_4n, data_range=1.0)
        m["ssim_input"]   = skimage.metrics.structural_similarity(gt_4n, inp_4n, data_range=1.0, channel_axis=-1)

        m["psnr_teacher"] = skimage.metrics.peak_signal_noise_ratio(gt_4n, teacher_4n, data_range=1.0)
        m["ssim_teacher"] = skimage.metrics.structural_similarity(gt_4n, teacher_4n, data_range=1.0, channel_axis=-1)

        m["psnr_student"] = skimage.metrics.peak_signal_noise_ratio(gt_4n, student_4n, data_range=1.0)
        m["ssim_student"] = skimage.metrics.structural_similarity(gt_4n, student_4n, data_range=1.0, channel_axis=-1)

        # 5) LPIPS on balanced RGB
        inp_rgb     = bayer4_to_rgb_balanced(inp_patch_4ch)
        gt_rgb      = bayer4_to_rgb_balanced(gt_patch_4ch)
        teacher_rgb = bayer4_to_rgb_balanced(teacher_4n * 1023.0)
        student_rgb = bayer4_to_rgb_balanced(student_4n * 1023.0)

        gt_lp      = numpy_to_lpips_tensor(gt_rgb,      DEVICE)
        inp_lp     = numpy_to_lpips_tensor(inp_rgb,     DEVICE)
        teacher_lp = numpy_to_lpips_tensor(teacher_rgb, DEVICE)
        student_lp = numpy_to_lpips_tensor(student_rgb, DEVICE)

        with torch.no_grad():
            m["lpips_input"]   = lpips_model(gt_lp, inp_lp).item()
            m["lpips_teacher"] = lpips_model(gt_lp, teacher_lp).item()
            m["lpips_student"] = lpips_model(gt_lp, student_lp).item()

        # Store images for plotting
        plot_inputs.append(inp_rgb)
        plot_teachers.append(teacher_rgb)
        plot_students.append(student_rgb)
        plot_gts.append(gt_rgb)

        # Print a compact summary for this image
        print(f"  Input   -> PSNR: {m['psnr_input']:.2f}, SSIM: {m['ssim_input']:.4f}, LPIPS: {m['lpips_input']:.4f}")
        print(f"  Teacher -> PSNR: {m['psnr_teacher']:.2f}, SSIM: {m['ssim_teacher']:.4f}, LPIPS: {m['lpips_teacher']:.4f}, Time: {m['time_teacher_ms']:.2f} ms")
        print(f"  Student -> PSNR: {m['psnr_student']:.2f}, SSIM: {m['ssim_student']:.4f}, LPIPS: {m['lpips_student']:.4f}, Time: {m['time_student_ms']:.2f} ms")
    else:
        print("  No GT available for this split - metrics skipped, only timings logged.")

    # 6) Save 4-ch patches as .npy (in original [0,1023] scale)
    np.save(os.path.join(PATCH_SAVE_DIR, f"{img_id}_input_patch.npy"),   inp_patch_4ch)
    if have_gt:
        np.save(os.path.join(PATCH_SAVE_DIR, f"{img_id}_gt_patch.npy"),      gt_patch_4ch)
    np.save(os.path.join(PATCH_SAVE_DIR, f"{img_id}_teacher_patch.npy"), teacher_4n * 1023.0)
    np.save(os.path.join(PATCH_SAVE_DIR, f"{img_id}_student_patch.npy"), student_4n * 1023.0)

    # 7) Save this record
    results_per_image.append(m)

# ============ Convert to DataFrame and print summary ============

df = pd.DataFrame(results_per_image)
csv_name = f"metrics_{SPLIT}.csv"
df.to_csv(csv_name, index=False)
print(f"\nSaved per-image metrics to {csv_name}")

if have_gt:
    # Only average over non-NaN entries
    cols_to_avg = [
        "psnr_input", "ssim_input", "lpips_input",
        "psnr_teacher", "ssim_teacher", "lpips_teacher",
        "psnr_student", "ssim_student", "lpips_student",
        "time_teacher_ms", "time_student_ms",
    ]
    print(f"\n=== Averages for split: {SPLIT} ===")
    print(df[cols_to_avg].mean(numeric_only=True))

    # Optional: quick 5×4 grid of first few images
    num_show = min(5, len(plot_inputs))
    if num_show > 0:
        fig, axes = plt.subplots(num_show, 4, figsize=(20, 4 * num_show))
        if num_show == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(num_show):
            axes[i, 0].imshow(plot_inputs[i])
            axes[i, 0].set_title(f"{df['id'][i]} - Input")
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

# ============ Zip and download patches + CSV ============

zip_path = f"patch_outputs_{SPLIT}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(PATCH_SAVE_DIR):
        full_path = os.path.join(PATCH_SAVE_DIR, fname)
        zf.write(full_path, arcname=fname)
    zf.write(csv_name, arcname=csv_name)

print(f"\nZipped patches + metrics CSV to: {zip_path}")
files.download(zip_path)
