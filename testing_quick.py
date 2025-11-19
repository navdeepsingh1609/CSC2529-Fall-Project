# File: testing_quick.py
import os, sys, glob, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import skimage.metrics
import lpips

# ===================== CONFIG (quick subset) =====================
VAL_DIR = "data/UDC-SIT_subset/val"

TEACHER_WEIGHTS_PATH = "teacher_quick_1epoch.pth"
STUDENT_WEIGHTS_PATH = "student_quick_1epoch.pth"

NUM_EXAMPLES = 5       # number of val images to test (<= 10)
PATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SAVE_DIR = "patch_outputs_quick"
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

def load_center_patch_4ch(npy_path, patch_size=256):
    arr = np.load(npy_path).astype(np.float32)   # (H,W,4)
    H, W, C = arr.shape
    ps = patch_size
    r0 = (H - ps) // 2
    c0 = (W - ps) // 2
    r1, c1 = r0 + ps, c0 + ps
    patch = arr[r0:r1, c0:c1, :]
    return patch

def bayer4_to_rgb_naive(bayer_4ch):
    arr = np.asarray(bayer_4ch, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")

    if arr.shape[-1] == 4:
        GR = arr[..., 0]
        R  = arr[..., 1]
        B  = arr[..., 2]
        GB = arr[..., 3]
    elif arr.shape[0] == 4:
        GR = arr[0]
        R  = arr[1]
        B  = arr[2]
        GB = arr[3]
    else:
        raise ValueError(f"Expected shape (...,4), got {arr.shape}")

    if arr.max() > 2.0:
        scale = 1023.0
    else:
        scale = 1.0

    Rn = R / scale
    Bn = B / scale
    Gn = (GR + GB) / (2.0 * scale)

    rgb = np.stack([Rn, Gn, Bn], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb

def numpy_to_lpips_tensor(img_np, device):
    t = torch.from_numpy(img_np).float().to(device)
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    t = t * 2.0 - 1.0
    return t

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

input_paths = sorted(glob.glob(os.path.join(VAL_DIR, "input", "*.npy")))
if not input_paths:
    raise RuntimeError(f"No .npy files found in {os.path.join(VAL_DIR, 'input')}")

input_paths = input_paths[:min(NUM_EXAMPLES, len(input_paths))]
print("\nUsing validation images:")
for p in input_paths:
    print("  ", os.path.basename(p))

results_per_image = []

plot_inputs   = []
plot_teachers = []
plot_students = []
plot_gts      = []

for idx, input_path in enumerate(input_paths):
    img_id = os.path.splitext(os.path.basename(input_path))[0]
    gt_path = input_path.replace("/input/", "/GT/")

    print(f"\n[{idx+1}/{len(input_paths)}] Image ID: {img_id}")

    # 1) Load center patches (H,W,4) in [0,1023]
    inp_patch_4ch = load_center_patch_4ch(input_path, PATCH_SIZE)
    gt_patch_4ch  = load_center_patch_4ch(gt_path,    PATCH_SIZE)

    # 2) Normalize to [0,1]
    inp_norm = (inp_patch_4ch / 1023.0).astype(np.float32)
    gt_norm  = (gt_patch_4ch  / 1023.0).astype(np.float32)

    inp_chw = np.transpose(inp_norm, (2, 0, 1))  # (4,H,W)
    gt_chw  = np.transpose(gt_norm,  (2, 0, 1))

    inp_bchw = torch.from_numpy(inp_chw).unsqueeze(0).to(DEVICE)
    gt_bchw  = torch.from_numpy(gt_chw).unsqueeze(0).to(DEVICE)

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

    # 5) Back to numpy (H,W,4) in [0,1]
    inp_4n     = inp_norm
    gt_4n      = gt_norm
    teacher_4n = teacher_out_bchw.squeeze(0).permute(1, 2, 0).cpu().numpy()
    student_4n = student_out_bchw.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 6) 4-channel PSNR/SSIM
    psnr_input   = skimage.metrics.peak_signal_noise_ratio(gt_4n, inp_4n, data_range=1.0)
    ssim_input   = skimage.metrics.structural_similarity(gt_4n, inp_4n, data_range=1.0, channel_axis=-1)

    psnr_teacher = skimage.metrics.peak_signal_noise_ratio(gt_4n, teacher_4n, data_range=1.0)
    ssim_teacher = skimage.metrics.structural_similarity(gt_4n, teacher_4n, data_range=1.0, channel_axis=-1)

    psnr_student = skimage.metrics.peak_signal_noise_ratio(gt_4n, student_4n, data_range=1.0)
    ssim_student = skimage.metrics.structural_similarity(gt_4n, student_4n, data_range=1.0, channel_axis=-1)

    # 7) LPIPS on naive RGB
    inp_rgb     = bayer4_to_rgb_naive(inp_patch_4ch)
    gt_rgb      = bayer4_to_rgb_naive(gt_patch_4ch)
    teacher_rgb = bayer4_to_rgb_naive(teacher_4n * 1023.0)
    student_rgb = bayer4_to_rgb_naive(student_4n * 1023.0)

    gt_lp      = numpy_to_lpips_tensor(gt_rgb,      DEVICE)
    inp_lp     = numpy_to_lpips_tensor(inp_rgb,     DEVICE)
    teacher_lp = numpy_to_lpips_tensor(teacher_rgb, DEVICE)
    student_lp = numpy_to_lpips_tensor(student_rgb, DEVICE)

    with torch.no_grad():
        lpips_input   = lpips_model(gt_lp, inp_lp).item()
        lpips_teacher = lpips_model(gt_lp, teacher_lp).item()
        lpips_student = lpips_model(gt_lp, student_lp).item()

    # 8) Save metrics
    results_per_image.append({
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
    })

    print(f"  Input   -> PSNR: {psnr_input:.2f}, SSIM: {ssim_input:.4f}, LPIPS: {lpips_input:.4f}")
    print(f"  Teacher -> PSNR: {psnr_teacher:.2f}, SSIM: {ssim_teacher:.4f}, LPIPS: {lpips_teacher:.4f}, Time: {t_time*1000:.2f} ms")
    print(f"  Student -> PSNR: {psnr_student:.2f}, SSIM: {ssim_student:.4f}, LPIPS: {lpips_student:.4f}, Time: {s_time*1000:.2f} ms")

    # store for plotting
    plot_inputs.append(inp_rgb)
    plot_teachers.append(teacher_rgb)
    plot_students.append(student_rgb)
    plot_gts.append(gt_rgb)

# ============ Plot grid ============

rows = len(results_per_image)
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

if rows == 1:
    axes = np.expand_dims(axes, axis=0)

for i, res in enumerate(results_per_image):
    img_id = res["id"]

    axes[i, 0].imshow(plot_inputs[i])
    axes[i, 0].set_title(f"{img_id} - Input")
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

print("\n\n=== Per-image metrics (4-ch PSNR/SSIM + RGB LPIPS) ===")
for res in results_per_image:
    print(f"\nImage: {res['id']}")
    print(f"  Input   -> PSNR: {res['psnr_input']:.2f}, SSIM: {res['ssim_input']:.4f}, LPIPS: {res['lpips_input']:.4f}")
    print(f"  Teacher -> PSNR: {res['psnr_teacher']:.2f}, SSIM: {res['ssim_teacher']:.4f}, LPIPS: {res['lpips_teacher']:.4f}, Time: {res['time_teacher']*1000:.2f} ms")
    print(f"  Student -> PSNR: {res['psnr_student']:.2f}, SSIM: {res['ssim_student']:.4f}, LPIPS: {res['lpips_student']:.4f}, Time: {res['time_student']*1000:.2f} ms")
