# scripts/regression_tests.py

import os
import sys

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ------------------------------------

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent
from losses.frequency_loss import FFTAmplitudeLoss

import torch
import torch.nn.functional as F
import numpy as np
import lpips
import skimage.metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "/content/dataset/UDC-SIT/training"
VAL_DIR   = "/content/dataset/UDC-SIT/validation"
PATCH_SIZE = 256
TEACHER_WEIGHTS = "teacher_4ch_26_epochs_bs9.pth"
STUDENT_WEIGHTS = "student_distilled_4ch_full_data.pth"


def test_dataset():
    print("=== [Test A] Dataset ===")
    train_ds = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    val_ds   = UDCDataset(VAL_DIR,   patch_size=PATCH_SIZE, is_train=False)
    print(f"Train len: {len(train_ds)}, Val len: {len(val_ds)}")

    x, y = train_ds[0]
    print("Shape:", x.shape, y.shape)
    print("UDC min/max/mean:", x.min().item(), x.max().item(), x.mean().item())
    print("GT  min/max/mean:", y.min().item(), y.max().item(), y.mean().item())

    x_hw4 = x.permute(1,2,0).numpy()
    print("Round-trip shape (H,W,4):", x_hw4.shape)
    print()


def test_channel_means(teacher, student):
    print("=== [Test B] Channel means ===")
    # Use a fixed ID to keep this consistent
    IMG_ID = "1004"
    input_path = os.path.join(VAL_DIR, "input", f"{IMG_ID}.npy")
    gt_path    = os.path.join(VAL_DIR, "GT",    f"{IMG_ID}.npy")

    arr_in = np.load(input_path).astype(np.float32) / 1023.0
    arr_gt = np.load(gt_path).astype(np.float32) / 1023.0

    # Center crop
    H, W, C = arr_in.shape
    ps = PATCH_SIZE
    r0 = (H - ps) // 2
    c0 = (W - ps) // 2
    inp = arr_in[r0:r0+ps, c0:c0+ps, :]
    gt  = arr_gt[r0:r0+ps, c0:c0+ps, :]

    inp_chw = torch.from_numpy(np.transpose(inp, (2,0,1))).unsqueeze(0).to(DEVICE)
    gt_chw  = torch.from_numpy(np.transpose(gt,  (2,0,1))).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        t_out, _, _ = teacher(inp_chw)
        s_out, _    = student(inp_chw)

    t_out = torch.clamp(t_out, 0, 1)
    s_out = torch.clamp(s_out, 0, 1)

    def ch_means(name, bchw):
        arr = bchw.squeeze(0).cpu().numpy()
        m = arr.mean(axis=(1,2))
        print(f"{name} (GR,R,B,GB): {m}")

    ch_means("Input",   inp_chw)
    ch_means("GT",      gt_chw)
    ch_means("Teacher", t_out)
    ch_means("Student", s_out)
    print()


def test_fft_loss():
    print("=== [Test C] FFT loss ===")
    fft_loss = FFTAmplitudeLoss().to(DEVICE)
    x = torch.rand(1, 4, 64, 64, device=DEVICE)
    y = x.clone()
    y_low  = y + 0.1
    noise  = (torch.rand_like(x) - 0.5) * 0.1
    y_high = y + noise

    print("same     :", fft_loss(x, y).item())
    print("low freq :", fft_loss(x, y_low).item())
    print("high freq:", fft_loss(x, y_high).item())
    print()


def test_kd_step(teacher, student):
    print("=== [Test D] KD single step ===")
    train_ds = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

    pixel_loss_fn     = torch.nn.L1Loss().to(DEVICE)
    feature_loss_fn   = torch.nn.L1Loss().to(DEVICE)
    freq_loss_fn      = FFTAmplitudeLoss().to(DEVICE)
    perceptual_loss_fn= lpips.LPIPS(net='vgg').to(DEVICE)

    opt = torch.optim.Adam(student.parameters(), lr=2e-4)

    udc, gt = next(iter(train_loader))
    udc = udc.to(DEVICE)
    gt  = gt.to(DEVICE)

    teacher.eval()
    student.train()
    opt.zero_grad()

    with torch.no_grad():
        t_out, t_feat, _ = teacher(udc)

    s_out, s_feat_raw = student(udc)

    loss_pixel = pixel_loss_fn(s_out, gt)

    s_feat = F.interpolate(
        s_feat_raw,
        size=t_feat.shape[2:],
        mode="bilinear",
        align_corners=False
    )
    loss_feat = feature_loss_fn(s_feat, t_feat)

    loss_freq = freq_loss_fn(s_out, t_out)

    pred_rgb = s_out[:, :3]
    gt_rgb   = gt[:, :3]
    loss_lpips = perceptual_loss_fn(pred_rgb*2-1, gt_rgb*2-1).mean()

    total = 1.0*loss_pixel + 0.5*loss_feat + 0.2*loss_freq + 0.1*loss_lpips
    total.backward()
    opt.step()

    print("pixel :", loss_pixel.item())
    print("feat  :", loss_feat.item())
    print("freq  :", loss_freq.item())
    print("lpips :", loss_lpips.item())
    print("total :", total.item())
    print()


def test_metrics_on_five(teacher, student):
    print("=== [Test E] Metrics on 5 center patches ===")
    input_files = sorted(os.listdir(os.path.join(VAL_DIR, "input")))
    ids = [os.path.splitext(f)[0] for f in input_files][:5]
    print("Using images:", ids)

    loss_fn_lpips = lpips.LPIPS(net='vgg').to(DEVICE)

    for img_id in ids:
        inp_path = os.path.join(VAL_DIR, "input", f"{img_id}.npy")
        gt_path  = os.path.join(VAL_DIR, "GT",    f"{img_id}.npy")

        arr_in = np.load(inp_path).astype(np.float32) / 1023.0
        arr_gt = np.load(gt_path).astype(np.float32) / 1023.0

        H, W, C = arr_in.shape
        ps = PATCH_SIZE
        r0 = (H - ps) // 2
        c0 = (W - ps) // 2
        inp = arr_in[r0:r0+ps, c0:c0+ps, :]
        gt  = arr_gt[r0:r0+ps, c0:c0+ps, :]

        inp_chw = torch.from_numpy(np.transpose(inp, (2,0,1))).unsqueeze(0).to(DEVICE)
        gt_chw  = torch.from_numpy(np.transpose(gt,  (2,0,1))).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            t_start = time.time()
            t_out, _, _ = teacher(inp_chw)
            t_time = (time.time() - t_start) * 1000.0

            s_start = time.time()
            s_out, _ = student(inp_chw)
            s_time = (time.time() - s_start) * 1000.0

        # 4-ch PSNR/SSIM
        inp_np = inp_chw.squeeze(0).cpu().numpy()
        gt_np  = gt_chw.squeeze(0).cpu().numpy()
        t_np   = torch.clamp(t_out, 0, 1).squeeze(0).cpu().numpy()
        s_np   = torch.clamp(s_out, 0, 1).squeeze(0).cpu().numpy()

        def psnr_ssim_4ch(x, y):
            # Average PSNR and SSIM across channels
            psnrs = []
            ssims = []
            for c in range(4):
                psnrs.append(skimage.metrics.peak_signal_noise_ratio(
                    y[c], x[c], data_range=1.0
                ))
                ssims.append(skimage.metrics.structural_similarity(
                    y[c], x[c], data_range=1.0
                ))
            return float(np.mean(psnrs)), float(np.mean(ssims))

        psnr_in, ssim_in = psnr_ssim_4ch(inp_np, gt_np)
        psnr_t,  ssim_t  = psnr_ssim_4ch(t_np,   gt_np)
        psnr_s,  ssim_s  = psnr_ssim_4ch(s_np,   gt_np)

        # Simple LPIPS on “fake RGB”: take channels 0,1,2 as RGB
        def lpips_rgb(x3, y3):
            x_t = torch.from_numpy(x3).permute(2,0,1).unsqueeze(0).to(DEVICE)
            y_t = torch.from_numpy(y3).permute(2,0,1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                return float(loss_fn_lpips(x_t*2-1, y_t*2-1).item())

        inp_rgb = np.stack([inp_np[0], inp_np[1], inp_np[2]], axis=-1)
        gt_rgb  = np.stack([gt_np[0],  gt_np[1],  gt_np[2]],  axis=-1)
        t_rgb   = np.stack([t_np[0],   t_np[1],   t_np[2]],   axis=-1)
        s_rgb   = np.stack([s_np[0],   s_np[1],   s_np[2]],   axis=-1)

        lpips_in = lpips_rgb(inp_rgb, gt_rgb)
        lpips_t  = lpips_rgb(t_rgb,   gt_rgb)
        lpips_s  = lpips_rgb(s_rgb,   gt_rgb)

        print(f"Image {img_id}:")
        print(f"  Input   -> PSNR: {psnr_in:.2f}, SSIM: {ssim_in:.4f}, LPIPS: {lpips_in:.4f}")
        print(f"  Teacher -> PSNR: {psnr_t:.2f}, SSIM: {ssim_t:.4f}, LPIPS: {lpips_t:.4f}, Time: {t_time:.2f} ms")
        print(f"  Student -> PSNR: {psnr_s:.2f}, SSIM: {ssim_s:.4f}, LPIPS: {lpips_s:.4f}, Time: {s_time:.2f} ms")
        print()


if __name__ == "__main__":
    print("Device:", DEVICE)
    print("Loading models...")
    teacher = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=DEVICE))
    teacher.eval()

    student = UNetStudent(in_channels=4, out_channels=4).to(DEVICE)
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=DEVICE))
    student.eval()

    test_dataset()
    test_fft_loss()
    test_channel_means(teacher, student)
    test_kd_step(teacher, student)
    test_metrics_on_five(teacher, student)
