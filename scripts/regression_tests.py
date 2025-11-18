#!/usr/bin/env python3
"""
Regression tests for UDC-SIT project.

Runs:
- Test A: Dataset sanity (lengths, shapes, basic stats)
- Test B: Channel statistics (per-channel means for Input / GT / Teacher / Student)
- Test C: FFTAmplitudeLoss behaviour (same vs low-freq vs high-freq noise)
- Test D: Single KD training step sanity (loss components for student KD)

Run from project root:

    %cd /content/CSC2529-Fall-Project
    !python scripts/regression_tests.py
"""

import os
import sys
import time

# -----------------------------
# 1. PATH FIXES
# -----------------------------

# Project root: one level up from scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# MambaIR submodule root: contains `basicsr` etc.
MAMBAIR_ROOT = os.path.join(PROJECT_ROOT, "models", "external", "MambaIR")
if MAMBAIR_ROOT not in sys.path:
    sys.path.insert(0, MAMBAIR_ROOT)
    print(f"[regression_tests] Added MambaIR path: {MAMBAIR_ROOT}")

# -----------------------------
# 2. IMPORTS
# -----------------------------

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import lpips
import skimage.metrics

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent
from losses.frequency_loss import FFTAmplitudeLoss

# -----------------------------
# 3. CONFIG
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "/content/dataset/UDC-SIT/training"
VAL_DIR = "/content/dataset/UDC-SIT/validation"
PATCH_SIZE = 256

TEACHER_WEIGHTS = "teacher_4ch_26_epochs_bs9.pth"
STUDENT_WEIGHTS = "student_distilled_4ch_full_data.pth"

# -----------------------------
# 4. TEST A – DATASET SANITY
# -----------------------------


def test_a_dataset_sanity():
    print("=== Test A: Dataset sanity check ===")

    train_ds = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    val_ds = UDCDataset(VAL_DIR, patch_size=PATCH_SIZE, is_train=False)

    print(f"Train len: {len(train_ds)}, Val len: {len(val_ds)}")

    # Grab one sample from train
    udc, gt = train_ds[0]
    print("Train sample shapes:", udc.shape, gt.shape)

    print("Train sample stats (should be ~[0,1]):")
    print(
        f"  UDC min/max/mean: {udc.min().item()} "
        f"{udc.max().item()} {udc.mean().item()}"
    )
    print(
        f"  GT  min/max/mean: {gt.min().item()} "
        f"{gt.max().item()} {gt.mean().item()}"
    )

    # Round-trip to (H, W, 4) just to confirm ordering
    udc_hw4 = udc.permute(1, 2, 0).cpu().numpy()
    print("Round-trip shape (H,W,4):", udc_hw4.shape)
    print()


# -----------------------------
# 5. TEST B – CHANNEL STATS
# -----------------------------


def test_b_channel_means():
    print("=== Test B: Channel means (GR, R, B, GB) ===")

    # One validation sample
    val_ds = UDCDataset(VAL_DIR, patch_size=PATCH_SIZE, is_train=False)
    udc, gt = val_ds[0]  # (4,H,W) in [0,1]

    # Load models + weights
    teacher = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=DEVICE))
    teacher.eval()

    student = UNetStudent(in_channels=4, out_channels=4).to(DEVICE)
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=DEVICE))
    student.eval()

    with torch.no_grad():
        udc_batch = udc.unsqueeze(0).to(DEVICE)  # (1,4,H,W)
        teacher_out, _, _ = teacher(udc_batch)   # (1,4,H,W)
        student_out, _ = student(udc_batch)      # (1,4,H,W)

    # Compute per-channel means
    def channel_means(x4chw: torch.Tensor):
        # x4chw: (1,4,H,W)
        arr = x4chw.squeeze(0).cpu().numpy()  # (4,H,W)
        means = arr.reshape(4, -1).mean(axis=1)
        return means

    input_means = channel_means(udc_batch)
    gt_means = channel_means(gt.unsqueeze(0).to(DEVICE))
    teacher_means = channel_means(teacher_out)
    student_means = channel_means(student_out)

    print("=== Channel means (normalized [0,1]) ===")
    print(f"Input channel means (GR,R,B,GB): {input_means}")
    print(f"GT channel means    (GR,R,B,GB): {gt_means}")
    print(f"Teacher channel means (GR,R,B,GB): {teacher_means}")
    print(f"Student channel means (GR,R,B,GB): {student_means}")
    print()


# -----------------------------
# 6. TEST C – FFT LOSS BEHAVIOUR
# -----------------------------


def test_c_fft_loss():
    print("=== Test C: FFTAmplitudeLoss behaviour ===")
    loss_fn = FFTAmplitudeLoss().to(DEVICE)

    # Synthetic "image"
    torch.manual_seed(0)
    x = torch.rand(1, 4, 64, 64, device=DEVICE)

    # Case 1: identical
    y_same = x.clone()
    loss_same = loss_fn(x, y_same).item()

    # Case 2: low-frequency bias – add smooth offset
    y_lowfreq = x + 0.05
    loss_low = loss_fn(x, y_lowfreq).item()

    # Case 3: high-frequency noise – add random noise
    noise = 0.05 * torch.randn_like(x)
    y_highfreq = x + noise
    loss_high = loss_fn(x, y_highfreq).item()

    print(f"FFT loss (same):        {loss_same}")
    print(f"FFT loss (low-freq bias): {loss_low}")
    print(f"FFT loss (high-freq noise): {loss_high}")
    print()


# -----------------------------
# 7. TEST D – KD STEP SANITY
# -----------------------------


def test_d_kd_step():
    print("=== Test D: Single KD training step sanity ===")

    # 1. Dataset / DataLoader (one small batch)
    train_ds = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True
    )

    # 2. Models
    teacher = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = UNetStudent(in_channels=4, out_channels=4).to(DEVICE)
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=DEVICE))
    student.train()

    # 3. Losses
    pixel_loss_fn = torch.nn.L1Loss().to(DEVICE)
    feature_loss_fn = torch.nn.L1Loss().to(DEVICE)
    freq_loss_fn = FFTAmplitudeLoss().to(DEVICE)
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(DEVICE)

    # 4. Optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=2e-4)

    # 5. Loss weights (same as train_student_kd.py)
    W_PIXEL = 1.0
    W_FEATURE = 0.5
    W_FREQ = 0.2
    W_LPIPS = 0.1

    # 6. One batch
    udc_batch, gt_batch = next(iter(train_loader))
    udc_batch = udc_batch.to(DEVICE, non_blocking=True)  # (B,4,H,W)
    gt_batch = gt_batch.to(DEVICE, non_blocking=True)    # (B,4,H,W)

    with torch.no_grad():
        teacher_out, teacher_feat_spatial, _ = teacher(udc_batch)

    student_out, student_features_raw = student(udc_batch)

    # Pixel loss
    loss_pixel = pixel_loss_fn(student_out, gt_batch)

    # Feature KD loss
    student_features_resized = F.interpolate(
        student_features_raw,
        size=teacher_feat_spatial.shape[2:],
        mode="bilinear",
        align_corners=False,
    )
    loss_feature = feature_loss_fn(student_features_resized, teacher_feat_spatial)

    # Frequency KD loss
    loss_freq = freq_loss_fn(student_out.float(), teacher_out.float())

    # LPIPS on RGB slices
    pred_rgb = student_out[:, :3, :, :]
    gt_rgb = gt_batch[:, :3, :, :]
    lpips_val = lpips_loss_fn(
        (pred_rgb * 2 - 1), (gt_rgb * 2 - 1)
    ).mean()
    loss_lpips = lpips_val

    total_loss = (
        W_PIXEL * loss_pixel
        + W_FEATURE * loss_feature
        + W_FREQ * loss_freq
        + W_LPIPS * loss_lpips
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("KD step losses:")
    print(f"  pixel   : {loss_pixel.item()}")
    print(f"  feature : {loss_feature.item()}")
    print(f"  freq    : {loss_freq.item()}")
    print(f"  lpips   : {loss_lpips.item()}")
    print(f"  total   : {total_loss.item()}")
    print()


# -----------------------------
# 8. MAIN
# -----------------------------

if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"MAMBAIR_ROOT = {MAMBAIR_ROOT}")
    print(f"Device: {DEVICE}")
    print()

    # Run tests in order
    test_a_dataset_sanity()
    test_b_channel_means()
    test_c_fft_loss()
    test_d_kd_step()

    print("All regression tests finished.")
