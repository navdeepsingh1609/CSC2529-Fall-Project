# File: train_student_kd_quick.py
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import lpips

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

print("--- [train_student_kd_quick] Setting up system path...")
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)
print(f"--- [train_student_kd_quick] Added {mambair_path} to sys.path")

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent
from losses.frequency_loss import FFTAmplitudeLoss
from losses.pixel_loss import CharbonnierLoss

# ---------------- CONFIG (quick subset) ----------------
TRAIN_DIR = "data/UDC-SIT_subset/train"
VAL_DIR   = "data/UDC-SIT_subset/val"

PATCH_SIZE = 256
BATCH_SIZE = 16        # good for a quick sanity run
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1         # just to test pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEACHER_WEIGHTS = "teacher_quick_1epoch.pth"
STUDENT_SAVE_PATH = "student_quick_1epoch.pth"

# Loss weights
W_PIXEL   = 1.0
W_FEATURE = 0.5
W_FREQ    = 0.2
W_LPIPS   = 0.1
# ------------------------------------------------------

print("\n--- [train_student_kd_quick] Configuration ---")
print(f"Train Dir: {TRAIN_DIR}")
print(f"Val Dir:   {VAL_DIR}")
print(f"Patch Size: {PATCH_SIZE}, Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}, Learning Rate: {LEARNING_RATE}")
print(f"Device: {DEVICE}")
print(f"Teacher Weights: {TEACHER_WEIGHTS}")
print(f"Student Save Path: {STUDENT_SAVE_PATH}")
print("-----------------------------------\n")

def main():
    # 1. Data
    print("--- [train_student_kd_quick] Loading datasets...")
    train_dataset = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    val_dataset   = UDCDataset(VAL_DIR,   patch_size=PATCH_SIZE, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 2. Teacher (frozen)
    print(f"--- [train_student_kd_quick] Loading teacher from {TEACHER_WEIGHTS}...")
    teacher = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("--- [train_student_kd_quick] Teacher loaded and frozen.")

    # 3. Student
    student = UNetStudent(in_channels=4, out_channels=4).to(DEVICE)

    # 4. Losses
    pixel_loss_fn     = CharbonnierLoss().to(DEVICE)
    feature_loss_fn   = nn.L1Loss().to(DEVICE)
    frequency_loss_fn = FFTAmplitudeLoss(
        loss_weight=1.0,
        focus_low_freq=True,
        cutoff=0.25
    ).to(DEVICE)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)

    # 5. Optimizer
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)

    # 6. AMP scaler — FIXED: no 'cuda' argument
    scaler = GradScaler()

    print("--- [train_student_kd_quick] Starting quick KD training...")

    for epoch in range(NUM_EPOCHS):
        student.train()
        train_loss = 0.0

        for udc_batch, gt_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            udc_batch = udc_batch.to(DEVICE, non_blocking=True)
            gt_batch  = gt_batch.to(DEVICE,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Teacher forward (no grad)
            with torch.no_grad():
                with autocast():
                    teacher_out_4ch, teacher_feat_spatial, _ = teacher(udc_batch)

            # Student forward
            with autocast():
                student_out_4ch, student_features_raw = student(udc_batch)

                # Pixel loss
                loss_pixel = pixel_loss_fn(student_out_4ch, gt_batch)

                # Spatial feature KD
                student_features = F.interpolate(
                    student_features_raw,
                    size=teacher_feat_spatial.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                loss_feature_spatial = feature_loss_fn(student_features, teacher_feat_spatial)

                # LPIPS on pseudo-RGB
                pred_rgb_slice = student_out_4ch[:, :3, :, :]
                gt_rgb_slice   = gt_batch[:, :3, :, :]
                loss_lpips = perceptual_loss_fn(
                    pred_rgb_slice * 2 - 1,
                    gt_rgb_slice   * 2 - 1
                ).mean()

            # Frequency-based KD in float32
            loss_freq_out = frequency_loss_fn(
                student_out_4ch.float(),
                teacher_out_4ch.float()
            )
            loss_feature_freq = frequency_loss_fn(
                student_features.float(),
                teacher_feat_spatial.float()
            )

            loss_feature = 0.5 * loss_feature_spatial + 0.5 * loss_feature_freq
            loss_freq    = loss_freq_out

            total_loss = (W_PIXEL   * loss_pixel)   + \
                         (W_FEATURE * loss_feature) + \
                         (W_FREQ    * loss_freq)    + \
                         (W_LPIPS   * loss_lpips)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()

        print(f"--- [train_student_kd_quick] Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}")

        # Quick val
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for udc_batch, gt_batch in val_loader:
                udc_batch = udc_batch.to(DEVICE)
                gt_batch  = gt_batch.to(DEVICE)

                with autocast():
                    student_out_4ch, _ = student(udc_batch)
                    loss_val = pixel_loss_fn(student_out_4ch, gt_batch)

                val_loss += loss_val.item()

        print(f"--- [train_student_kd_quick] Epoch {epoch+1} Val Loss: {val_loss / len(val_loader):.4f}")

    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f"--- [train_student_kd_quick] Saved quick student checkpoint: {STUDENT_SAVE_PATH}")

if __name__ == "__main__":
    main()
