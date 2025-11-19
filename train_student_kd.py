# File: train_student_kd.py
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import lpips
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

print("--- [train_student_kd] Setting up system path...")
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)
print(f"--- [train_student_kd] Added {mambair_path} to sys.path")

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent
from losses.frequency_loss import FFTAmpPhaseMultiScaleLoss
from losses.pixel_loss import CharbonnierLoss

# ---------------- CONFIG ----------------
TRAIN_DIR = "/content/dataset/UDC-SIT/training"
VAL_DIR   = "/content/dataset/UDC-SIT/validation"

PATCH_SIZE = 256

BATCH_SIZE    = 64
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 20
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

TEACHER_WEIGHTS = "teacher_4ch_22epochs_bs8.pth"

LOCAL_LATEST_STUDENT = "student_4ch_latest.pth"
FINAL_STUDENT_NAME   = "student_distilled_4ch_full_data.pth"

DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/Computational Imaging Project/checkpoints"
os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
DRIVE_LATEST_STUDENT = os.path.join(DRIVE_CHECKPOINT_DIR, "student_latest.pth")
DRIVE_BEST_STUDENT   = os.path.join(DRIVE_CHECKPOINT_DIR, "student_best.pth")

# Loss history paths
LOCAL_LOSS_HISTORY = "student_loss_history_full.npz"
DRIVE_LOSS_HISTORY = os.path.join(DRIVE_CHECKPOINT_DIR, "student_loss_history_full.npz")
LOSS_PLOT_PNG      = "student_loss_curves_full.png"
DRIVE_LOSS_PLOT    = os.path.join(DRIVE_CHECKPOINT_DIR, "student_loss_curves_full.png")

# Loss weights
W_PIXEL   = 1.0
W_FEATURE = 0.5    # spatial + freq KD
W_FREQ    = 0.3    # AP+multi-scale KD on outputs
W_LPIPS   = 0.1
# --------------------------------------

print("\n--- [train_student_kd] Configuration ---")
print(f"Train Dir: {TRAIN_DIR}")
print(f"Val Dir:   {VAL_DIR}")
print(f"Patch Size: {PATCH_SIZE}, Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}, Learning Rate: {LEARNING_RATE}")
print(f"Device: {DEVICE}")
print(f"Teacher Weights: {TEACHER_WEIGHTS}")
print(f"Local latest student ckpt: {LOCAL_LATEST_STUDENT}")
print(f"Drive latest student ckpt: {DRIVE_LATEST_STUDENT}")
print("-----------------------------------\n")


def main():
    # 1. Data
    print("--- [train_student_kd] Loading datasets...")
    train_dataset = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    val_dataset   = UDCDataset(VAL_DIR,   patch_size=PATCH_SIZE, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"--- [train_student_kd] Device: {DEVICE}")

    # 2. Teacher (frozen)
    print(f"--- [train_student_kd] Loading teacher model from {TEACHER_WEIGHTS}...")
    teacher = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("--- [train_student_kd] Teacher model loaded and frozen.")

    # 3. Student
    print("--- [train_student_kd] Initializing student model...")
    student = UNetStudent(in_channels=4, out_channels=4).to(DEVICE)

    # 4. Losses
    print("--- [train_student_kd] Initializing losses...")
    pixel_loss_fn      = CharbonnierLoss().to(DEVICE)
    feature_loss_fn    = nn.L1Loss().to(DEVICE)
    frequency_loss_fn  = FFTAmpPhaseMultiScaleLoss(
        loss_weight=1.0,
        focus_low_freq=True,
        cutoff=0.25,
        lambda_amp=1.0,
        lambda_phase=0.5,
        scales=(1.0, 0.5)
    ).to(DEVICE)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)

    # 5. Optimizer + scheduler
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    # 6. AMP scaler
    scaler = GradScaler()

    # 7. Loss histories
    train_loss_history = []
    val_loss_history   = []
    best_val_loss      = float("inf")

    print("--- [train_student_kd] Starting Knowledge Distillation training...")

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

                # 1) Pixel loss (raw 4-ch)
                loss_pixel = pixel_loss_fn(student_out_4ch, gt_batch)

                # 2) Spatial feature KD (L1)
                student_features = F.interpolate(
                    student_features_raw,
                    size=teacher_feat_spatial.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                loss_feature_spatial = feature_loss_fn(student_features, teacher_feat_spatial)

                # 3) LPIPS in pseudo-RGB (first 3 channels)
                pred_rgb_slice = student_out_4ch[:, :3, :, :]
                gt_rgb_slice   = gt_batch[:, :3, :, :]
                loss_lpips = perceptual_loss_fn(
                    pred_rgb_slice * 2 - 1,
                    gt_rgb_slice   * 2 - 1
                ).mean()

            # 4) Frequency-based KD (amplitude + phase, multi-scale)
            loss_freq_out = frequency_loss_fn(
                student_out_4ch.detach(),
                teacher_out_4ch.detach()
            )
            loss_feature_freq = frequency_loss_fn(
                student_features.detach(),
                teacher_feat_spatial.detach()
            )

            # Combine spatial + freq feature KD
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

        avg_train_loss = train_loss / len(train_loader)
        print(f"--- [train_student_kd] Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation (Charbonnier only)
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

        avg_val_loss = val_loss / len(val_loader)
        print(f"--- [train_student_kd] Epoch {epoch+1} Val Charbonnier Loss: {avg_val_loss:.4f}")

        scheduler.step()

        # Record histories
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Save latest checkpoints
        torch.save(student.state_dict(), LOCAL_LATEST_STUDENT)
        torch.save(student.state_dict(), DRIVE_LATEST_STUDENT)
        print(f"Saved latest student checkpoint to {LOCAL_LATEST_STUDENT} and {DRIVE_LATEST_STUDENT}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), DRIVE_BEST_STUDENT)
            print(f"New best val loss. Saved best student checkpoint to {DRIVE_BEST_STUDENT}")

        # Save loss history each epoch
        np.savez(
            LOCAL_LOSS_HISTORY,
            train_loss=np.array(train_loss_history, dtype=np.float32),
            val_loss=np.array(val_loss_history, dtype=np.float32),
        )
        np.savez(
            DRIVE_LOSS_HISTORY,
            train_loss=np.array(train_loss_history, dtype=np.float32),
            val_loss=np.array(val_loss_history, dtype=np.float32),
        )

    # Final named checkpoint
    torch.save(student.state_dict(), FINAL_STUDENT_NAME)
    torch.save(student.state_dict(), os.path.join(DRIVE_CHECKPOINT_DIR, FINAL_STUDENT_NAME))
    print(f"--- [train_student_kd] Final student saved as {FINAL_STUDENT_NAME} locally and on Drive")

    # Final loss curves plot
    epochs = np.arange(1, len(train_loss_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss_history, label="Train")
    ax.plot(epochs, val_loss_history,   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (KD total: pixel + feat + freq + LPIPS)")
    ax.set_title("Student KD Training & Validation Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PNG, dpi=150)
    plt.close(fig)

    # Copy plot to Drive
    try:
        import shutil
        shutil.copy(LOSS_PLOT_PNG, DRIVE_LOSS_PLOT)
        print(f"Saved loss curves plot to {LOSS_PLOT_PNG} and {DRIVE_LOSS_PLOT}")
    except Exception as e:
        print(f"Could not copy loss plot to Drive: {e}")


if __name__ == "__main__":
    main()
