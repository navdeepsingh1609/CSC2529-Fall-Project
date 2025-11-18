# File: train_student_kd.py
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

# Speed vs reproducibility: favour speed
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
from losses.frequency_loss import FFTAmplitudeLoss
from losses.pixel_loss import CharbonnierLoss

# ---------------- CONFIG ----------------
TRAIN_DIR = "/content/dataset/UDC-SIT/training"
VAL_DIR   = "/content/dataset/UDC-SIT/validation"

PATCH_SIZE = 256

BATCH_SIZE = 64          # adjust down to 24 if you hit OOM
LEARNING_RATE = 2e-4
NUM_EPOCHS = 20          # full setting; if needed, drop to 14

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Teacher checkpoint (already trained)
TEACHER_WEIGHTS = "teacher_4ch_22epochs_bs8.pth"  # or your previous teacher file

# Student checkpoints
LOCAL_LATEST_STUDENT = "student_4ch_latest.pth"
FINAL_STUDENT_NAME   = "student_distilled_4ch_full_data.pth"

DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/Computational Imaging Project/checkpoints"
os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
DRIVE_LATEST_STUDENT = os.path.join(DRIVE_CHECKPOINT_DIR, "student_latest.pth")
DRIVE_BEST_STUDENT   = os.path.join(DRIVE_CHECKPOINT_DIR, "student_best.pth")

# Loss weights
W_PIXEL   = 1.0
W_FEATURE = 0.5    # combined spatial + freq feature KD
W_FREQ    = 0.2    # output-level freq KD
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
    pixel_loss_fn     = CharbonnierLoss().to(DEVICE)
    feature_loss_fn   = nn.L1Loss().to(DEVICE)   # spatial KD
    frequency_loss_fn = FFTAmplitudeLoss(
        loss_weight=1.0,
        focus_low_freq=True,
        cutoff=0.25
    ).to(DEVICE)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)

    # 5. Optimizer + scheduler
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    # 6. AMP scaler
    scaler = GradScaler('cuda')

    print("--- [train_student_kd] Starting Knowledge Distillation training...")

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        student.train()
        train_loss = 0.0

        for udc_batch, gt_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            udc_batch = udc_batch.to(DEVICE, non_blocking=True)
            gt_batch  = gt_batch.to(DEVICE,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ---- Teacher forward (no grad) ----
            with torch.no_grad():
                with autocast():
                    teacher_out_4ch, teacher_feat_spatial, _ = teacher(udc_batch)

            # ---- Student forward + main losses ----
            with autocast():
                student_out_4ch, student_features_raw = student(udc_batch)

                # 1) Pixel loss (Charbonnier)
                loss_pixel = pixel_loss_fn(student_out_4ch, gt_batch)

                # 2) Spatial feature KD
                student_features = F.interpolate(
                    student_features_raw,
                    size=teacher_feat_spatial.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                loss_feature_spatial = feature_loss_fn(student_features, teacher_feat_spatial)

                # 3) LPIPS (pseudo-RGB)
                pred_rgb_slice = student_out_4ch[:, :3, :, :]
                gt_rgb_slice   = gt_batch[:, :3, :, :]
                loss_lpips = perceptual_loss_fn(
                    pred_rgb_slice * 2 - 1,
                    gt_rgb_slice   * 2 - 1
                ).mean()

            # ---- Frequency-based KD (float32) ----
            # (a) Output-level frequency KD: student vs teacher output
            loss_freq_out = frequency_loss_fn(
                student_out_4ch.float(),
                teacher_out_4ch.float()
            )

            # (b) Feature-level frequency KD: student vs teacher spatial features
            loss_feature_freq = frequency_loss_fn(
                student_features.float(),
                teacher_feat_spatial.float()
            )

            # Combine spatial + frequency feature KD (novel KD idea)
            loss_feature = 0.5 * loss_feature_spatial + 0.5 * loss_feature_freq
            loss_freq    = loss_freq_out

            # ---- Total loss ----
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

        # ---- Validation ----
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

        # ---- Save checkpoints (local + Drive) ----
        torch.save(student.state_dict(), LOCAL_LATEST_STUDENT)
        torch.save(student.state_dict(), DRIVE_LATEST_STUDENT)
        print(f"Saved latest student checkpoint to {LOCAL_LATEST_STUDENT} and {DRIVE_LATEST_STUDENT}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), DRIVE_BEST_STUDENT)
            print(f"New best val loss. Saved best student checkpoint to {DRIVE_BEST_STUDENT}")

    # Save final named checkpoint locally + to Drive
    torch.save(student.state_dict(), FINAL_STUDENT_NAME)
    torch.save(student.state_dict(), os.path.join(DRIVE_CHECKPOINT_DIR, FINAL_STUDENT_NAME))
    print(f"--- [train_student_kd] Final student saved as {FINAL_STUDENT_NAME} locally and on Drive")

if __name__ == "__main__":
    main()
