# File: train_teacher.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import lpips

# Speed vs reproducibility: favour speed on Colab
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

print("--- [train_teacher] Setting up system path...")
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)
print(f"--- [train_teacher] Added {mambair_path} to sys.path")

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from losses.frequency_loss import FFTAmplitudeLoss
from losses.pixel_loss import CharbonnierLoss

# ---------------- CONFIG ----------------
TRAIN_DIR = "/content/dataset/UDC-SIT/training"
VAL_DIR   = "/content/dataset/UDC-SIT/validation"

PATCH_SIZE = 256

BATCH_SIZE = 8          # was 9; 8 is a bit safer for memory
NUM_EPOCHS = 22         # full setting; if Colab is tight, you can drop to 16

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOCAL_CHECKPOINT_NAME = "teacher_4ch_latest.pth"
FINAL_CHECKPOINT_NAME = "teacher_4ch_22epochs_bs8.pth"  # adjust name as you like

# Path on Google Drive to keep checkpoints safe
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/Computational Imaging Project/checkpoints"
os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
DRIVE_LATEST_CKPT = os.path.join(DRIVE_CHECKPOINT_DIR, "teacher_latest.pth")
DRIVE_BEST_CKPT   = os.path.join(DRIVE_CHECKPOINT_DIR, "teacher_best.pth")

# Loss weights
W_PIXEL = 1.0
W_PERCEPTUAL = 0.1
W_FFT = 0.05
# ----------------------------------------

print("\n--- [train_teacher] Configuration ---")
print(f"Train Dir: {TRAIN_DIR}")
print(f"Val Dir:   {VAL_DIR}")
print(f"Patch Size: {PATCH_SIZE}, Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}, Learning Rate: {LEARNING_RATE}")
print(f"Device: {DEVICE}")
print(f"Local latest ckpt: {LOCAL_CHECKPOINT_NAME}")
print(f"Drive latest ckpt: {DRIVE_LATEST_CKPT}")
print("-----------------------------------\n")

def main():
    # 1. Data
    print("--- [train_teacher] Loading datasets...")
    train_dataset = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    val_dataset   = UDCDataset(VAL_DIR,   patch_size=PATCH_SIZE, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"--- [train_teacher] Device: {DEVICE}")
    print(f"--- [train_teacher] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Model
    print("--- [train_teacher] Initializing model...")
    model = FrequencyAwareTeacher(
        in_channels=4,
        out_channels=4,
        img_size=PATCH_SIZE
    ).to(DEVICE)

    # 3. Losses
    print("--- [train_teacher] Initializing losses...")
    pixel_loss_fn = CharbonnierLoss().to(DEVICE)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)
    fft_loss_fn = FFTAmplitudeLoss(
        loss_weight=1.0,
        focus_low_freq=True,
        cutoff=0.25
    ).to(DEVICE)

    # 4. Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    # 5. AMP scaler
    scaler = GradScaler('cuda')

    print("--- [train_teacher] Starting training...")

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for udc_batch, gt_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            udc_batch = udc_batch.to(DEVICE, non_blocking=True)
            gt_batch  = gt_batch.to(DEVICE,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                pred_batch_4ch, _, _ = model(udc_batch)

                # Charbonnier pixel loss
                loss_pixel = pixel_loss_fn(pred_batch_4ch, gt_batch)

                # LPIPS on pseudo-RGB (first 3 channels)
                pred_rgb = pred_batch_4ch[:, :3, :, :]
                gt_rgb   = gt_batch[:, :3, :, :]
                loss_perceptual = perceptual_loss_fn(
                    pred_rgb * 2 - 1,
                    gt_rgb   * 2 - 1
                ).mean()

            # Frequency-domain loss (float32)
            loss_fft = fft_loss_fn(pred_batch_4ch.float(), gt_batch.float())

            total_loss = (W_PIXEL * loss_pixel) + \
                         (W_PERCEPTUAL * loss_perceptual) + \
                         (W_FFT * loss_fft)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"--- [train_teacher] Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for udc_batch, gt_batch in val_loader:
                udc_batch = udc_batch.to(DEVICE)
                gt_batch  = gt_batch.to(DEVICE)

                with autocast():
                    pred_batch_4ch, _, _ = model(udc_batch)
                    loss_val = pixel_loss_fn(pred_batch_4ch, gt_batch)

                val_loss += loss_val.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"--- [train_teacher] Epoch {epoch+1} Val Charbonnier Loss: {avg_val_loss:.4f}")

        scheduler.step()

        # --- Save checkpoints (local + Drive) ---
        torch.save(model.state_dict(), LOCAL_CHECKPOINT_NAME)
        torch.save(model.state_dict(), DRIVE_LATEST_CKPT)
        print(f"Saved latest teacher checkpoint to {LOCAL_CHECKPOINT_NAME} and {DRIVE_LATEST_CKPT}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), DRIVE_BEST_CKPT)
            print(f"New best val loss. Saved best teacher checkpoint to {DRIVE_BEST_CKPT}")

    # Save final named checkpoint locally + to Drive
    torch.save(model.state_dict(), FINAL_CHECKPOINT_NAME)
    torch.save(model.state_dict(), os.path.join(DRIVE_CHECKPOINT_DIR, FINAL_CHECKPOINT_NAME))
    print(f"--- [train_teacher] Final model saved as {FINAL_CHECKPOINT_NAME} locally and on Drive")

if __name__ == "__main__":
    main()
