# File: train_teacher_quick.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import lpips

# Speed over reproducibility for quick testing
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

print("--- [train_teacher_quick] Setting up system path...")
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)
print(f"--- [train_teacher_quick] Added {mambair_path} to sys.path")

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from losses.frequency_loss import FFTAmplitudeLoss
from losses.pixel_loss import CharbonnierLoss

# ---------------- CONFIG (quick subset) ----------------
TRAIN_DIR = "data/UDC-SIT_subset/train"
VAL_DIR   = "data/UDC-SIT_subset/val"

PATCH_SIZE = 256
BATCH_SIZE = 4       # small for quick test
NUM_EPOCHS = 2       # just to see if everything runs

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_NAME = "teacher_quick_1epoch.pth"
# ------------------------------------------------------

print("\n--- [train_teacher_quick] Configuration ---")
print(f"Train Dir: {TRAIN_DIR}")
print(f"Val Dir:   {VAL_DIR}")
print(f"Patch Size: {PATCH_SIZE}, Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}, Learning Rate: {LEARNING_RATE}")
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_NAME}")
print("-----------------------------------\n")

def main():
    # 1. Data
    print("--- [train_teacher_quick] Loading datasets...")
    train_dataset = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    val_dataset   = UDCDataset(VAL_DIR,   patch_size=PATCH_SIZE, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # 2. Model
    print("--- [train_teacher_quick] Initializing model...")
    model = FrequencyAwareTeacher(
        in_channels=4,
        out_channels=4,
        img_size=PATCH_SIZE
    ).to(DEVICE)

    # 3. Losses
    pixel_loss_fn = CharbonnierLoss().to(DEVICE)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)
    fft_loss_fn = FFTAmplitudeLoss(
        loss_weight=1.0,
        focus_low_freq=True,
        cutoff=0.25
    ).to(DEVICE)

    # 4. Optimizer (no scheduler needed for 1 epoch)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. AMP scaler
    scaler = GradScaler('cuda')

    print("--- [train_teacher_quick] Starting quick training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for udc_batch, gt_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            udc_batch = udc_batch.to(DEVICE, non_blocking=True)
            gt_batch  = gt_batch.to(DEVICE,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                pred_batch_4ch, _, _ = model(udc_batch)

                loss_pixel = pixel_loss_fn(pred_batch_4ch, gt_batch)

                pred_rgb = pred_batch_4ch[:, :3, :, :]
                gt_rgb   = gt_batch[:, :3, :, :]
                loss_perceptual = perceptual_loss_fn(
                    pred_rgb * 2 - 1,
                    gt_rgb   * 2 - 1
                ).mean()

            loss_fft = fft_loss_fn(pred_batch_4ch.float(), gt_batch.float())

            total_loss = loss_pixel + 0.1 * loss_perceptual + 0.05 * loss_fft

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()

        print(f"--- [train_teacher_quick] Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}")

        # Quick val just to confirm no NaNs / runtime issues
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

        print(f"--- [train_teacher_quick] Epoch {epoch+1} Val Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), CHECKPOINT_NAME)
    print(f"--- [train_teacher_quick] Saved quick teacher checkpoint: {CHECKPOINT_NAME}")

if __name__ == "__main__":
    main()
