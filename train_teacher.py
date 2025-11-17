import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import lpips  

# --- Path fix (unchanged) ---
print("--- [train_teacher] Setting up system path...")
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)
print(f"--- [train_teacher] Added {mambair_path} to sys.path")
# --- End path fix ---

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from losses.frequency_loss import FFTAmplitudeLoss

# --- [FIX] CONFIG FOR 4-HOUR BUDGET (BS=10) ---
TRAIN_DIR = "/content/dataset/UDC-SIT/training" 
VAL_DIR = "/content/dataset/UDC-SIT/validation" 
PATCH_SIZE = 256
BATCH_SIZE = 10 # <-- UPDATED from 8
NUM_EPOCHS = 34 # <-- UPDATED from 50/24
# --- [END FIX] ---

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_NAME = "teacher_4ch_34_epochs.pth" # New name

# Loss weights
W_PIXEL = 1.0
W_PERCEPTUAL = 0.1
W_FFT = 0.05

print("\n--- [train_teacher] Configuration ---")
print(f"Train Dir: {TRAIN_DIR}")
print(f"Val Dir: {VAL_DIR}")
print(f"Patch Size: {PATCH_SIZE}, Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}, Learning Rate: {LEARNING_RATE}")
print(f"Device: {DEVICE}")
print(f"Checkpoint Name: {CHECKPOINT_NAME}")
print("-----------------------------------\n")
# --------------

def main():
    # 1. DataLoaders (unchanged, still high-speed)
    print("--- [train_teacher] Loading datasets...")
    train_dataset = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=8, pin_memory=True
    )
    val_dataset = UDCDataset(VAL_DIR, patch_size=PATCH_SIZE, is_train=False)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=8, pin_memory=True
    )
    print(f"--- [train_teacher] Device: {DEVICE}")
    print(f"--- [train_teacher] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Model (unchanged)
    print("--- [train_teacher] Initializing model...")
    model = FrequencyAwareTeacher(in_channels=4, out_channels=4, img_size=PATCH_SIZE).to(DEVICE)
    
    # 3. Losses (unchanged)
    print("--- [train_teacher] Initializing losses...")
    pixel_loss_fn = nn.L1Loss().to(DEVICE)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)
    fft_loss_fn = FFTAmplitudeLoss().to(DEVICE)
    
    # 4. Optimizer (unchanged)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. AMP Scaler (unchanged)
    scaler = torch.amp.GradScaler('cuda')

    print("--- [train_teacher] Starting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        # This loop will now have 187 steps
        for udc_batch, gt_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            udc_batch = udc_batch.to(DEVICE, non_blocking=True)
            gt_batch = gt_batch.to(DEVICE, non_blocking=True) 

            optimizer.zero_grad(set_to_none=True)

            # --- [FIX for 'nan' loss] ---
            with autocast():
                pred_batch_4ch, _, _ = model(udc_batch) 
                loss_pixel = pixel_loss_fn(pred_batch_4ch, gt_batch)
                pred_rgb_slice = pred_batch_4ch[:, :3, :, :]
                gt_rgb_slice = gt_batch[:, :3, :, :]
                loss_perceptual = perceptual_loss_fn(pred_rgb_slice * 2 - 1, gt_rgb_slice * 2 - 1).mean()
            
            loss_fft = fft_loss_fn(pred_batch_4ch.float(), gt_batch.float())
            # --- [END FIX] ---
                
            total_loss = (W_PIXEL * loss_pixel) + \
                         (W_PERCEPTUAL * loss_perceptual) + \
                         (W_FFT * loss_fft)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
        
        print(f"--- [train_teacher] Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}")
        
        # --- Validation Loop (unchanged) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for udc_batch, gt_batch in val_loader:
                udc_batch = udc_batch.to(DEVICE)
                gt_batch = gt_batch.to(DEVICE)
                
                with autocast():
                    pred_batch_4ch, _, _ = model(udc_batch)
                
                loss_pixel = pixel_loss_fn(pred_batch_4ch, gt_batch)
                val_loss += loss_pixel.item()
        
        print(f"--- [train_teacher] Epoch {epoch+1} Val L1 Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), CHECKPOINT_NAME)
    print(f"--- [train_teacher] Final model saved as {CHECKPOINT_NAME}")

if __name__ == "__main__":
    main()