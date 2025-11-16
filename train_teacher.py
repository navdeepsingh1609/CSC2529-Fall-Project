import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

# --- Config ---
TRAIN_DIR = "data/UDC-SIT_subset/train"
VAL_DIR = "data/UDC-SIT_subset/val"
PATCH_SIZE = 256
BATCH_SIZE = 2  
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_NAME = "teacher_4ch_restoration.pth" 

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
    # 1. DataLoaders
    print("--- [train_teacher] Loading datasets...")
    train_dataset = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataset = UDCDataset(VAL_DIR, patch_size=PATCH_SIZE, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"--- [train_teacher] Device: {DEVICE}")
    print(f"--- [train_teacher] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Model (now 4-ch in, 4-ch out)
    print("--- [train_teacher] Initializing model...")
    model = FrequencyAwareTeacher(
        in_channels=4, 
        out_channels=4, # 4-channel output
        img_size=PATCH_SIZE
    ).to(DEVICE)
    
    # 3. Losses
    print("--- [train_teacher] Initializing losses...")
    pixel_loss_fn = nn.L1Loss().to(DEVICE)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)
    fft_loss_fn = FFTAmplitudeLoss().to(DEVICE)
    
    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("--- [train_teacher] Starting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for udc_batch, gt_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            udc_batch = udc_batch.to(DEVICE)
            gt_batch = gt_batch.to(DEVICE) # gt_batch is (B, 4, H, W)

            # Unpack the 3-item tuple: (main_4ch_output, spatial_feat, freq_feat)
            pred_batch_4ch, _, _ = model(udc_batch) 
            
            # 1. Pixel Loss (4-channel)
            loss_pixel = pixel_loss_fn(pred_batch_4ch, gt_batch)
            
            # 2. Perceptual Loss (3-channel)
            pred_rgb_slice = pred_batch_4ch[:, :3, :, :]
            gt_rgb_slice = gt_batch[:, :3, :, :]
            loss_perceptual = perceptual_loss_fn(pred_rgb_slice * 2 - 1, gt_rgb_slice * 2 - 1).mean()
            
            # 3. FFT Loss (4-channel)
            loss_fft = fft_loss_fn(pred_batch_4ch, gt_batch)
            
            total_loss = (W_PIXEL * loss_pixel) + \
                         (W_PERCEPTUAL * loss_perceptual) + \
                         (W_FFT * loss_fft)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        print(f"--- [train_teacher] Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}")
        
        # --- Validation Loop (4-channel L1) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for udc_batch, gt_batch in val_loader:
                udc_batch = udc_batch.to(DEVICE)
                gt_batch = gt_batch.to(DEVICE)
                
                pred_batch_4ch, _, _ = model(udc_batch)
                
                loss_pixel = pixel_loss_fn(pred_batch_4ch, gt_batch)
                val_loss += loss_pixel.item()
        
        print(f"--- [train_teacher] Epoch {epoch+1} Val L1 Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), CHECKPOINT_NAME)
    print(f"--- [train_teacher] Final model saved as {CHECKPOINT_NAME}")


if __name__ == "__main__":
    main()