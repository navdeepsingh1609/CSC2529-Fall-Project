import sys
import os

# --- NEW CODE: Fix for 'basicsr' ModuleNotFoundError ---
# We must add the MambaIR submodule path to the system path
# so that its internal 'from basicsr...' imports can work.

# Get the absolute path to the directory containing this script
project_root = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the MambaIR package
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')

# Add this path to the beginning of the sys.path
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)
# --- END NEW CODE ---


# --- Original code continues below ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips  # Make sure you pip installed this

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from losses.frequency_loss import FFTAmplitudeLoss

# --- Config ---
TRAIN_DIR = "data/UDC-SIT_subset/train"
VAL_DIR = "data/UDC-SIT_subset/val"
PATCH_SIZE = 256
BATCH_SIZE = 4  # Adjust based on Colab Pro GPU (A100/V100)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights
W_PIXEL = 1.0
W_PERCEPTUAL = 0.1
W_FFT = 0.05
# --------------

def main():
    # 1. DataLoaders
    train_dataset = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    val_dataset = UDCDataset(VAL_DIR, patch_size=PATCH_SIZE, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Device: {DEVICE}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Model
    # Note: UDC-SIT paper says input is 4-channel, output is 3-channel
    model = FrequencyAwareTeacher(
        in_channels=4, 
        out_channels=3, 
        img_size=PATCH_SIZE
    ).to(DEVICE)
    
    # 3. Losses
    pixel_loss_fn = nn.L1Loss().to(DEVICE)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)
    fft_loss_fn = FFTAmplitudeLoss().to(DEVICE)
    
    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for udc_batch, gt_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            udc_batch = udc_batch.to(DEVICE)
            gt_batch = gt_batch.to(DEVICE) 

            # Check if GT is 4-channel and select RGB
            if gt_batch.shape[1] == 4:
                gt_batch_rgb = gt_batch[:, :3, :, :]
            else:
                gt_batch_rgb = gt_batch

            # Forward pass
            pred_batch = model(udc_batch)
            
            # Compute losses
            loss_pixel = pixel_loss_fn(pred_batch, gt_batch_rgb)
            # LPIPS expects input in range [-1, 1], our data is [0, 1]
            # We scale it: (x * 2) - 1
            loss_perceptual = perceptual_loss_fn(pred_batch * 2 - 1, gt_batch_rgb * 2 - 1).mean()
            loss_fft = fft_loss_fn(pred_batch, gt_batch_rgb)
            
            total_loss = (W_PIXEL * loss_pixel) + \
                         (W_PERCEPTUAL * loss_perceptual) + \
                         (W_FFT * loss_fft)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        print(f"Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}")
        
        # --- Validation Loop (Recommended) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for udc_batch, gt_batch in val_loader:
                udc_batch = udc_batch.to(DEVICE)
                gt_batch = gt_batch.to(DEVICE)
                if gt_batch.shape[1] == 4:
                    gt_batch_rgb = gt_batch[:, :3, :, :]
                else:
                    gt_batch_rgb = gt_batch
                
                pred_batch = model(udc_batch)
                
                loss_pixel = pixel_loss_fn(pred_batch, gt_batch_rgb)
                val_loss += loss_pixel.item() # Just use L1 for val for speed
        
        print(f"Epoch {epoch+1} Val L1 Loss: {val_loss / len(val_loader):.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"teacher_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()