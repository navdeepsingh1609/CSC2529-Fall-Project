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

# --- Path fix (unchanged) ---
print("--- [train_student_kd] Setting up system path...")
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)
print(f"--- [train_student_kd] Added {mambair_path} to sys.path")
# --- End path fix ---

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent
from losses.frequency_loss import FFTAmplitudeLoss

# --- [FIX] CONFIGURATION ---
# Pointing to the FULL dataset, just like the teacher
TRAIN_DIR = "/content/dataset/UDC-SIT/training" 
VAL_DIR = "/content/dataset/UDC-SIT/validation" 
# --- [END FIX] ---

PATCH_SIZE = 256
BATCH_SIZE = 16  # Student is lightweight, we can use a larger batch than the teacher
LEARNING_RATE = 2e-4
NUM_EPOCHS = 40 # We can use more epochs, but let's match the teacher for a fair comparison
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- [FIX] Correct teacher checkpoint name ---
TEACHER_WEIGHTS = "teacher_4ch_26_epochs_bs9.pth" 
# --- [END FIX] ---
STUDENT_SAVE_PATH = "student_distilled_4ch_full_data.pth" # New name for student

# Loss weights
W_PIXEL = 1.0     
W_FEATURE = 0.5   
W_FREQ = 0.2
W_LPIPS = 0.1 

print("\n--- [train_student_kd] Configuration ---")
print(f"Train Dir: {TRAIN_DIR}")
print(f"Val Dir: {VAL_DIR}")
print(f"Patch Size: {PATCH_SIZE}, Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}, Learning Rate: {LEARNING_RATE}")
print(f"Device: {DEVICE}")
print(f"Teacher Weights: {TEACHER_WEIGHTS}")
print(f"Student Save Path: {STUDENT_SAVE_PATH}")
print("-----------------------------------\n")
# --------------

def main():
    # 1. DataLoaders
    print("--- [train_student_kd] Loading datasets...")
    train_dataset = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True
    )
    val_dataset = UDCDataset(VAL_DIR, patch_size=PATCH_SIZE, is_train=False)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    print(f"--- [train_student_kd] Device: {DEVICE}")

    # 2. Load and FREEZE Teacher Model
    print(f"--- [train_student_kd] Loading teacher model from {TEACHER_WEIGHTS}...")
    teacher = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(DEVICE) # 4-ch out
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS))
    teacher.eval() 
    for param in teacher.parameters():
        param.requires_grad = False 
    print("--- [train_student_kd] Teacher model loaded and frozen.")

    # 3. Initialize Student Model
    print("--- [train_student_kd] Initializing student model...")
    student = UNetStudent(in_channels=4, out_channels=4).to(DEVICE) # 4-ch out

    # 4. Losses
    print("--- [train_student_kd] Initializing losses...")
    pixel_loss_fn = nn.L1Loss().to(DEVICE)
    feature_loss_fn = nn.L1Loss().to(DEVICE)
    frequency_loss_fn = FFTAmplitudeLoss().to(DEVICE) 
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)

    # 5. Optimizer
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)
    
    # 6. AMP Scaler
    scaler = torch.amp.GradScaler('cuda')

    print("--- [train_student_kd] Starting Knowledge Distillation training...")

    for epoch in range(NUM_EPOCHS):
        student.train()
        train_loss = 0.0
        
        for udc_batch, gt_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            udc_batch = udc_batch.to(DEVICE, non_blocking=True)
            gt_batch = gt_batch.to(DEVICE, non_blocking=True) # Full 4-channel GT

            optimizer.zero_grad(set_to_none=True)

            # --- Get Teacher Outputs (with no_grad) ---
            with torch.no_grad():
                with autocast(): # Teacher can run in AMP
                    teacher_out_4ch, teacher_feat_spatial, _ = teacher(udc_batch)
            
            # --- Get Student Outputs (in AMP) ---
            with autocast():
                student_out_4ch, student_features_raw = student(udc_batch)
            
                # --- Compute Losses ---
                
                # 1. Pixel Loss (Student vs. Ground Truth) - 4 channel
                loss_pixel = pixel_loss_fn(student_out_4ch, gt_batch)
                
                # 2. Feature Distillation Loss - 4 channel
                student_features = F.interpolate(
                    student_features_raw, 
                    size=teacher_feat_spatial.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                loss_feature = feature_loss_fn(student_features, teacher_feat_spatial)
                
                # 4. LPIPS Loss (Student vs. Ground Truth, 3-ch slices)
                pred_rgb_slice = student_out_4ch[:, :3, :, :]
                gt_rgb_slice = gt_batch[:, :3, :, :]
                loss_lpips = perceptual_loss_fn(pred_rgb_slice * 2 - 1, gt_rgb_slice * 2 - 1).mean()
            
            # 3. Frequency Distillation Loss (Full Float32)
            loss_freq = frequency_loss_fn(student_out_4ch.float(), teacher_out_4ch.float())
            
            # Combine all losses
            total_loss = (W_PIXEL * loss_pixel) + \
                         (W_FEATURE * loss_feature) + \
                         (W_FREQ * loss_freq) + \
                         (W_LPIPS * loss_lpips)
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
        
        print(f"--- [train_student_kd] Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}")
        
        # --- Validation Loop (4-channel L1) ---
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for udc_batch, gt_batch in val_loader:
                udc_batch = udc_batch.to(DEVICE)
                gt_batch = gt_batch.to(DEVICE)
                
                with autocast():
                    student_out_4ch, _ = student(udc_batch)
                    
                loss_pixel = pixel_loss_fn(student_out_4ch, gt_batch)
                val_loss += loss_pixel.item()
        
        print(f"--- [train_student_kd] Epoch {epoch+1} Val L1 Loss: {val_loss / len(val_loader):.4f}")

    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f"--- [train_student_kd] Student model saved as {STUDENT_SAVE_PATH}")

if __name__ == "__main__":
    main()