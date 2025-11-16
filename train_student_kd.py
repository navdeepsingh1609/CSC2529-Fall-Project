import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips

# --- Path fix (unchanged) ---
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)
# --- End path fix ---

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent
from losses.frequency_loss import FFTAmplitudeLoss

# --- Config (unchanged) ---
TRAIN_DIR = "data/UDC-SIT_subset/train"
VAL_DIR = "data/UDC-SIT_subset/val"
PATCH_SIZE = 256
BATCH_SIZE = 8  
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEACHER_WEIGHTS = "teacher_10bit_normalized.pth" # Make sure to re-train the teacher
STUDENT_SAVE_PATH = "student_distilled_4ch.pth"

# Loss weights (unchanged)
W_PIXEL = 1.0     
W_FEATURE = 0.5   
W_FREQ = 0.2      
# --------------

def main():
    # 1. DataLoaders (unchanged)
    train_dataset = UDCDataset(TRAIN_DIR, patch_size=PATCH_SIZE, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataset = UDCDataset(VAL_DIR, patch_size=PATCH_SIZE, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Device: {DEVICE}")

    # 2. Load and FREEZE Teacher Model (unchanged)
    print(f"Loading teacher model from {TEACHER_WEIGHTS}...")
    teacher = FrequencyAwareTeacher(in_channels=4, out_channels=3).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS))
    teacher.eval() 
    for param in teacher.parameters():
        param.requires_grad = False 
    print("Teacher model loaded and frozen.")

    # 3. Initialize Student Model (unchanged)
    student = UNetStudent(in_channels=4, out_channels=3).to(DEVICE)

    # 4. Losses (unchanged)
    pixel_loss_fn = nn.L1Loss().to(DEVICE)
    feature_loss_fn = nn.L1Loss().to(DEVICE)
    frequency_loss_fn = FFTAmplitudeLoss().to(DEVICE) 

    # 5. Optimizer (unchanged)
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)
    
    print("Starting Knowledge Distillation training...")

    for epoch in range(NUM_EPOCHS):
        student.train()
        train_loss = 0.0
        
        for udc_batch, gt_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            udc_batch = udc_batch.to(DEVICE)
            gt_batch_rgb = gt_batch[:, :3, :, :].to(DEVICE)

            # --- Get Teacher Outputs (with no_grad) ---
            with torch.no_grad():
                # teacher_out: (B, 3, 256, 256)
                # teacher_features: (B, 4, 256, 256) <- This is now the fused 4ch map
                teacher_out, teacher_features = teacher(udc_batch)
            
            # --- Get Student Outputs ---
            # student_out: (B, 3, 256, 256)
            # student_features_raw: (B, 4, 16, 16) <- bottleneck
            student_out, student_features_raw = student(udc_batch)
            
            # --- Compute Losses ---
            
            # 1. Pixel Loss (Student vs. Ground Truth)
            loss_pixel = pixel_loss_fn(student_out, gt_batch_rgb)
            
            # 2. Feature Distillation Loss (Student vs. Teacher)
            # --- THIS IS THE ONLY CHANGE ---
            # We now resize the student's 4-ch bottleneck to match the
            # teacher's final 4-ch fused map. This is a very strong signal.
            student_features = F.interpolate(
                student_features_raw, 
                size=teacher_features.shape[2:], # Match (256, 256)
                mode='bilinear', 
                align_corners=False
            )
            loss_feature = feature_loss_fn(student_features, teacher_features)
            # --- END CHANGE ---

            # 3. Frequency Distillation Loss (Student vs. Teacher)
            loss_freq = frequency_loss_fn(student_out, teacher_out)
            
            # Combine all losses
            total_loss = (W_PIXEL * loss_pixel) + \
                         (W_FEATURE * loss_feature) + \
                         (W_FREQ * loss_freq)
            
            # Backward pass (only updates student)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        print(f"Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}")
        
        # --- Validation Loop (unchanged) ---
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for udc_batch, gt_batch in val_loader:
                udc_batch = udc_batch.to(DEVICE)
                gt_batch_rgb = gt_batch[:, :3, :, :].to(DEVICE)
                
                student_out, _ = student(udc_batch)
                loss_pixel = pixel_loss_fn(student_out, gt_batch_rgb)
                val_loss += loss_pixel.item()
        
        print(f"Epoch {epoch+1} Val L1 Loss: {val_loss / len(val_loader):.4f}")

    # Save final student model
    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f"Student model saved as {STUDENT_SAVE_PATH}")


if __name__ == "__main__":
    main()