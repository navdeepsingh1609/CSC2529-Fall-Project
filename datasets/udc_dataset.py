# File: datasets/udc_dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class UDCDataset(Dataset):
    def __init__(self, data_dir, patch_size=256, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train
        self.udc_files = sorted(glob.glob(os.path.join(data_dir, "*_udc.npy")))
        
        if not self.udc_files:
            raise FileNotFoundError(f"No '*_udc.npy' files found in {data_dir}")

    def __len__(self):
        return len(self.udc_files)

    def __getitem__(self, idx):
        # Load UDC image
        udc_path = self.udc_files[idx]
        udc_img = np.load(udc_path) # Shape (H, W, C)
        
        # Load corresponding GT image
        gt_path = udc_path.replace("_udc.npy", "_gt.npy")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing ground truth file: {gt_path}")
        gt_img = np.load(gt_path) # Shape (H, W, C)

        # --- Get a random patch ---
        H, W, _ = udc_img.shape
        if self.is_train:
            # Get random top-left corner
            ps = self.patch_size
            rr = random.randint(0, H - ps)
            rc = random.randint(0, W - ps)
            
            udc_patch = udc_img[rr : rr + ps, rc : rc + ps, :]
            gt_patch = gt_img[rr : rr + ps, rc : rc + ps, :]
        else:
            # For validation, just take a center crop
            ps = self.patch_size
            rr = (H - ps) // 2
            rc = (W - ps) // 2
            udc_patch = udc_img[rr : rr + ps, rc : rc + ps, :]
            gt_patch = gt_img[rr : rr + ps, rc : rc + ps, :]

        # --- Augmentation (simple flip) ---
        if self.is_train and random.random() > 0.5:
            udc_patch = np.fliplr(udc_patch)
            gt_patch = np.fliplr(gt_patch)

        # Convert to PyTorch tensor (H, W, C) -> (C, H, W)
        # The UDC-SIT paper mentions 4 channels. We'll assume they are the input.
        udc_tensor = torch.from_numpy(udc_patch.copy()).permute(2, 0, 1).float()
        gt_tensor = torch.from_numpy(gt_patch.copy()).permute(2, 0, 1).float()
        
        # Normalize. Assuming LDR images [0, 255]
        udc_tensor /= 255.0
        gt_tensor /= 255.0
        
        return udc_tensor, gt_tensor