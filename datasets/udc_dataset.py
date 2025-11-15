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
        
        # The data_dir is now 'data/UDC-SIT_subset/train' or '.../val'
        self.input_files = sorted(glob.glob(os.path.join(data_dir, "input", "*.npy")))
        
        if not self.input_files:
            raise FileNotFoundError(f"No .npy files found in {os.path.join(data_dir, 'input')}")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load Input image
        input_path = self.input_files[idx]
        udc_img = np.load(input_path) # Shape (H, W, C)
        
        # Find corresponding GT image
        # It's in a parallel 'GT' folder
        gt_path = input_path.replace("/input/", "/GT/")
        
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing ground truth file: {gt_path}")
        gt_img = np.load(gt_path) # Shape (H, W, C)

        # --- Get a random patch ---
        H, W, _ = udc_img.shape
        if self.is_train:
            ps = self.patch_size
            rr = random.randint(0, H - ps)
            rc = random.randint(0, W - ps)
            udc_patch = udc_img[rr : rr + ps, rc : rc + ps, :]
            gt_patch = gt_img[rr : rr + ps, rc : rc + ps, :]
        else:
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
        udc_tensor = torch.from_numpy(udc_patch.copy()).permute(2, 0, 1).float()
        gt_tensor = torch.from_numpy(gt_patch.copy()).permute(2, 0, 1).float()
        
        # Normalize. Assuming LDR images [0, 255]
        udc_tensor /= 255.0
        gt_tensor /= 255.0
        
        return udc_tensor, gt_tensor