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
        
        self.input_files = sorted(glob.glob(os.path.join(data_dir, "input", "*.npy")))
        
        if not self.input_files:
            raise FileNotFoundError(f"No .npy files found in {os.path.join(data_dir, 'input')}")
        
        print(f"--- [UDCDataset] Loaded {len(self.input_files)} files from {data_dir}")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # --- [THE FIX] ---
        # Load images. Shape is (W, H, C) = (1792, 1280, 4)
        input_path = self.input_files[idx]
        udc_img = np.load(input_path)
        
        gt_path = input_path.replace("/input/", "/GT/")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing ground truth file: {gt_path}")
        gt_img = np.load(gt_path)

        # 1. Correctly get dimensions
        W, H, _ = udc_img.shape # W = 1792, H = 1280
        ps = self.patch_size

        # 2. Get random top-left corner
        if self.is_train:
            rc = random.randint(0, W - ps) # Random crop for W (dim 0)
            rr = random.randint(0, H - ps) # Random crop for H (dim 1)
        else:
            # Center crop
            rc = (W - ps) // 2
            rr = (H - ps) // 2

        # 3. Crop the patch (W_crop, H_crop, C)
        udc_patch = udc_img[rc : rc + ps, rr : rr + ps, :]
        gt_patch = gt_img[rc : rc + ps, rr : rr + ps, :]
        # --- [END FIX] ---
        
        if self.is_train and random.random() > 0.5:
            udc_patch = np.fliplr(udc_patch)
            gt_patch = np.fliplr(gt_patch)

        # --- [THE FIX] ---
        # 4. Correctly permute (W, H, C) to (C, H, W)
        # We need to map (0, 1, 2) -> (2, 1, 0)
        # (patch_W, patch_H, C) -> (C, patch_H, patch_W)
        udc_tensor = torch.from_numpy(udc_patch.copy()).permute(2, 1, 0).float()
        gt_tensor = torch.from_numpy(gt_patch.copy()).permute(2, 1, 0).float()
        # --- [END FIX] ---

        # Normalize
        udc_tensor /= 1023.0
        gt_tensor /= 1023.0
        
        return udc_tensor, gt_tensor