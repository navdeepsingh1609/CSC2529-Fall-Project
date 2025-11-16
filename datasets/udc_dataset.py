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
        # Load images
        input_path = self.input_files[idx]
        udc_img = np.load(input_path)
        
        gt_path = input_path.replace("/input/", "/GT/")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing ground truth file: {gt_path}")
        gt_img = np.load(gt_path)

        # Get a random patch
        H, W, _ = udc_img.shape
        if self.is_train:
            ps = self.patch_size
            ps_h = min(ps, H)
            ps_w = min(ps, W)
            rr = random.randint(0, H - ps_h)
            rc = random.randint(0, W - ps_w)
            udc_patch = udc_img[rr : rr + ps_h, rc : rc + ps_w, :]
            gt_patch = gt_img[rr : rr + ps_h, rc : rc + ps_w, :]
        else:
            ps = self.patch_size
            ps_h = min(ps, H)
            ps_w = min(ps, W)
            rr = (H - ps_h) // 2
            rc = (W - ps_w) // 2
            udc_patch = udc_img[rr : rr + ps_h, rc : rc + ps_w, :]
            gt_patch = gt_img[rr : rr + ps_h, rc : rc + ps_w, :]

        if self.is_train and random.random() > 0.5:
            udc_patch = np.fliplr(udc_patch)
            gt_patch = np.fliplr(gt_patch)

        # Convert to tensor
        udc_tensor = torch.from_numpy(udc_patch.copy()).permute(2, 0, 1).float()
        gt_tensor = torch.from_numpy(gt_patch.copy()).permute(2, 0, 1).float()
        
        # Normalize the 10-bit [0, 1023] data to the [0, 1] range
        udc_tensor /= 1023.0
        gt_tensor /= 1023.0
        
        return udc_tensor, gt_tensor