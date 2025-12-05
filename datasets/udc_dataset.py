"""
UDC-SIT Dataset Loader.

Handles loading and preprocessing of the UDC-SIT dataset (Numpy format).
Supports:
- Loading Input/GT pairs for Training, Validation, and Testing.
- On-the-fly random cropping and augmentation (flip/rotation).
- Normalization (14-bit to [0,1]).
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class UDCDataset(Dataset):
    """
    PyTorch Dataset for UDC-SIT (Under-Display Camera) data.
    
    Expects data structure:
        root/
            split/ (training/validation/testing)
                input/ (containing .npy files)
                GT/    (containing .npy files)
    
    Args:
        root_dir (str): Root directory of the dataset.
        split (str): 'training', 'validation', or 'testing'.
        patch_size (int): Size of random crops for training (default: 256).
        is_train (bool): Enable data augmentation (default: True).
    """
    def __init__(self, root_dir, split='training', patch_size=256, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train
        self.root_dir = root_dir
        self.split = split
        
        # Expect files under root_dir/split/input/*.npy
        data_path = os.path.join(self.root_dir, self.split, "input")
        self.input_files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
        
        if not self.input_files:
            raise FileNotFoundError(f"No .npy files found in {data_path}")
        

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load UDC-SIT .npy: shape is (H, W, 4), values in [0, 1023]
        input_path = self.input_files[idx]
        udc_img = np.load(input_path)  # (H, W, C)
        
        gt_path = input_path.replace("/input/", "/GT/")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing ground truth file: {gt_path}")
        gt_img = np.load(gt_path)      # (H, W, C)

        if udc_img.shape != gt_img.shape:
            raise ValueError(f"Input and GT shapes differ: {udc_img.shape} vs {gt_img.shape}")

        H, W, C = udc_img.shape
        ps = self.patch_size

        if ps > H or ps > W:
            raise ValueError(f"Patch size {ps} is larger than image size {(H, W)}")

        # Random crop for training, center crop for validation
        if self.is_train:
            # rows (y), cols (x)
            rr = random.randint(0, H - ps)  # top
            rc = random.randint(0, W - ps)  # left
        else:
            rr = (H - ps) // 2
            rc = (W - ps) // 2

        # Crop (H, W, C) -> (ps, ps, C)
        udc_patch = udc_img[rr:rr + ps, rc:rc + ps, :]
        gt_patch  = gt_img[rr:rr + ps, rc:rc + ps, :]

        # Simple horizontal flip augmentation
        if self.is_train and random.random() > 0.5:
            udc_patch = np.fliplr(udc_patch)
            gt_patch  = np.fliplr(gt_patch)

        # Convert to tensor: HWC -> CHW
        udc_tensor = torch.from_numpy(udc_patch.copy()).permute(2, 0, 1).float()  # (4, ps, ps)
        gt_tensor  = torch.from_numpy(gt_patch.copy()).permute(2, 0, 1).float()   # (4, ps, ps)

        # Normalize 10-bit [0, 1023] -> [0, 1]
        udc_tensor /= 1023.0
        gt_tensor  /= 1023.0
        
        return udc_tensor, gt_tensor
