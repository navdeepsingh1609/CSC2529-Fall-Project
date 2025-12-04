# utils/bayer_utils.py
import torch
import numpy as np

def bayer4_to_rgb_torch(x: torch.Tensor, r_gain: float = 1.9, b_gain: float = 1.9) -> torch.Tensor:
    """
    Differentiable Bayer-4ch -> RGB for LPIPS / perceptual loss.

    Args:
        x: (B,4,H,W) tensor in [0,1], channel order: (GR, R, B, GB).
        r_gain, b_gain: simple white-balance gains.

    Returns:
        rgb: (B,3,H,W) tensor in [0,1].
    """
    assert x.dim() == 4 and x.size(1) == 4, f"Expected (B,4,H,W), got {x.shape}"

    GR = x[:, 0, :, :]
    R  = x[:, 1, :, :]
    B  = x[:, 2, :, :]
    GB = x[:, 3, :, :]

    G  = 0.5 * (GR + GB)
    Rn = torch.clamp(R * r_gain, 0.0, 1.0)
    Bn = torch.clamp(B * b_gain, 0.0, 1.0)

    rgb = torch.stack([Rn, G, Bn], dim=1)  # (B,3,H,W)
    return rgb


def bayer4_to_rgb_numpy(arr, r_gain: float = 1.9, b_gain: float = 1.9) -> np.ndarray:
    """
    NumPy Bayer-4ch -> RGB for testing / visualization.

    Args:
        arr: (H,W,4) or (4,H,W) array in [0,1023] or [0,1].
    Returns:
        (H,W,3) in [0,1].
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")

    if arr.shape[-1] == 4:
        # (H,W,4)
        GR = arr[..., 0]
        R  = arr[..., 1]
        B  = arr[..., 2]
        GB = arr[..., 3]
    elif arr.shape[0] == 4:
        # (4,H,W)
        GR = arr[0]
        R  = arr[1]
        B  = arr[2]
        GB = arr[3]
    else:
        raise ValueError(f"Expected shape (...,4), got {arr.shape}")

    max_val = arr.max()
    scale = 1023.0 if max_val > 2.0 else 1.0

    G  = (GR + GB) / (2.0 * scale)
    Rn = (R / scale) * r_gain
    Bn = (B / scale) * b_gain

    rgb = np.stack([Rn, G, Bn], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb
