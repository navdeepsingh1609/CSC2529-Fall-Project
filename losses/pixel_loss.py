"""
Pixel-wise Loss Functions.

Implements robust reconstruction losses for spatial domain supervision.
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1-like robust loss).
    
    Formulated as: sqrt((x - y)^2 + eps^2)
    It is more robust to outliers than L2 loss and smoother than L1 loss near zero.
    
    Args:
        eps (float): Small constant for numerical stability (default: 1e-6).
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()
