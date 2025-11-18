# File: losses/pixel_loss.py
import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (a smooth L1) widely used in image restoration.

    L(x, y) = mean( sqrt( (x - y)^2 + eps^2 ) )
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()
