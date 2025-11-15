# File: losses/frequency_loss.py
import torch
import torch.nn as nn

class FFTAmplitudeLoss(nn.Module):
    """
    Computes the L1 loss between the amplitudes of the
    2D FFT of the predicted and target images.
    """
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        # pred, target shape: (B, C, H, W)
        
        # Compute 2D FFT
        fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
        fft_target = torch.fft.fft2(target, dim=(-2, -1))
        
        # Get amplitude
        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)
        
        # To focus on low frequencies (as flares are low-freq),
        # you could apply a mask here. For simplicity, we'll
        # compare the full amplitude spectrum.
        
        loss = self.l1_loss(amp_pred, amp_target)
        
        return loss * self.loss_weight