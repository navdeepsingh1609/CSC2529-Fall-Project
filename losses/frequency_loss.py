# File: losses/frequency_loss.py
import torch
import torch.nn as nn

class FFTAmplitudeLoss(nn.Module):
    """
    Frequency-domain loss on amplitude of 2D FFT.

    This version applies a smooth radial weighting that
    emphasises low frequencies (where UDC flare lives),
    while still using the full spectrum.

    pred, target: (B, C, H, W) in real space.
    """
    def __init__(self,
                 loss_weight: float = 1.0,
                 focus_low_freq: bool = True,
                 cutoff: float = 0.25):
        """
        Args:
            loss_weight: global multiplier.
            focus_low_freq: if True, apply smooth low-pass weight.
            cutoff: controls how fast high frequencies are down-weighted.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.focus_low_freq = focus_low_freq
        self.cutoff = cutoff
        self.l1_loss = nn.L1Loss()

    def _frequency_weight(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build a smooth low-pass weight mask in the frequency domain.
        Returns a tensor of shape (1, 1, H, W).
        """
        B, C, H, W = x.shape
        device = x.device

        fy = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)
        fx = torch.fft.fftfreq(W, device=device).view(1, 1, 1, W)
        radius = torch.sqrt(fx * fx + fy * fy)  # normalized radial freq

        # Smooth low-pass: 1 at DC, decays as radius increases
        weight = torch.exp(- (radius / self.cutoff) ** 2)  # (1,1,H,W)
        return weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, C, H, W) float32
        """
        fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
        fft_target = torch.fft.fft2(target, dim=(-2, -1))

        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)

        if self.focus_low_freq:
            w = self._frequency_weight(pred)  # (1,1,H,W)
            amp_pred = amp_pred * w
            amp_target = amp_target * w

        loss = self.l1_loss(amp_pred, amp_target)
        return loss * self.loss_weight
