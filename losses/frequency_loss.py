"""
Frequency Domain Loss Functions (for constraining model predictions in the Fourier domain):
- Amplitude Loss: Minimizes L1 difference of spectral amplitudes
- Phase Loss: Minimizes L1 difference of spectral phases
- Multi-Scale Loss: Applies frequency constraints at multiple resolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class FrequencyLoss(nn.Module):
    """
    Computes loss in the frequency domain.
    
    Args:
        loss_type (str): 'l1' or 'l2' (default: 'l1').
        alpha (float): Weight for amplitude loss (default: 1.0).
        beta (float): Weight for phase loss (default: 1.0).
    """
    def __init__(self, loss_type='l1', alpha=1.0, beta=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta


class FFTAmpPhaseMultiScaleLoss(nn.Module):
    """
    Multi-scale frequency domain loss combining amplitude and phase differences.
    
    Computes:
        L_fft = Σ w_s * [ λ_amp * L1(Amp_pred - Amp_gt) + λ_phase * (1 - cos(Phase_pred - Phase_gt)) ]
    
    Supports optional low-frequency emphasis via radial weighting.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        focus_low_freq: bool = True,
        cutoff: float = 0.25,
        lambda_amp: float = 1.0,
        lambda_phase: float = 0.5,
        scales=(1.0, 0.5),
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.focus_low_freq = focus_low_freq
        self.cutoff = cutoff
        self.lambda_amp = lambda_amp
        self.lambda_phase = lambda_phase
        self.scales = tuple(scales)

    def _frequency_weight(self, H: int, W: int, device, dtype):
        """Generates a radial low-pass filter mask."""
        fy = torch.fft.fftfreq(H, device=device, dtype=dtype).view(1, 1, H, 1)
        fx = torch.fft.fftfreq(W, device=device, dtype=dtype).view(1, 1, 1, W)
        radius = torch.sqrt(fx * fx + fy * fy)
        return torch.exp(- (radius / self.cutoff) ** 2)

    def _fft_components(self, x: torch.Tensor):
        """Computes amplitude and phase from 2D FFT."""
        fft_x = torch.fft.fft2(x, dim=(-2, -1))
        return torch.abs(fft_x), torch.angle(fft_x)

    def _single_scale_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes loss for a single scale."""
        B, C, H, W = pred.shape
        amp_pred, phase_pred = self._fft_components(pred)
        amp_target, phase_target = self._fft_components(target)

        w = None
        if self.focus_low_freq:
            w = self._frequency_weight(H, W, pred.device, pred.dtype)
            amp_pred = amp_pred * w
            amp_target = amp_target * w

        # Amplitude Loss (L1)
        amp_loss = F.l1_loss(amp_pred, amp_target)

        # Phase Loss (1 - cos(delta))
        delta_phase = phase_pred - phase_target
        phase_loss_map = 1.0 - torch.cos(delta_phase)
        
        if w is not None:
            phase_loss_map = phase_loss_map * w
            
        phase_loss = phase_loss_map.mean()

        return self.lambda_amp * amp_loss + self.lambda_phase * phase_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        total_loss = 0.0
        weight_sum = 0.0

        for i, s in enumerate(self.scales):
            if s == 1.0:
                pred_s, target_s = pred, target
            else:
                H, W = pred.shape[-2:]
                new_H, new_W = max(int(H * s), 2), max(int(W * s), 2)
                if new_H < 2 or new_W < 2:
                    continue
                
                pred_s = F.interpolate(pred, size=(new_H, new_W), mode='bilinear', align_corners=False)
                target_s = F.interpolate(target, size=(new_H, new_W), mode='bilinear', align_corners=False)

            w_s = 1.0 / (i + 1)
            total_loss += w_s * self._single_scale_loss(pred_s, target_s)
            weight_sum += w_s

        if weight_sum == 0.0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        return self.loss_weight * (total_loss / weight_sum)


class FFTAmplitudeLoss(nn.Module):
    """
    Legacy amplitude-only FFT loss.
    Retained for compatibility with Model V1.
    """
    def __init__(
        self,
        loss_weight: float = 1.0,
        focus_low_freq: bool = True,
        cutoff: float = 0.25
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.focus_low_freq = focus_low_freq
        self.cutoff = cutoff
        self.l1_loss = nn.L1Loss()

    def _frequency_weight(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        fy = torch.fft.fftfreq(H, device=x.device).view(1, 1, H, 1)
        fx = torch.fft.fftfreq(W, device=x.device).view(1, 1, 1, W)
        radius = torch.sqrt(fx * fx + fy * fy)
        return torch.exp(- (radius / self.cutoff) ** 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
        fft_target = torch.fft.fft2(target, dim=(-2, -1))

        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)

        if self.focus_low_freq:
            w = self._frequency_weight(pred)
            amp_pred = amp_pred * w
            amp_target = amp_target * w

        return self.l1_loss(amp_pred, amp_target) * self.loss_weight
