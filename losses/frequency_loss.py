# File: losses/frequency_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTAmpPhaseMultiScaleLoss(nn.Module):
    """
    Amplitude + Phase FFT loss with optional low-frequency emphasis
    and multi-scale computation.

    pred, target: (B, C, H, W) in real space, typically in [0, 1].

    L_fft = sum_s w_s * [ λ_amp * L1(amp_pred_s - amp_gt_s)
                        + λ_phase * E[1 - cos(phase_pred_s - phase_gt_s)] ]
    where each term is optionally weighted by a smooth low-pass mask.
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
        """
        Args:
            loss_weight: global multiplier on the final loss.
            focus_low_freq: if True, apply smooth low-pass weighting.
            cutoff: radial cutoff for the low-pass weight (in normalized freq).
            lambda_amp: weight for amplitude term.
            lambda_phase: weight for phase term.
            scales: iterable of scale factors (e.g. (1.0, 0.5) for full + half).
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.focus_low_freq = focus_low_freq
        self.cutoff = cutoff
        self.lambda_amp = lambda_amp
        self.lambda_phase = lambda_phase
        self.scales = tuple(scales)

    def _frequency_weight(self, H: int, W: int, device, dtype):
        """
        Build a smooth low-pass weight mask in the frequency domain.
        Returns tensor of shape (1, 1, H, W).
        """
        fy = torch.fft.fftfreq(H, device=device, dtype=dtype).view(1, 1, H, 1)
        fx = torch.fft.fftfreq(W, device=device, dtype=dtype).view(1, 1, 1, W)
        radius = torch.sqrt(fx * fx + fy * fy)  # normalized radial frequency
        weight = torch.exp(- (radius / self.cutoff) ** 2)
        return weight  # (1, 1, H, W)

    def _fft_components(self, x: torch.Tensor):
        """
        Compute amplitude and phase of 2D FFT.
        x: (B, C, H, W), real.
        Returns:
            amp:   (B, C, H, W)
            phase: (B, C, H, W)
        """
        fft_x = torch.fft.fft2(x, dim=(-2, -1))
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        return amp, phase

    def _single_scale_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute amplitude + phase loss at a single resolution.
        pred, target: (B, C, H, W)
        """
        B, C, H, W = pred.shape
        device = pred.device
        dtype = pred.dtype

        amp_pred, phase_pred = self._fft_components(pred)
        amp_target, phase_target = self._fft_components(target)

        if self.focus_low_freq:
            w = self._frequency_weight(H, W, device, dtype)  # (1, 1, H, W)
            # Broadcast to (B, C, H, W)
            amp_pred = amp_pred * w
            amp_target = amp_target * w
        else:
            w = None

        # Amplitude term (L1)
        amp_loss = F.l1_loss(amp_pred, amp_target)

        # Phase term: periodic loss 1 - cos(Δφ)
        delta_phase = phase_pred - phase_target
        phase_loss_map = 1.0 - torch.cos(delta_phase)
        if w is not None:
            phase_loss_map = phase_loss_map * w
        phase_loss = phase_loss_map.mean()

        return self.lambda_amp * amp_loss + self.lambda_phase * phase_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, C, H, W) real tensors.
        """
        # Ensure float32 for stable FFT
        pred = pred.float()
        target = target.float()

        total = 0.0
        weight_sum = 0.0

        for i, s in enumerate(self.scales):
            if s == 1.0:
                pred_s, target_s = pred, target
            else:
                # Downsample spatially if large enough
                H, W = pred.shape[-2:]
                new_H = max(int(H * s), 2)
                new_W = max(int(W * s), 2)
                if new_H < 2 or new_W < 2:
                    continue

                pred_s = F.interpolate(pred, size=(new_H, new_W),
                                       mode='bilinear', align_corners=False)
                target_s = F.interpolate(target, size=(new_H, new_W),
                                         mode='bilinear', align_corners=False)

            # Simple decreasing weight for finer scales
            w_s = 1.0 / (i + 1)
            total = total + w_s * self._single_scale_loss(pred_s, target_s)
            weight_sum += w_s

        if weight_sum == 0.0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        return self.loss_weight * (total / weight_sum)


# (Optional) Keep the old amplitude-only loss for reference / debugging
class FFTAmplitudeLoss(nn.Module):
    """
    Original amplitude-only FFT loss (kept for compatibility / debugging).
    Not used in the new training pipeline.
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
        device = x.device
        fy = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)
        fx = torch.fft.fftfreq(W, device=device).view(1, 1, 1, W)
        radius = torch.sqrt(fx * fx + fy * fy)
        weight = torch.exp(- (radius / self.cutoff) ** 2)
        return weight

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

        loss = self.l1_loss(amp_pred, amp_target)
        return loss * self.loss_weight
