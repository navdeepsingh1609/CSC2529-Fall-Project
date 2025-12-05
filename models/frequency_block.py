"""
Frequency Domain Processing Modules implementing blocks for handling Fourier domain operations, including:
- FFT/IFFT transformations
- Amplitude and phase processing
- Frequency-domain feature extraction
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class FrequencyDomainBlock(nn.Module):
    """
    Extracts frequency-domain features using 2D FFT
    
    Processes the amplitude and phase components of the input signal
    to capture global degradation patterns inherent in UDC images
    
    Args:
        in_channels (int): Number of input channels
    """
    def __init__(self, in_channels, num_kernels=64):
        super().__init__()
        self.fft_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, num_kernels * 2, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_kernels * 2, in_channels * 2, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,C,H,W) float tensor.
        Returns:
            (B,C,H,W) float tensor (real part of iFFT).
        """
        with autocast(enabled=False):
            x_float32 = x.float()

            # 1. FFT over spatial dims
            fft_features = torch.fft.fft2(x_float32, dim=(-2, -1))
            fft_features = torch.view_as_real(fft_features)  # (B,C,H,W,2)
            fft_features = fft_features.permute(0, 1, 4, 2, 3).contiguous()
            B, C, _, H, W = fft_features.shape
            fft_features = fft_features.view(B, C * 2, H, W)  # (B,2C,H,W)

            # 2. Process in freq domain
            processed_fft = self.fft_conv(fft_features)

            # 3. Back to complex, then iFFT
            processed_fft = processed_fft.view(B, C, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
            processed_fft = torch.view_as_complex(processed_fft)

            ifft_features = torch.fft.ifft2(processed_fft, dim=(-2, -1))

        return ifft_features.real
