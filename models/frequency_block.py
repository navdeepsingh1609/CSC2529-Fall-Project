import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDomainBlock(nn.Module):
    """
    A block that processes features in the frequency domain.
    """
    def __init__(self, in_channels, num_kernels=64):
        super().__init__()
        self.fft_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, num_kernels * 2, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_kernels * 2, in_channels * 2, kernel_size=1)
        )
        print("--- [FrequencyDomainBlock] Initialized.")
        
    def forward(self, x):
        # print(f"--- [FreqBlock] Input shape: {x.shape}")
        
        # 1. Apply FFT
        fft_features = torch.fft.fft2(x, dim=(-2, -1))
        fft_features = torch.view_as_real(fft_features) 
        fft_features = fft_features.permute(0, 1, 4, 2, 3).contiguous()
        B, C, _, H, W = fft_features.shape
        fft_features = fft_features.view(B, C * 2, H, W) 
        # print(f"--- [FreqBlock] FFT features shape: {fft_features.shape}")

        # 2. Process frequency features
        processed_fft = self.fft_conv(fft_features)
        
        # 3. Apply iFFT
        processed_fft = processed_fft.view(B, C, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
        processed_fft = torch.view_as_complex(processed_fft)
        
        ifft_features = torch.fft.ifft2(processed_fft, dim=(-2, -1))
        # print(f"--- [FreqBlock] Output shape: {ifft_features.real.shape}")
        
        return ifft_features.real