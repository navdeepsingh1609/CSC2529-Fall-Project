# File: models/mambair_teacher.py
import torch
import torch.nn as nn

from models.frequency_block import FrequencyDomainBlock
from basicsr.archs.mambair_arch import MambaIR as OfficialMambaIR


class FrequencyAwareTeacher(nn.Module):
    """
    Frequency-aware teacher model combining MambaIR with a frequency domain branch.

    Architecture:
    - Spatial Branch: MambaIR backbone processing 4-channel raw input.
    - Frequency Branch: FrequencyDomainBlock processing 4-channel raw input.
    - Fusion: Gated residual connection where frequency features modulate spatial features.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        img_size: int = 256,
        embed_dim: int = 96,
        depths=(2, 2, 2, 2),
        **kwargs,
    ):
        super().__init__()
        
        # 1. Spatial Branch (MambaIR)
        self.spatial_model = OfficialMambaIR(
            img_size=img_size,
            in_chans=in_channels,
            out_chans=in_channels,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=[3, 6, 12, 24],
            window_size=8,
            ssm_d_state=16,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=2.0,
        )

        # 2. Frequency Branch
        self.frequency_model = FrequencyDomainBlock(in_channels=in_channels)

        # 3. Channel Gating (Global Avg Pool + Conv + Sigmoid)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

        # 4. Final Fusion
        self.fusion = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: (B, 4, H, W) Raw UDC-SIT Bayer-like input, normalized to [0, 1].

        Returns:
            restored_4ch_output: (B, 4, H, W) Final restored output.
            spatial_features:    (B, 4, H, W) Intermediate spatial features for KD.
            freq_features:       (B, 4, H, W) Intermediate frequency features for KD.
        """
        # Spatial path: MambaIR
        spatial_features = self.spatial_model(x)

        # Frequency path: FFT block on raw input
        freq_features = self.frequency_model(x)

        # Channel-wise gating
        gate = self.channel_gate(spatial_features)
        freq_mod = freq_features * gate

        # Residual fusion
        fused = spatial_features + freq_mod

        # Final projection
        restored_4ch_output = self.fusion(fused)

        return restored_4ch_output, spatial_features, freq_features


# Alias for backward compatibility if needed
FrequencyAwareTeacherV2 = FrequencyAwareTeacher
