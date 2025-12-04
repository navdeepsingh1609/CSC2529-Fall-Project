# File: models/mambair_teacher.py
import torch
import torch.nn as nn

from models.frequency_block import FrequencyDomainBlock
from basicsr.archs.mambair_arch import MambaIR as OfficialMambaIR


class FrequencyAwareTeacherV2(nn.Module):
    """
    Frequency-aware teacher based on MambaIR + FFT frequency branch.

    - Spatial branch: Official MambaIR, 4-ch in → 4-ch out (raw Bayer domain).
    - Frequency branch: FrequencyDomainBlock on the 4-ch raw input.
    - Fusion: residual-style, freq acts as a gated correction on top of spatial.

    Returns:
        restored_4ch_output: (B, 4, H, W)
        spatial_features:    (B, 4, H, W)   # for KD
        freq_features:       (B, 4, H, W)   # for KD / analysis
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
        print(f"--- [Teacher] Initializing with {in_channels} in-channels and {out_channels} out-channels.")

        # 1. Spatial branch: MambaIR backbone (4-ch → 4-ch)
        self.spatial_model = OfficialMambaIR(
            img_size=img_size,
            in_chans=in_channels,
            out_chans=in_channels,  # 4-ch output in Bayer domain
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=[3, 6, 12, 24],
            window_size=8,
            ssm_d_state=16,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=2.0,
        )

        # 2. Frequency branch: FFT block on raw input (4-ch)
        self.frequency_model = FrequencyDomainBlock(
            in_channels=in_channels
        )

        # 3. Very lightweight channel gate (global avg pool + 1x1 conv + sigmoid)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # (B, C, H, W) -> (B, C, 1, 1)
            nn.Conv2d(in_channels, in_channels, 1),  # learn per-channel scaling
            nn.Sigmoid()
        )

        # 4. Final 1x1 fusion to 4-ch output
        self.fusion = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        print("--- [Teacher] Model initialized (MambaIR + Freq residual gating).")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 4, H, W) raw UDC-SIT Bayer-like input normalized to [0, 1].

        Returns:
            restored_4ch_output: (B, 4, H, W)
            spatial_features:    (B, 4, H, W)
            freq_features:       (B, 4, H, W)
        """
        # Spatial path: MambaIR
        spatial_features = self.spatial_model(x)  # (B, 4, H, W)

        # Frequency path: FFT block on raw input
        freq_features = self.frequency_model(x)   # (B, 4, H, W)

        # Channel-wise gate from spatial features
        gate = self.channel_gate(spatial_features)    # (B, 4, 1, 1) in [0,1]
        freq_mod = freq_features * gate               # gated correction

        # Residual fusion: spatial + gated frequency correction
        fused = spatial_features + freq_mod           # (B, 4, H, W)

        # Final 4-ch output
        restored_4ch_output = self.fusion(fused)      # (B, 4, H, W)

        # For KD we still expose spatial & freq branch outputs
        return restored_4ch_output, spatial_features, freq_features
