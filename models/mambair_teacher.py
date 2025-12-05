import torch
import torch.nn as nn
from models.frequency_block import FrequencyDomainBlock
from basicsr.archs.mambairv2_arch import MambaIRv2 as OfficialMambaIR


class FrequencyAwareTeacher(nn.Module):
    """
    Frequency-Aware Teacher Network for UDC Image Restoration
    
    Integrates a Spatial Branch (MambaIR v2) and a parallel Frequency Branch
    to handle both global diffraction artifacts and local details
    
    Args:
        in_channels (int): Number of input channels (default: 4 for RAW)
        out_channels (int): Number of output channels (default: 4)
        img_size (int): Input image size (default: 256)
        embed_dim (int): Embedding dimension for MambaIRv2 (default: 96)
        depths (tuple): Depths of MambaIRv2 stages (default: (2, 2, 2, 2))
        variant (str): 'v1' or 'v2' to select fusion strategy (default: 'v2')
        **kwargs: Additional keyword arguments for MambaIRv2
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        img_size: int = 256,
        embed_dim: int = 96,
        depths=(2, 2, 2, 2),
        variant: str = "v2",
        **kwargs,
    ):
        super().__init__()
        self.variant = variant
        
        # 1. Spatial Branch (MambaIR v2)
        # MambaIRv2 requires: inner_rank, num_tokens, convffn_kernel_size for ASSM
        # upscale=1, upsampler='' for denoising (no upscaling, same resolution output)
        self.spatial_model = OfficialMambaIR(
            img_size=img_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=[3, 6, 12, 24],
            window_size=16,
            d_state=16,
            inner_rank=32,           # For Attentive State-Space Module
            num_tokens=64,           # Token dictionary size for SGN
            convffn_kernel_size=5,   # Kernel size for ConvFFN
            mlp_ratio=2.0,
            upscale=1,               # No upscaling (denoising mode)
            upsampler='',            # No upsampler (denoising mode)
        )

        # 2. Frequency Branch
        self.frequency_model = FrequencyDomainBlock(in_channels=in_channels)

        # 3. Gating (Only for v2)
        if self.variant == "v2":
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.channel_gate = None

        # 4. Final Fusion (Concatenation -> Conv)
        # Both v1 and v2 diagrams show Concatenation.
        # Input to fusion is Spatial (4ch) + Freq (4ch) = 8ch
        self.fusion = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

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

        if self.variant == "v2":
            # Gated Fusion (v2): Modulate frequency features before concatenation
            gate = self.channel_gate(spatial_features)
            freq_to_fuse = freq_features * gate
        else:
            # Baseline Fusion (v1): Direct concatenation
            freq_to_fuse = freq_features

        # Concatenation (as per diagrams)
        # Spatial (4) + Freq (4) = 8 channels
        fused = torch.cat([spatial_features, freq_to_fuse], dim=1)  # (B, 8, H, W)

        # Final projection 8->4
        restored_4ch_output = self.fusion(fused)

        return restored_4ch_output, spatial_features, freq_features


# Alias for backward compatibility if needed
FrequencyAwareTeacherV2 = FrequencyAwareTeacher
