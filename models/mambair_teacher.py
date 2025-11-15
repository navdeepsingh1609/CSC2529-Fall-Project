import torch
import torch.nn as nn
# This is the line we fixed
from models.external.MambaIR.basicsr.archs.mambair_arch import MambaIR as OfficialMambaIR
from models.frequency_block import FrequencyDomainBlock

class FrequencyAwareTeacher(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, img_size=256, 
                 embed_dim=96, depths=[2, 2, 2, 2], **kwargs):
        super().__init__()
        
        # 1. The main MambaIR backbone (for spatial features)
        # This instantiation should work correctly with the new import.
        self.spatial_model = OfficialMambaIR(
            img_size=img_size,
            in_chans=in_channels,
            out_chans=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=[3, 6, 12, 24], # Example, tune as needed
            window_size=8,
            ssm_d_state=16,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=2.0,
        )
        
        # 2. The frequency domain block (for global flare)
        self.frequency_model = FrequencyDomainBlock(
            in_channels=in_channels
        )

        # 3. A simple fusion layer
        self.freq_head = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Fusion conv
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x shape: (B, 4, H, W)
        
        # Branch 1: Spatial path
        spatial_out = self.spatial_model(x) # (B, 3, H, W)
        
        # Branch 2: Frequency path
        freq_features = self.frequency_model(x) # (B, 4, H, W)
        freq_out = self.freq_head(freq_features) # (B, 3, H, W)
        
        # Fuse the two outputs
        fused_features = torch.cat([spatial_out, freq_out], dim=1)
        final_out = self.fusion(fused_features)
        
        return final_out