# File: models/mambair_teacher.py
import torch
import torch.nn as nn
from models.external.MambaIR.models.mambair import MambaIR as OfficialMambaIR
from models.frequency_block import FrequencyDomainBlock

class FrequencyAwareTeacher(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, img_size=256, 
                 embed_dim=96, depths=[2, 2, 2, 2], **kwargs):
        super().__init__()
        
        # 1. The main MambaIR backbone (for spatial features)
        # We need to configure MambaIR based on its own defaults.
        # This part requires inspecting the MambaIR repo for config.
        # Let's assume a basic setup for restoration:
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
            # ... other MambaIR params
        )
        
        # 2. The frequency domain block (for global flare)
        self.frequency_model = FrequencyDomainBlock(
            in_channels=in_channels
        )

        # 3. A simple fusion layer
        # We'll fuse the *outputs* of both branches
        # MambaIR outputs (B, C, H, W). Freq block outputs (B, C_in, H, W)
        # We need the Freq block output to match the target channel count (3)
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
        
        # Your description also mentioned feature distillation.
        # To enable that, you'd modify MambaIR to return intermediate
        # features and return them here. For now, we just return the final output.
        # For distillation, you might return: (final_out, spatial_features, freq_features)
        
        return final_out