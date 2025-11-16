import torch
import torch.nn as nn
from models.frequency_block import FrequencyDomainBlock

# This import is correct
from basicsr.archs.mambair_arch import MambaIR as OfficialMambaIR

class FrequencyAwareTeacher(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, img_size=256, 
                 embed_dim=96, depths=[2, 2, 2, 2], **kwargs):
        super().__init__()
        
        # 1. The MambaIR backbone (now outputs 4 channels)
        self.spatial_model = OfficialMambaIR(
            img_size=img_size,
            in_chans=in_channels,
            out_chans=in_channels,  # Keep 4 channels
            embed_dim=embed_dim,
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=8,
            ssm_d_state=16,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=2.0,
        )
        
        # 2. The frequency domain block (already 4 channels)
        self.frequency_model = FrequencyDomainBlock(
            in_channels=in_channels
        )

        # 3. --- NEW ARCHITECTURE ---
        # REMOVED the old spatial_head and freq_head
        
        # Fusion layer (now 4 + 4 = 8 channels in, 4 channels out)
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
        # Final head to convert 4-channel restored to 3-channel RGB
        self.final_head = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # --- END NEW ARCHITECTURE ---
        
    def forward(self, x):
        # x shape: (B, 4, H, W)
        
        # Branch 1: Spatial path
        # spatial_features is the 4-channel restored output
        spatial_features = self.spatial_model(x) # (B, 4, H, W)
        
        # Branch 2: Frequency path
        # freq_features is the 4-channel restored output
        freq_features = self.frequency_model(x) # (B, 4, H, W)
        
        # Fuse the two 4-channel outputs
        fused_features = torch.cat([spatial_features, freq_features], dim=1) # (B, 8, H, W)

        # Run fusion to get a single 4-channel feature map
        fused_4ch_output = self.fusion(fused_features) # (B, 4, H, W)
        
        # Convert the 4-channel restored image to 3-channel RGB
        final_rgb_output = self.final_head(fused_4ch_output) # (B, 3, H, W)
        
        # For distillation, we provide the 3-channel output
        # AND the 4-channel intermediate feature map
        return final_rgb_output, fused_4ch_output