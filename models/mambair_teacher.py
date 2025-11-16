import torch
import torch.nn as nn
from models.frequency_block import FrequencyDomainBlock
from basicsr.archs.mambair_arch import MambaIR as OfficialMambaIR

class FrequencyAwareTeacher(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, img_size=256, 
                 embed_dim=96, depths=[2, 2, 2, 2], **kwargs):
        super().__init__()
        
        # 1. MambaIR backbone
        self.spatial_model = OfficialMambaIR(
            img_size=img_size,
            in_chans=in_channels,
            out_chans=in_channels,  # Outputs 4 channels
            embed_dim=embed_dim,
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=8,
            ssm_d_state=16,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=2.0,
        )
        
        # 2. Frequency block
        self.frequency_model = FrequencyDomainBlock(
            in_channels=in_channels
        )

        # 3. Heads
        self.spatial_head = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.freq_head = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 4. Fusion
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Branch 1: Spatial path
        spatial_features = self.spatial_model(x) # (B, 4, 256, 256)
        spatial_out = self.spatial_head(spatial_features) # (B, 3, 256, 256)
        
        # Branch 2: Frequency path
        freq_features = self.frequency_model(x)
        freq_out = self.freq_head(freq_features)
        
        # Concatenate
        fused_features = torch.cat([spatial_out, freq_out], dim=1)
        
        # Remove debug prints if they are still there
        
        final_out = self.fusion(fused_features)
        
        # --- [THIS IS THE MODIFICATION] ---
        # Return the final image AND the intermediate spatial feature map
        return final_out, spatial_features
        # --- [END MODIFICATION] ---