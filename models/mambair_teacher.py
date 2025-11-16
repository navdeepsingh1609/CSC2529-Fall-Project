import torch
import torch.nn as nn
from models.frequency_block import FrequencyDomainBlock
from basicsr.archs.mambair_arch import MambaIR as OfficialMambaIR

class FrequencyAwareTeacher(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, img_size=256, 
                 embed_dim=96, depths=[2, 2, 2, 2], **kwargs):
        super().__init__()
        print(f"--- [Teacher] Initializing with {in_channels} in-channels and {out_channels} out-channels.")
        
        # 1. The MambaIR backbone (4-ch in, 4-ch out)
        self.spatial_model = OfficialMambaIR(
            img_size=img_size,
            in_chans=in_channels,
            out_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=8,
            ssm_d_state=16,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=2.0,
        )
        
        # 2. The frequency domain block (4-ch in, 4-ch out)
        self.frequency_model = FrequencyDomainBlock(
            in_channels=in_channels
        )

        # 3. Fusion layer (4 + 4 = 8 channels in, 4 channels out)
        self.fusion = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        
        print("--- [Teacher] Model initialized.")
        
    def forward(self, x):
        # print(f"--- [Teacher] Input shape: {x.shape}")
        
        # Branch 1: Spatial path
        spatial_features = self.spatial_model(x) 
        # print(f"--- [Teacher] Spatial branch output shape: {spatial_features.shape}")
        
        # Branch 2: Frequency path
        freq_features = self.frequency_model(x) 
        # print(f"--- [Teacher] Freq branch output shape: {freq_features.shape}")
        
        # Fuse the two 4-channel outputs
        fused_features = torch.cat([spatial_features, freq_features], dim=1) 
        # print(f"--- [Teacher] Fused features shape: {fused_features.shape}")

        # Run fusion to get a single 4-channel feature map
        restored_4ch_output = self.fusion(fused_features) 
        # print(f"--- [Teacher] Final 4ch output shape: {restored_4ch_output.shape}")
        
        # The main output is the restored 4-channel image.
        # We also return the intermediate branch outputs for distillation.
        return restored_4ch_output, spatial_features, freq_features