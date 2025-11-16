import torch
import torch.nn as nn
from models.frequency_block import FrequencyDomainBlock

# This import is correct
from basicsr.archs.mambair_arch import MambaIR as OfficialMambaIR

class FrequencyAwareTeacher(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, img_size=256, 
                 embed_dim=96, depths=[2, 2, 2, 2], **kwargs):
        super().__init__()
        
        # 1. The main MambaIR backbone (for spatial features)
        self.spatial_model = OfficialMambaIR(
            img_size=img_size,
            in_chans=in_channels,
            out_chans=in_channels,  # We now know it outputs 4 channels
            embed_dim=embed_dim,
            depths=depths,
            num_heads=[3, 6, 12, 24],
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

        # 3. --- NEW FIX ---
        # Add a head to the spatial branch to reduce 4 channels -> 3 channels
        self.spatial_head = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # --- END NEW FIX ---
        
        # Head for the frequency branch (this was already correct)
        self.freq_head = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Fusion layer (this was already correct, 3 + 3 = 6)
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x shape: (B, 4, H, W)
        
        # --- [FIXED BRANCH 1] ---
        # Branch 1: Spatial path
        spatial_features = self.spatial_model(x) # (B, 4, H, W)
        spatial_out = self.spatial_head(spatial_features) # (B, 3, H, W)
        # --- END FIX ---
        
        # Branch 2: Frequency path (Correct)
        freq_features = self.frequency_model(x) # (B, 4, H, W)
        freq_out = self.freq_head(freq_features) # (B, 3, H, W)
        
        # Concatenate the two 3-channel outputs
        fused_features = torch.cat([spatial_out, freq_out], dim=1) # (B, 6, H, W)

        # --- [DEBUG] PRINTING SHAPES ---
        # The debug prints are still here so you can confirm the fix.
        print("\n--- [DEBUG] SHAPES ---")
        print(f"spatial_features shape: {spatial_features.shape} <-- MambaIR output")
        print(f"spatial_out shape: {spatial_out.shape}      <-- After new head")
        print(f"freq_out shape: {freq_out.shape}      <-- Freq head output")
        print(f"fused_features shape: {fused_features.shape}  <-- This MUST be 6 channels")
        print("----------------------\n")
        # --- END DEBUG ---

        final_out = self.fusion(fused_features)
        
        return final_out