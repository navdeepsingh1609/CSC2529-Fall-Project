"""
U-Net Student Network for UDC Image Restoration

Lightweight architecture designed for efficient inference, incorporating
dual-domain features distilled from the teacher network
"""

import torch
import torch.nn as nn
from models.basic_block import ResBlock, ConvBlock
from models.frequency_block import FrequencyDomainBlock

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: deeper feature, x2: skip
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetStudent(nn.Module):
    """
    Frequency-aware U-Net student.

    - Standard U-Net encoder/decoder on 4-ch raw.
    - FrequencyDomainBlock at:
        * bottleneck (global flare handling)
        * highest-res skip (diffraction / high-frequency artifacts)
    - feature_head projects bottleneck to 4-ch for KD.
    """
    def __init__(self, in_channels=4, out_channels=4, n_base_filters=64, bilinear=True, enable_skip_freq=True):
        super(UNetStudent, self).__init__()
        self.enable_skip_freq = enable_skip_freq
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc   = ConvBlock(in_channels, n_base_filters)
        self.down1 = Down(n_base_filters, n_base_filters * 2)
        self.down2 = Down(n_base_filters * 2, n_base_filters * 4)
        self.down3 = Down(n_base_filters * 4, n_base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_base_filters * 8, n_base_filters * 16 // factor)

        # Frequency blocks
        bottleneck_channels = n_base_filters * 16 // factor
        self.freq_bottleneck = FrequencyDomainBlock(in_channels=bottleneck_channels)
        
        if self.enable_skip_freq:
            self.freq_skip_high = FrequencyDomainBlock(in_channels=n_base_filters)
        else:
            self.freq_skip_high = None

        self.up1 = Up(n_base_filters * 16, n_base_filters * 8 // factor, bilinear)
        self.up2 = Up(n_base_filters * 8,  n_base_filters * 4 // factor, bilinear)
        self.up3 = Up(n_base_filters * 4,  n_base_filters * 2 // factor, bilinear)
        self.up4 = Up(n_base_filters * 2,  n_base_filters, bilinear)

        self.outc = nn.Conv2d(n_base_filters, out_channels, kernel_size=1)

        # Feature map for KD: project bottleneck to 4 channels
        self.feature_head = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)


    def forward(self, x):
        # Encoder
        x1 = self.inc(x)        # (B,64,H,W)
        x2 = self.down1(x1)     # (B,128,H/2,W/2)
        x3 = self.down2(x2)     # (B,256,H/4,W/4)
        x4 = self.down3(x3)     # (B,512,H/8,W/8)
        bottleneck = self.down4(x4)  # (B,512,H/16,W/16) if bilinear

        # Frequency at bottleneck (global flare)
        bottleneck = bottleneck + self.freq_bottleneck(bottleneck)

        # Feature map for KD from bottleneck
        feature_map = self.feature_head(bottleneck)  # (B,4,H/16,W/16)

        # Frequency on high-res skip (x1) - Conditional
        if self.enable_skip_freq and self.freq_skip_high is not None:
            x1_enh = x1 + self.freq_skip_high(x1)
        else:
            x1_enh = x1

        # Decoder
        x = self.up1(bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1_enh)

        final_4ch_output = self.outc(x)  # (B,4,H,W)

        return final_4ch_output, feature_map
