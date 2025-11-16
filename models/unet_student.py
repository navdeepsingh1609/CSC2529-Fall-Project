import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ConvBlock, Down, Up classes (unchanged) ---
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
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
# --- End of unchanged classes ---

class UNetStudent(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, n_base_filters=64, bilinear=True):
        super(UNetStudent, self).__init__()
        print(f"--- [Student] Initializing with {in_channels} in-channels and {out_channels} out-channels.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = ConvBlock(in_channels, n_base_filters)
        self.down1 = Down(n_base_filters, n_base_filters * 2)
        self.down2 = Down(n_base_filters * 2, n_base_filters * 4)
        self.down3 = Down(n_base_filters * 4, n_base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_base_filters * 8, n_base_filters * 16 // factor)
        
        self.up1 = Up(n_base_filters * 16, n_base_filters * 8 // factor, bilinear)
        self.up2 = Up(n_base_filters * 8, n_base_filters * 4 // factor, bilinear)
        self.up3 = Up(n_base_filters * 4, n_base_filters * 2 // factor, bilinear)
        self.up4 = Up(n_base_filters * 2, n_base_filters, bilinear)
        
        self.outc = nn.Conv2d(n_base_filters, out_channels, kernel_size=1)
        self.feature_head = nn.Conv2d(n_base_filters * 16 // factor, 4, kernel_size=1)
        print("--- [Student] Model initialized.")

    def forward(self, x):
        # print(f"--- [Student] Input shape: {x.shape}")
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        bottleneck = self.down4(x4)
        # print(f"--- [Student] Bottleneck shape: {bottleneck.shape}")
        
        feature_map = self.feature_head(bottleneck)
        # print(f"--- [Student] Feature map shape: {feature_map.shape}")
        
        x = self.up1(bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        final_4ch_output = self.outc(x)
        # print(f"--- [Student] Final 4ch output shape: {final_4ch_output.shape}")
        
        return final_4ch_output, feature_map