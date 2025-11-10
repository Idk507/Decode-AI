import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=1):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        # Final
        self.final = nn.Conv2d(64, out_classes, 1)
        self.sigmoid = nn.Sigmoid() if out_classes == 1 else nn.Softmax(dim=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder + Skip
        d4 = self.dec4(self.up4(b) + e4.shape[2:], e4)  # crop if needed
        d3 = self.dec3(self.up3(d4) + e3.shape[2:], e3)
        d2 = self.dec2(self.up2(d3) + e2.shape[2:], e2)
        d1 = self.dec1(self.up1(d2) + e1.shape[2:], e1)
        
        out = self.final(d1)
        return self.sigmoid(out) if out_classes == 1 else out

# Test
model = UNet(in_channels=3, out_classes=1)
x = torch.randn(1, 3, 572, 572)
y = model(x)
print(y.shape)  # → [1, 1, 388, 388]
