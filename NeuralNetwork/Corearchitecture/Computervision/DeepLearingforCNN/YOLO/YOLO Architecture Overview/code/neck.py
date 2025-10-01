import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNLikeNeck(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024]):
        super(FPNLikeNeck, self).__init__()
        # Lateral convolutions to align channels
        self.lateral1 = nn.Conv2d(in_channels[2], 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels[1], 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels[0], 256, kernel_size=1)
        # Smooth convolutions after fusion
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, features):
        # Features: List of [C3, C4, C5] from backbone (e.g., 52x52, 26x26, 13x13)
        C3, C4, C5 = features
        # Lateral connections
        P5 = F.leaky_relu(self.lateral1(C5), 0.1)  # 13x13
        # Upsample and merge
        P4 = F.leaky_relu(self.lateral2(C4), 0.1) + F.interpolate(P5, scale_factor=2, mode='nearest')  # 26x26
        P3 = F.leaky_relu(self.lateral3(C3), 0.1) + F.interpolate(P4, scale_factor=2, mode='nearest')  # 52x52
        # Smooth fused features
        P5 = F.leaky_relu(self.smooth1(P5), 0.1)
        P4 = F.leaky_relu(self.smooth2(P4), 0.1)
        P3 = F.leaky_relu(self.smooth3(P3), 0.1)
        return [P3, P4, P5]  # Fused feature maps for head

# Example usage with dummy backbone
class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1)  # 416x416 -> 208x208
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 208x208 -> 104x104
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # 104x104 -> 52x52
        self.conv4 = nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1)  # 52x52 -> 26x26
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # 26x26 -> 13x13

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1)
        C3 = F.leaky_relu(self.conv2(x), 0.1)  # 52x52
        C4 = F.leaky_relu(self.conv3(C3), 0.1)  # 26x26
        C5 = F.leaky_relu(self.conv4(C4), 0.1)  # 13x13
        return [C3, C4, C5]

# Combine backbone and neck
backbone = SimpleBackbone()
neck = FPNLikeNeck(in_channels=[256, 512, 1024])
input_img = torch.randn(1, 3, 416, 416)
features = backbone(input_img)
fused_features = neck(features)
for i, feat in enumerate(fused_features):
    print(f"Fused feature map {i+1} shape: {feat.shape}")
