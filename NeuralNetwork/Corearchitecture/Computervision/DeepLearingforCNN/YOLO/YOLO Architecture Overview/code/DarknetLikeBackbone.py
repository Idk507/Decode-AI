import torch
import torch.nn as nn
import torch.nn.functional as F

class DarknetLikeBackbone(nn.Module):
    def __init__(self):
        super(DarknetLikeBackbone, self).__init__()
        # Layer 1: 416x416 -> 208x208
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2: 208x208 -> 104x104
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Layer 3: 104x104 -> 52x52
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # Layer 4: 52x52 -> 26x26
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        # Layer 5: 26x26 -> 13x13
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)  # 208x208
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), 0.1)  # 104x104
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)), 0.1)  # 52x52
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)), 0.1)  # 26x26
        x5 = F.leaky_relu(self.bn5(self.conv5(x4)), 0.1)  # 13x13
        return [x3, x4, x5]  # Multi-scale feature maps

# Example usage
backbone = DarknetLikeBackbone()
input_img = torch.randn(1, 3, 416, 416)
features = backbone(input_img)
for i, feat in enumerate(features):
    print(f"Feature map {i+1} shape: {feat.shape}")
