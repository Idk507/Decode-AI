import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDarknetBackbone(nn.Module):
    def __init__(self):
        super(SimpleDarknetBackbone, self).__init__()
        # Layer 1: Downsample to 208x208
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 416x416 -> 208x208
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 208x208 -> 104x104
        # Layer 2: Downsample to 52x52
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 104x104 -> 52x52
        self.bn2 = nn.BatchNorm2d(128)
        # Layer 3: Downsample to 26x26
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 52x52 -> 26x26
        self.bn3 = nn.BatchNorm2d(256)
        # Layer 4: Downsample to 13x13
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 26x26 -> 13x13
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self, x):
        # Feature extraction with multi-scale outputs
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)  # 208x208
        x1 = self.pool1(x1)  # 104x104
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), 0.1)  # 52x52
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)), 0.1)  # 26x26
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)), 0.1)  # 13x13
        return [x2, x3, x4]  # Multi-scale feature maps

# Example usage
backbone = SimpleDarknetBackbone()
input_img = torch.randn(1, 3, 416, 416)
features = backbone(input_img)
for i, feat in enumerate(features):
    print(f"Feature map {i+1} shape: {feat.shape}")
