import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv3Head(nn.Module):
    def __init__(self, num_anchors=3, num_classes=80, in_channels=256):
        super(YOLOv3Head, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Output: (x, y, w, h, objectness, classes) per anchor
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        # x: Feature map from neck (batch, in_channels, S, S)
        x = self.conv(x)  # (batch, num_anchors*(5+num_classes), S, S)
        batch_size, _, S, _ = x.shape
        x = x.view(batch_size, S, S, self.num_anchors, 5 + self.num_classes)
        # Split predictions
        boxes = x[..., :4]  # tx, ty, tw, th
        obj = x[..., 4:5]   # Objectness
        cls = x[..., 5:]    # Class probabilities
        # Apply activations
        boxes[..., 0:2] = torch.sigmoid(boxes[..., 0:2])  # x, y
        obj = torch.sigmoid(obj)  # Objectness score
        cls = torch.sigmoid(cls)  # Class probabilities
        return boxes, obj, cls

# Example usage with dummy backbone and neck
class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)  # 416x416 -> 208x208
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 208x208 -> 104x104
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # 104x104 -> 52x52
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # 52x52 -> 26x26
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # 26x26 -> 13x13

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.leaky_relu(self.conv3(x), 0.1)
        x = F.leaky_relu(self.conv4(x), 0.1)
        x = F.leaky_relu(self.conv5(x), 0.1)
        return [x]  # Single scale for simplicity

class SimpleNeck(nn.Module):
    def __init__(self):
        super(SimpleNeck, self).__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, features):
        return [F.leaky_relu(self.conv(f), 0.1) for f in features]

# Combine components
backbone = SimpleBackbone()
neck = SimpleNeck()
head = YOLOv3Head(num_anchors=3, num_classes=80, in_channels=256)
input_img = torch.randn(1, 3, 416, 416)
features = backbone(input_img)
fused_features = neck(features)
boxes, obj, cls = head(fused_features[0])
print(f"Boxes shape: {boxes.shape}, Objectness shape: {obj.shape}, Classes shape: {cls.shape}")
