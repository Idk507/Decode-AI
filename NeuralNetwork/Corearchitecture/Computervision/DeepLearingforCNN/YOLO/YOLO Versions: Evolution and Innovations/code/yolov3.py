import torch
import torch.nn as nn
import torch.nn.functional as F

class Darknet53Backbone(nn.Module):
    def __init__(self):
        super(Darknet53Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 416x416 -> 208x208
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 208x208 -> 104x104
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 104x104 -> 52x52
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 52x52 -> 26x26
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # 26x26 -> 13x13
        self.bn6 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        C3 = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)  # 52x52
        C4 = F.leaky_relu(self.bn5(self.conv5(C3)), 0.1)  # 26x26
        C5 = F.leaky_relu(self.bn6(self.conv6(C4)), 0.1)  # 13x13
        return [C3, C4, C5]

class FPNNeck(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024]):
        super(FPNNeck, self).__init__()
        self.lateral1 = nn.Conv2d(in_channels[2], 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels[1], 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels[0], 256, kernel_size=1)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, features):
        C3, C4, C5 = features
        P5 = F.leaky_relu(self.lateral1(C5), 0.1)  # 13x13
        P4 = F.leaky_relu(self.lateral2(C4), 0.1) + F.interpolate(P5, scale_factor=2, mode='nearest')  # 26x26
        P3 = F.leaky_relu(self.lateral3(C3), 0.1) + F.interpolate(P4, scale_factor=2, mode='nearest')  # 52x52
        P5 = F.leaky_relu(self.smooth1(P5), 0.1)
        P4 = F.leaky_relu(self.smooth2(P4), 0.1)
        P3 = F.leaky_relu(self.smooth3(P3), 0.1)
        return [P3, P4, P5]

class YOLOv3Head(nn.Module):
    def __init__(self, num_anchors=3, num_classes=80, in_channels=256):
        super(YOLOv3Head, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        batch_size, _, S, _ = x.shape
        x = x.view(batch_size, S, S, self.num_anchors, 5 + self.num_classes)
        boxes = x[..., :4]
        obj = x[..., 4:5]
        cls = x[..., 5:]
        boxes[..., 0:2] = torch.sigmoid(boxes[..., 0:2])  # x, y
        obj = torch.sigmoid(obj)  # Objectness
        cls = torch.sigmoid(cls)  # Classes
        return boxes, obj, cls

class YOLOv3(nn.Module):
    def __init__(self, S=[13, 26, 52], num_anchors=3, num_classes=80):
        super(YOLOv3, self).__init__()
        self.S = S
        self.backbone = Darknet53Backbone()
        self.neck = FPNNeck(in_channels=[256, 512, 1024])
        self.heads = nn.ModuleList([
            YOLOv3Head(num_anchors, num_classes, 256) for _ in range(3)
        ])

    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.neck(features)
        outputs = [self.heads[i](fused_features[i]) for i in range(3)]
        return outputs  # List of (boxes, obj, cls) for each scale

# Simplified loss (BCE for simplicity)
def yolov3_loss(preds, targets):
    loss = 0
    for (boxes_pred, obj_pred, cls_pred), (boxes_gt, obj_gt, cls_gt) in zip(preds, targets):
        obj_mask = obj_gt[..., 0] > 0
        coord_loss = 5.0 * torch.sum((boxes_pred[obj_mask] - boxes_gt[obj_mask]) ** 2)
        obj_loss = F.binary_cross_entropy(obj_pred, obj_gt, reduction='sum')
        cls_loss = F.binary_cross_entropy(cls_pred[obj_mask], cls_gt[obj_mask], reduction='sum')
        loss += coord_loss + obj_loss + cls_loss
    return loss

# Example usage
model = YOLOv3(S=[13, 26, 52], num_anchors=3, num_classes=80)
input_img = torch.randn(1, 3, 416, 416)
outputs = model(input_img)
for i, (boxes, obj, cls) in enumerate(outputs):
    print(f"Scale {i+1}: Boxes shape: {boxes.shape}, Objectness shape: {obj.shape}, Classes shape: {cls.shape}")
# Dummy targets
targets = [
    (torch.randn(1, S, S, 3, 4), torch.randn(1, S, S, 3, 1), torch.randn(1, S, S, 80))
    for S in [13, 26, 52]
]
loss = yolov3_loss(outputs, targets)
print("Loss:", loss.item())
