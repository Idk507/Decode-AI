import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels // 2)

    def forward(self, x):
        x1 = F.mish(self.bn1(self.conv1(x)))
        x1 = F.mish(self.bn2(self.conv2(x1)))
        x2 = F.mish(self.bn3(self.conv3(x)))
        return torch.cat([x1, x2], dim=1)

class CSPDarknet53(nn.Module):
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 416x416 -> 208x208
        self.bn2 = nn.BatchNorm2d(64)
        self.csp1 = CSPBlock(64, 128)  # 208x208
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 208x208 -> 104x104
        self.bn3 = nn.BatchNorm2d(256)
        self.csp2 = CSPBlock(256, 512)  # 104x104 -> 52x52
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # 52x52 -> 26x26
        self.bn4 = nn.BatchNorm2d(512)
        self.csp3 = CSPBlock(512, 1024)  # 26x26 -> 13x13
        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)  # 13x13
        self.bn5 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = F.mish(self.bn1(self.conv1(x)))
        x = F.mish(self.bn2(self.conv2(x)))
        C3 = self.csp1(x)  # 208x208
        x = F.mish(self.bn3(self.conv3(C3)))
        C4 = self.csp2(x)  # 52x52
        x = F.mish(self.bn4(self.conv4(C4)))
        C5 = self.csp3(x)  # 26x26
        C5 = F.mish(self.bn5(self.conv5(C5)))  # 13x13
        return [C3, C4, C5]

class SPP(nn.Module):
    def __init__(self, in_channels):
        super(SPP, self).__init__()
        self.pool1 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(13, stride=1, padding=6)
        self.conv = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = x
        x2 = self.pool1(x)
        x3 = self.pool2(x)
        x4 = self.pool3(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return F.mish(self.conv(x))

class PANet(nn.Module):
    def __init__(self, in_channels=[128, 512, 1024]):
        super(PANet, self).__init__()
        self.spp = SPP(in_channels[2])
        self.lateral1 = nn.Conv2d(in_channels[2], 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels[1], 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels[0], 256, kernel_size=1)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # Bottom-up
        self.down1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, features):
        C3, C4, C5 = features
        # SPP on deepest features
        P5 = self.spp(C5)
        P5 = F.mish(self.lateral1(P5))  # 13x13
        # Top-down
        P4 = F.mish(self.lateral2(C4)) + F.interpolate(P5, scale_factor=2, mode='nearest')  # 26x26
        P3 = F.mish(self.lateral3(C3)) + F.interpolate(P4, scale_factor=2, mode='nearest')  # 52x52
        # Bottom-up
        P4 = F.mish(self.smooth2(P4)) + F.mish(self.down1(P3))  # 26x26
        P5 = F.mish(self.smooth1(P5)) + F.mish(self.down2(P4))  # 13x13
        P3 = F.mish(self.smooth3(P3))  # 52x52
        return [P3, P4, P5]

class YOLOv4Head(nn.Module):
    def __init__(self, num_anchors=3, num_classes=80, in_channels=256):
        super(YOLOv4Head, self).__init__()
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

class YOLOv4(nn.Module):
    def __init__(self, S=[13, 26, 52], num_anchors=3, num_classes=80):
        super(YOLOv4, self).__init__()
        self.S = S
        self.backbone = CSPDarknet53()
        self.neck = PANet(in_channels=[128, 512, 1024])
        self.heads = nn.ModuleList([
            YOLOv4Head(num_anchors, num_classes, 256) for _ in range(3)
        ])

    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.neck(features)
        outputs = [self.heads[i](fused_features[i]) for i in range(3)]
        return outputs

# Simplified Mosaic augmentation
def mosaic_augmentation(images, size=416):
    batch_size, _, h, w = images.shape
    mosaic = torch.zeros(batch_size, 3, size, size)
    for i in range(batch_size):
        img1, img2, img3, img4 = [F.interpolate(images[i:i+1], size=(size//2, size//2)) for _ in range(4)]
        mosaic[i, :, 0:size//2, 0:size//2] = img1
        mosaic[i, :, 0:size//2, size//2:size] = img2
        mosaic[i, :, size//2:size, 0:size//2] = img3
        mosaic[i, :, size//2:size, size//2:size] = img4
    return mosaic

# Simplified loss (BCE + CIoU placeholder)
def yolov4_loss(preds, targets):
    loss = 0
    for (boxes_pred, obj_pred, cls_pred), (boxes_gt, obj_gt, cls_gt) in zip(preds, targets):
        obj_mask = obj_gt[..., 0] > 0
        coord_loss = 5.0 * torch.sum((boxes_pred[obj_mask] - boxes_gt[obj_mask]) ** 2)  # Placeholder for CIoU
        obj_loss = F.binary_cross_entropy(obj_pred, obj_gt, reduction='sum')
        cls_loss = F.binary_cross_entropy(cls_pred[obj_mask], cls_gt[obj_mask], reduction='sum')
        loss += coord_loss + obj_loss + cls_loss
    return loss

# Example usage
model = YOLOv4(S=[13, 26, 52], num_anchors=3, num_classes=80)
input_img = torch.randn(4, 3, 416, 416)
input_img = mosaic_augmentation(input_img)  # Apply Mosaic
outputs = model(input_img)
for i, (boxes, obj, cls) in enumerate(outputs):
    print(f"Scale {i+1}: Boxes shape: {boxes.shape}, Objectness shape: {obj.shape}, Classes shape: {cls.shape}")
# Dummy targets
targets = [
    (torch.randn(4, S, S, 3, 4), torch.randn(4, S, S, 3, 1), torch.randn(4, S, S, 80))
    for S in [52, 26, 13]
]
loss = yolov4_loss(outputs, targets)
print("Loss:", loss.item())
