import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class YOLOv2(nn.Module):
    def __init__(self, S=13, B=5, C=20, anchors=None):
        super(YOLOv2, self).__init__()
        self.S = S  # Grid size (13x13)
        self.B = B  # Anchors per cell
        self.C = C  # Number of classes
        # Example anchors (width, height) from k-means on VOC
        self.anchors = anchors if anchors is not None else torch.tensor([
            [1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]
        ])
        # Simplified Darknet-19 backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 416x416 -> 208x208
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 208x208 -> 104x104
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, stride=2)  # 104x104 -> 52x52
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, stride=2)  # 52x52 -> 26x26
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, stride=2)  # 26x26 -> 13x13
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        # Passthrough layer (simplified)
        self.passthrough = nn.Conv2d(256, 64, kernel_size=1)  # 26x26 features
        # Head
        self.head = nn.Conv2d(1088, B * (5 + C), kernel_size=1)  # 1024 + 64 from passthrough

    def forward(self, x):
        # Backbone with batch normalization
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = self.pool1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = self.pool2(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = self.pool3(x)
        x4 = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)  # 26x26 for passthrough
        x = self.pool4(x4)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1)
        x = self.pool5(x)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.1)  # 13x13x1024
        # Passthrough: Concatenate 26x26 features
        pass_x = self.passthrough(x4)  # 26x26x64
        pass_x = pass_x.view(-1, 64, 26 // 2, 2, 26 // 2, 2).permute(0, 1, 3, 5, 2, 4).contiguous()
        pass_x = pass_x.view(-1, 64 * 4, 13, 13)  # 13x13x256
        x = torch.cat([x, pass_x], dim=1)  # 13x13x1088
        # Head
        x = self.head(x)  # 13x13x(5*(5+20))
        x = x.view(-1, self.S, self.S, self.B, 5 + self.C)
        # Split predictions
        boxes = x[..., :5]
        cls = x[..., 5:]
        boxes[..., 0:2] = torch.sigmoid(boxes[..., 0:2])  # x, y
        boxes[..., 4:5] = torch.sigmoid(boxes[..., 4:5])  # Objectness
        cls = torch.sigmoid(cls)  # Class probabilities
        return boxes, cls

# Simplified loss (squared error, no IoU for brevity)
def yolov2_loss(preds, targets):
    boxes_pred, cls_pred = preds
    boxes_gt, cls_gt = targets
    obj_mask = boxes_gt[..., 4] > 0
    coord_loss = 5.0 * torch.sum((boxes_pred[obj_mask, :4] - boxes_gt[obj_mask, :4]) ** 2)
    obj_loss = torch.sum((boxes_pred[obj_mask, 4] - boxes_gt[obj_mask, 4]) ** 2)
    noobj_loss = 0.5 * torch.sum((boxes_pred[~obj_mask, 4] - boxes_gt[~obj_mask, 4]) ** 2)
    cls_loss = torch.sum((cls_pred[obj_mask] - cls_gt[obj_mask]) ** 2)
    return coord_loss + obj_loss + noobj_loss + cls_loss

# Multi-scale training simulation
def multi_scale_input(batch, sizes=[320, 416, 608]):
    size = random.choice(sizes)
    return F.interpolate(batch, size=(size, size), mode='bilinear', align_corners=False)

# Example usage
model = YOLOv2(S=13, B=5, C=20)
input_img = torch.randn(1, 3, 416, 416)
input_img = multi_scale_input(input_img)  # Simulate multi-scale
boxes, cls = model(input_img)
print(f"Boxes shape: {boxes.shape}, Classes shape: {cls.shape}")
# Dummy targets
targets = (
    torch.randn(1, 13, 13, 5, 5),  # boxes (x, y, w, h, objectness)
    torch.randn(1, 13, 13, 20)     # classes
)
loss = yolov2_loss((boxes, cls), targets)
print("Loss:", loss.item())
