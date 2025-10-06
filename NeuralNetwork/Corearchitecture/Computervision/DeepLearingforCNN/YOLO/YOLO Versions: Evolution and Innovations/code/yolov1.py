import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S  # Grid size (7x7)
        self.B = B  # Boxes per cell
        self.C = C  # Number of classes
        # Simplified backbone (subset of 24-layer Darknet)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 448x448 -> 224x224
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 224x224 -> 112x112
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)  # 112x112
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 112x112 -> 56x56
        self.conv3 = nn.Conv2d(192, 512, kernel_size=3, stride=1, padding=1)  # 56x56
        self.pool3 = nn.MaxPool2d(2, stride=2)  # 56x56 -> 28x28
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)  # 28x28
        self.pool4 = nn.MaxPool2d(2, stride=2)  # 28x28 -> 14x14
        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)  # 14x14 -> 7x7
        # Head: Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 1024, 4096)
        self.fc2 = nn.Linear(4096, S * S * (B * 5 + C))  # 7x7x30

    def forward(self, x):
        # Backbone
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = self.pool2(x)
        x = F.leaky_relu(self.conv3(x), 0.1)
        x = self.pool3(x)
        x = F.leaky_relu(self.conv4(x), 0.1)
        x = self.pool4(x)
        x = F.leaky_relu(self.conv5(x), 0.1)  # 7x7x1024
        # Head
        x = x.view(x.size(0), -1)  # Flatten
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.fc2(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)  # 7x7x30
        # Split predictions
        boxes = x[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5)
        boxes[..., 0:2] = torch.sigmoid(boxes[..., 0:2])  # x, y
        boxes[..., 4:5] = torch.sigmoid(boxes[..., 4:5])  # Objectness
        cls = torch.softmax(x[..., self.B * 5:], dim=-1)  # Class probabilities
        return boxes, cls

# Simplified loss (squared error, no IoU for brevity)
def yolov1_loss(preds, targets):
    boxes_pred, cls_pred = preds
    boxes_gt, cls_gt = targets
    obj_mask = boxes_gt[..., 4] > 0
    # Localization loss
    coord_loss = 5.0 * torch.sum((boxes_pred[obj_mask, :4] - boxes_gt[obj_mask, :4]) ** 2)
    # Objectness loss
    obj_loss = torch.sum((boxes_pred[obj_mask, 4] - boxes_gt[obj_mask, 4]) ** 2)
    noobj_loss = 0.5 * torch.sum((boxes_pred[~obj_mask, 4] - boxes_gt[~obj_mask, 4]) ** 2)
    # Classification loss
    cls_loss = torch.sum((cls_pred[obj_mask] - cls_gt[obj_mask]) ** 2)
    return coord_loss + obj_loss + noobj_loss + cls_loss

# Example usage
model = YOLOv1(S=7, B=2, C=20)
input_img = torch.randn(1, 3, 448, 448)
boxes, cls = model(input_img)
print(f"Boxes shape: {boxes.shape}, Classes shape: {cls.shape}")
# Dummy targets
targets = (
    torch.randn(1, 7, 7, 2, 5),  # boxes (x, y, w, h, objectness)
    torch.randn(1, 7, 7, 20)     # classes
)
loss = yolov1_loss((boxes, cls), targets)
print("Loss:", loss.item())
